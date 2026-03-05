"""
Telegram AI-агент для управления задачами
Работает полностью на российских сервисах — без VPN и обходов:
  - Транскрипция: Yandex SpeechKit ИЛИ Vosk (локально, бесплатно)
  - Диалог и разбор задач: YandexGPT ИЛИ GigaChat (Сбер)
  - Хранение: SQLite (локально)
  - Уведомления: APScheduler
"""

import json
import logging
import os
import re
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Optional

import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Состояния ConversationHandler для редактирования задачи
EDIT_CHOOSE_FIELD, EDIT_ENTER_VALUE = range(2)
# Состояния для настройки напоминаний
REMIND_ENTER_MINUTES = range(1)[0] + 10  # 10, чтобы не пересекаться

# ──────────────────────────────────────────────
# Конфигурация из переменных окружения
# ──────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# Выбор AI-провайдера: "yandex" или "gigachat"
AI_PROVIDER = os.environ.get("AI_PROVIDER", "yandex").lower()

# Выбор транскрипции: "speechkit" или "vosk"
STT_PROVIDER = os.environ.get("STT_PROVIDER", "speechkit").lower()

# Яндекс Cloud
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "")
YANDEX_SPEECHKIT_KEY = os.environ.get("YANDEX_SPEECHKIT_KEY", "") or YANDEX_API_KEY

# GigaChat (Сбер)
GIGACHAT_CLIENT_ID = os.environ.get("GIGACHAT_CLIENT_ID", "")
GIGACHAT_CLIENT_SECRET = os.environ.get("GIGACHAT_CLIENT_SECRET", "")

# Vosk — путь к скачанной модели
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "./vosk-model-ru")

# Время утреннего дайджеста
DIGEST_HOUR = int(os.environ.get("DIGEST_HOUR", "8"))
DIGEST_MINUTE = int(os.environ.get("DIGEST_MINUTE", "0"))

DB_PATH = os.environ.get("DB_PATH", "tasks.db")

# ──────────────────────────────────────────────
# Логирование
# ──────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# История диалогов в памяти
conversation_history: dict[int, list[dict]] = {}


# ══════════════════════════════════════════════
# БЛОК 1: ТРАНСКРИПЦИЯ ГОЛОСА
# ══════════════════════════════════════════════

def transcribe_speechkit(ogg_path: str) -> str:
    """Yandex SpeechKit — облачное распознавание речи на русском."""
    with open(ogg_path, "rb") as f:
        audio_data = f.read()

    resp = requests.post(
        "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize",
        headers={"Authorization": f"Api-Key {YANDEX_SPEECHKIT_KEY}"},
        params={
            "folderId": YANDEX_FOLDER_ID,
            "lang": "ru-RU",
            "format": "oggopus",
        },
        data=audio_data,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("result", "")


def transcribe_vosk(ogg_path: str) -> str:
    """
    Vosk — локальное распознавание, полностью бесплатно, без интернета.
    Установка: pip install vosk
    Модель: скачай vosk-model-ru-0.42 → https://alphacephei.com/vosk/models
    """
    try:
        import wave
        from vosk import KaldiRecognizer, Model
    except ImportError:
        raise RuntimeError("Vosk не установлен. Запусти: pip install vosk")

    wav_path = ogg_path.replace(".ogg", ".wav")
    # Конвертируем OGG (формат Telegram) в WAV через ffmpeg
    subprocess.run(
        ["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )

    model = Model(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(model, 16000)

    with wave.open(wav_path, "rb") as wf:
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            recognizer.AcceptWaveform(data)

    os.unlink(wav_path)
    return json.loads(recognizer.FinalResult()).get("text", "")


async def transcribe_voice(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Скачиваем голосовое из Telegram и транскрибируем."""
    tg_file = await context.bot.get_file(file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
    await tg_file.download_to_drive(tmp_path)

    try:
        if STT_PROVIDER == "vosk":
            return transcribe_vosk(tmp_path)
        else:
            return transcribe_speechkit(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ══════════════════════════════════════════════
# БЛОК 2: AI-ДИАЛОГ (YandexGPT / GigaChat)
# ══════════════════════════════════════════════

# ── GigaChat (Сбер) ───────────────────────────

_gigachat_token: Optional[str] = None
_gigachat_token_expires: float = 0.0


def _get_gigachat_token() -> str:
    """Получаем/обновляем OAuth токен GigaChat."""
    import base64
    import time

    global _gigachat_token, _gigachat_token_expires
    if _gigachat_token and time.time() < _gigachat_token_expires:
        return _gigachat_token

    credentials = base64.b64encode(
        f"{GIGACHAT_CLIENT_ID}:{GIGACHAT_CLIENT_SECRET}".encode()
    ).decode()

    resp = requests.post(
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
            "RqUID": "a1b2c3d4-0000-0000-0000-000000000001",
        },
        data={"scope": "GIGACHAT_API_PERS"},
        verify=False,  # Сбер использует собственный CA-сертификат
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    _gigachat_token = data["access_token"]
    # expires_at приходит в миллисекундах
    _gigachat_token_expires = time.time() + data.get("expires_at", 1_800_000) / 1000 - 60
    return _gigachat_token


def gigachat_complete(messages: list[dict], system: str = "") -> str:
    token = _get_gigachat_token()
    payload = []
    if system:
        payload.append({"role": "system", "content": system})
    payload.extend(messages)

    resp = requests.post(
        "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": "GigaChat", "messages": payload, "max_tokens": 1000, "temperature": 0.7},
        verify=False,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── YandexGPT ─────────────────────────────────

def yandexgpt_complete(messages: list[dict], system: str = "") -> str:
    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "text": system})
    for m in messages:
        payload_messages.append({"role": m["role"], "text": m["content"]})

    resp = requests.post(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        headers={
            "Authorization": f"Api-Key {YANDEX_API_KEY}",
            "x-folder-id": YANDEX_FOLDER_ID,
        },
        json={
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest",
            "completionOptions": {"stream": False, "temperature": 0.6, "maxTokens": "1000"},
            "messages": payload_messages,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["result"]["alternatives"][0]["message"]["text"]


# ── Единая точка входа ────────────────────────

def ai_complete(messages: list[dict], system: str = "") -> str:
    if AI_PROVIDER == "gigachat":
        return gigachat_complete(messages, system)
    return yandexgpt_complete(messages, system)


# ── Разбор задачи и определение намерения ─────

def parse_task(text: str, remind_default: int = 30) -> dict:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    prompt = (
        f"Сейчас: {now_str}.\n"
        f'Пользователь написал задачу: "{text}"\n\n'
        "Извлеки структуру задачи. Ответь ТОЛЬКО JSON без пояснений и markdown:\n"
        "{\n"
        '  "task": "краткое описание задачи на русском",\n'
        '  "due_time": "YYYY-MM-DD HH:MM" или null,\n'
        f'  "remind_minutes_before": целое число ({remind_default} по умолчанию, 0 если нет срока)\n'
        "}\n\n"
        "Примеры:\n"
        '- "позвони маме завтра в 18:00" → due_time=завтра 18:00, remind=30\n'
        '- "купить молоко" → due_time=null, remind=0\n'
        '- "встреча в пятницу в 10, напомни за час" → due_time=пятница 10:00, remind=60'
    )
    try:
        raw = ai_complete([{"role": "user", "content": prompt}])
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Ошибка разбора задачи: {e}")
        return {"task": text, "due_time": None, "remind_minutes_before": 0}


def detect_intent(text: str) -> str:
    """TASK — задача для добавления, CHAT — обычный разговор."""
    prompt = (
        f'Пользователь написал: "{text}"\n'
        "Это задача/дело/напоминание которое нужно выполнить — ответь TASK.\n"
        "Это вопрос, разговор или обсуждение — ответь CHAT.\n"
        "Ответь ТОЛЬКО одним словом: TASK или CHAT."
    )
    try:
        result = ai_complete([{"role": "user", "content": prompt}]).strip().upper()
        return "TASK" if "TASK" in result else "CHAT"
    except Exception:
        return "CHAT"


async def chat_reply(user_id: int, user_message: str) -> str:
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"role": "user", "content": user_message})
    history = conversation_history[user_id][-20:]

    system = (
        "Ты умный и дружелюбный AI-ассистент в Telegram. "
        "Помогаешь управлять задачами и поддерживаешь разговор на любую тему. "
        "Отвечай всегда на русском языке. Будь кратким, но содержательным."
    )
    try:
        reply = ai_complete(history, system=system)
    except Exception as e:
        logger.error(f"Ошибка AI: {e}")
        reply = "Извини, произошла ошибка при обращении к AI. Попробуй ещё раз."

    conversation_history[user_id].append({"role": "assistant", "content": reply})
    return reply


# ══════════════════════════════════════════════
# БЛОК 3: БАЗА ДАННЫХ (SQLite)
# ══════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            text        TEXT    NOT NULL,
            due_time    TEXT,
            remind_at   TEXT,
            reminded    INTEGER DEFAULT 0,
            done        INTEGER DEFAULT 0,
            created_at  TEXT    DEFAULT (datetime('now', 'localtime'))
        )
    """)
    # Настройки пользователя (напр. remind_default — минут по умолчанию)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id         INTEGER PRIMARY KEY,
            remind_default  INTEGER DEFAULT 30
        )
    """)
    conn.commit()
    conn.close()


# ── Настройки пользователя ────────────────────

def db_get_remind_default(user_id: int) -> int:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT remind_default FROM user_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else 30


def db_set_remind_default(user_id: int, minutes: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO user_settings (user_id, remind_default) VALUES (?, ?)
           ON CONFLICT(user_id) DO UPDATE SET remind_default = excluded.remind_default""",
        (user_id, minutes),
    )
    conn.commit()
    conn.close()


# ── Редактирование задачи ─────────────────────

def db_get_task(task_id: int, user_id: int) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM tasks WHERE id = ? AND user_id = ? AND done = 0", (task_id, user_id)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def db_update_task(task_id: int, user_id: int, **fields) -> bool:
    """Обновляет указанные поля задачи. Автоматически сбрасывает reminded при смене remind_at."""
    if not fields:
        return False
    if "remind_at" in fields:
        fields["reminded"] = 0  # сброс флага — чтобы напомнило заново
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [task_id, user_id]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        f"UPDATE tasks SET {set_clause} WHERE id = ? AND user_id = ? AND done = 0",
        values,
    )
    ok = c.rowcount > 0
    conn.commit()
    conn.close()
    return ok


def db_add_task(user_id: int, text: str, due_time: Optional[str], remind_at: Optional[str]) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO tasks (user_id, text, due_time, remind_at) VALUES (?, ?, ?, ?)",
        (user_id, text, due_time, remind_at),
    )
    task_id = c.lastrowid
    conn.commit()
    conn.close()
    return task_id


def db_get_pending_reminders() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = conn.execute(
        "SELECT * FROM tasks WHERE remind_at <= ? AND reminded = 0 AND done = 0", (now,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_mark_reminded(task_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE tasks SET reminded = 1 WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()


def db_get_today_tasks(user_id: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    today = datetime.now().strftime("%Y-%m-%d")
    rows = conn.execute(
        """SELECT * FROM tasks
           WHERE user_id = ? AND done = 0
             AND (due_time LIKE ? OR due_time IS NULL)
           ORDER BY due_time""",
        (user_id, f"{today}%"),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_get_active_tasks(user_id: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tasks WHERE user_id = ? AND done = 0 ORDER BY due_time NULLS LAST",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_complete_task(task_id: int, user_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE tasks SET done = 1 WHERE id = ? AND user_id = ?", (task_id, user_id))
    ok = c.rowcount > 0
    conn.commit()
    conn.close()
    return ok


def db_all_user_ids() -> list[int]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT user_id FROM tasks WHERE done = 0"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ══════════════════════════════════════════════
# БЛОК 4: TELEGRAM ХЭНДЛЕРЫ
# ══════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ai_name = "GigaChat (Сбер)" if AI_PROVIDER == "gigachat" else "YandexGPT (Яндекс)"
    stt_name = "Vosk (локально)" if STT_PROVIDER == "vosk" else "Yandex SpeechKit"
    await update.message.reply_text(
        "👋 *Привет! Я твой AI-помощник по задачам.*\n\n"
        "Напиши или скажи голосом что нужно сделать:\n"
        "  • «Позвони врачу завтра в 15:00»\n"
        "  • «Встреча в пятницу в 10, напомни за час»\n"
        "  • «Купить продукты»\n\n"
        "📋 /tasks — список активных задач\n"
        "✅ /done [id] — отметить выполненной\n"
        "✏️ /edit [id] — редактировать задачу\n"
        "🔔 /remindset — настроить напоминание по умолчанию\n"
        "🗑 /clear — очистить историю диалога\n"
        "ℹ️ /status — текущие настройки\n\n"
        "💬 Или просто поговори со мной на любую тему!\n\n"
        f"_🤖 {ai_name}  |  🎙 {stt_name}_",
        parse_mode="Markdown",
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    ai_name = "GigaChat (Сбер)" if AI_PROVIDER == "gigachat" else "YandexGPT (Яндекс)"
    stt_name = "Vosk (локально, бесплатно)" if STT_PROVIDER == "vosk" else "Yandex SpeechKit"
    remind_min = db_get_remind_default(user_id)
    await update.message.reply_text(
        f"⚙️ *Настройки бота:*\n\n"
        f"🤖 AI: {ai_name}\n"
        f"🎙 Речь: {stt_name}\n"
        f"🔔 Напоминание по умолч.: {_minutes_label(remind_min)}\n"
        f"☀️ Дайджест: {DIGEST_HOUR:02d}:{DIGEST_MINUTE:02d}\n"
        f"🗄 БД: `{DB_PATH}`\n\n"
        "Изменить напоминание: /remindset",
        parse_mode="Markdown",
    )


async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tasks = db_get_active_tasks(update.effective_user.id)
    if not tasks:
        await update.message.reply_text("✅ Нет активных задач — всё чисто!")
        return
    lines = ["📋 *Активные задачи:*\n"]
    for t in tasks:
        due = f"\n    ⏰ {t['due_time'][:16]}" if t["due_time"] else ""
        lines.append(f"*[{t['id']}]* {t['text']}{due}")
    lines.append(f"\n_Итого: {len(tasks)} задач_")
    lines.append("Напиши /done [id] чтобы завершить")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or not args[0].isdigit():
        await update.message.reply_text("Используй: /done [номер]\nПример: /done 3\nСписок: /tasks")
        return
    task_id = int(args[0])
    if db_complete_task(task_id, update.effective_user.id):
        await update.message.reply_text(f"✅ Задача #{task_id} выполнена! Молодец 💪")
    else:
        await update.message.reply_text("❌ Задача не найдена. Проверь номер: /tasks")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conversation_history.pop(update.effective_user.id, None)
    await update.message.reply_text("🧹 История диалога очищена.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if detect_intent(text) == "TASK":
        await _process_task(update, text)
    else:
        await update.message.reply_chat_action("typing")
        reply = await chat_reply(update.effective_user.id, text)
        await update.message.reply_text(reply)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎙 Распознаю голосовое...")
    try:
        text = await transcribe_voice(update.message.voice.file_id, context)
        if not text:
            await update.message.reply_text("❌ Не смог разобрать речь. Попробуй чётче.")
            return
        await update.message.reply_text(f"📝 *Распознал:* _{text}_", parse_mode="Markdown")
        update.message.text = text
        await handle_text(update, context)
    except Exception as e:
        logger.error(f"Ошибка транскрипции: {e}")
        await update.message.reply_text("❌ Ошибка распознавания. Напиши задачу текстом.")


async def _process_task(update: Update, text: str):
    await update.message.reply_chat_action("typing")
    user_id = update.effective_user.id
    remind_default = db_get_remind_default(user_id)
    parsed = parse_task(text, remind_default=remind_default)

    task_text = parsed.get("task", text)
    due_time = parsed.get("due_time")
    remind_min = int(parsed.get("remind_minutes_before", remind_default))

    remind_at = None
    if due_time and remind_min > 0:
        try:
            due_dt = datetime.strptime(due_time, "%Y-%m-%d %H:%M")
            remind_at = (due_dt - timedelta(minutes=remind_min)).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass

    task_id = db_add_task(user_id, task_text, due_time, remind_at)

    lines = [f"✅ *Задача добавлена* (#{task_id})", f"📌 {task_text}"]
    if due_time:
        lines.append(f"⏰ Срок: {due_time}")
        lines.append(f"🔔 Напомню: {remind_at}" if remind_at else "")
    else:
        lines.append("_(без конкретного срока)_")

    await update.message.reply_text("\n".join(l for l in lines if l), parse_mode="Markdown")


# ══════════════════════════════════════════════
# БЛОК 4б: РЕДАКТИРОВАНИЕ ЗАДАЧ
# ══════════════════════════════════════════════

async def cmd_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /edit [id] — начать редактирование задачи.
    Если id не указан — показывает список для выбора.
    """
    user_id = update.effective_user.id
    args = context.args

    if args and args[0].isdigit():
        task_id = int(args[0])
    else:
        # Показываем список задач с кнопками
        tasks = db_get_active_tasks(user_id)
        if not tasks:
            await update.message.reply_text("✅ Нет активных задач для редактирования.")
            return ConversationHandler.END

        keyboard = [
            [InlineKeyboardButton(
                f"[{t['id']}] {t['text'][:35]}{'…' if len(t['text']) > 35 else ''}",
                callback_data=f"edit_select:{t['id']}"
            )]
            for t in tasks
        ]
        await update.message.reply_text(
            "✏️ *Выбери задачу для редактирования:*",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )
        return EDIT_CHOOSE_FIELD

    return await _show_edit_menu(update, context, task_id)


async def _show_edit_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, task_id: int):
    """Показываем меню выбора поля для редактирования."""
    user_id = update.effective_user.id
    task = db_get_task(task_id, user_id)
    if not task:
        msg = update.message or update.callback_query.message
        await msg.reply_text("❌ Задача не найдена или уже выполнена.")
        return ConversationHandler.END

    context.user_data["editing_task_id"] = task_id

    due_str = task["due_time"][:16] if task["due_time"] else "не задан"
    remind_str = task["remind_at"][:16] if task["remind_at"] else "нет"

    keyboard = [
        [InlineKeyboardButton("📝 Текст задачи", callback_data="edit_field:text")],
        [InlineKeyboardButton("⏰ Срок выполнения", callback_data="edit_field:due_time")],
        [InlineKeyboardButton("🔔 Время напоминания", callback_data="edit_field:remind_at")],
        [InlineKeyboardButton("❌ Отмена", callback_data="edit_field:cancel")],
    ]
    text = (
        f"✏️ *Редактирование задачи #{task_id}*\n\n"
        f"📌 Текст: {task['text']}\n"
        f"⏰ Срок: {due_str}\n"
        f"🔔 Напоминание: {remind_str}\n\n"
        "Что хочешь изменить?"
    )
    msg = update.message or update.callback_query.message
    await msg.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    return EDIT_CHOOSE_FIELD


async def edit_callback_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь выбрал задачу из списка."""
    query = update.callback_query
    await query.answer()
    task_id = int(query.data.split(":")[1])
    return await _show_edit_menu(update, context, task_id)


async def edit_callback_field(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь выбрал поле для редактирования."""
    query = update.callback_query
    await query.answer()
    field = query.data.split(":")[1]

    if field == "cancel":
        await query.message.reply_text("❌ Редактирование отменено.")
        return ConversationHandler.END

    context.user_data["editing_field"] = field

    prompts = {
        "text": (
            "📝 Введи новый текст задачи:"
        ),
        "due_time": (
            "⏰ Введи новый срок выполнения:\n"
            "Примеры: _завтра в 15:00_, _пятница 10:00_, _2025-12-31 18:00_\n"
            "Или напиши *убрать* чтобы снять срок."
        ),
        "remind_at": (
            "🔔 Введи новое время напоминания:\n"
            "Примеры: _за 30 минут_, _за 2 часа_, _за день_\n"
            "Или укажи точное время: _2025-12-31 17:30_\n"
            "Или напиши *убрать* чтобы снять напоминание."
        ),
    }
    await query.message.reply_text(prompts[field], parse_mode="Markdown")
    return EDIT_ENTER_VALUE


async def edit_receive_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получаем новое значение и сохраняем."""
    user_id = update.effective_user.id
    task_id = context.user_data.get("editing_task_id")
    field = context.user_data.get("editing_field")
    value_raw = update.message.text.strip()

    if not task_id or not field:
        await update.message.reply_text("❌ Что-то пошло не так. Начни заново: /edit")
        return ConversationHandler.END

    task = db_get_task(task_id, user_id)
    if not task:
        await update.message.reply_text("❌ Задача не найдена.")
        return ConversationHandler.END

    # ── Обрабатываем значение в зависимости от поля ──
    if field == "text":
        db_update_task(task_id, user_id, text=value_raw)
        await update.message.reply_text(
            f"✅ Текст задачи #{task_id} обновлён:\n📌 {value_raw}"
        )

    elif field == "due_time":
        if value_raw.lower() in ("убрать", "удалить", "нет", "-"):
            db_update_task(task_id, user_id, due_time=None, remind_at=None)
            await update.message.reply_text(f"✅ Срок задачи #{task_id} снят.")
        else:
            # Просим AI распарсить дату
            new_due = _parse_datetime_with_ai(value_raw)
            if not new_due:
                await update.message.reply_text(
                    "❌ Не смог распознать дату. Попробуй в формате: _завтра в 15:00_ или _2025-12-31 15:00_",
                    parse_mode="Markdown",
                )
                return EDIT_ENTER_VALUE  # Остаёмся в том же состоянии

            # Пересчитываем remind_at если было
            remind_at = task.get("remind_at")
            if task.get("due_time") and remind_at:
                try:
                    old_due = datetime.strptime(task["due_time"], "%Y-%m-%d %H:%M")
                    old_remind = datetime.strptime(remind_at, "%Y-%m-%d %H:%M")
                    delta = old_due - old_remind  # разница между сроком и напоминанием
                    new_due_dt = datetime.strptime(new_due, "%Y-%m-%d %H:%M")
                    remind_at = (new_due_dt - delta).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    remind_at = None

            db_update_task(task_id, user_id, due_time=new_due, remind_at=remind_at)
            msg = f"✅ Срок задачи #{task_id} обновлён: ⏰ {new_due}"
            if remind_at:
                msg += f"\n🔔 Напоминание автоматически сдвинуто на: {remind_at}"
            await update.message.reply_text(msg)

    elif field == "remind_at":
        if value_raw.lower() in ("убрать", "удалить", "нет", "-"):
            db_update_task(task_id, user_id, remind_at=None)
            await update.message.reply_text(f"✅ Напоминание для задачи #{task_id} снято.")
        else:
            new_remind = _parse_remind_with_ai(value_raw, task.get("due_time"))
            if not new_remind:
                await update.message.reply_text(
                    "❌ Не смог распознать время. Попробуй: _за 30 минут_, _за 2 часа_, или _2025-12-31 14:30_",
                    parse_mode="Markdown",
                )
                return EDIT_ENTER_VALUE

            db_update_task(task_id, user_id, remind_at=new_remind)
            await update.message.reply_text(
                f"✅ Напоминание для задачи #{task_id} установлено:\n🔔 {new_remind}"
            )

    context.user_data.pop("editing_task_id", None)
    context.user_data.pop("editing_field", None)
    return ConversationHandler.END


async def edit_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("editing_task_id", None)
    context.user_data.pop("editing_field", None)
    await update.message.reply_text("❌ Редактирование отменено.")
    return ConversationHandler.END


def _parse_datetime_with_ai(text: str) -> Optional[str]:
    """Парсим произвольную дату/время через AI."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    prompt = (
        f"Сейчас: {now_str}.\n"
        f'Пользователь написал: "{text}"\n'
        "Преобразуй в дату и время формата YYYY-MM-DD HH:MM.\n"
        "Ответь ТОЛЬКО строкой в этом формате или словом null если не удалось распознать."
    )
    try:
        result = ai_complete([{"role": "user", "content": prompt}]).strip()
        if result.lower() == "null" or not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result):
            return None
        datetime.strptime(result[:16], "%Y-%m-%d %H:%M")  # валидация
        return result[:16]
    except Exception:
        return None


def _parse_remind_with_ai(text: str, due_time: Optional[str]) -> Optional[str]:
    """Парсим время напоминания — либо 'за N минут/часов', либо конкретное время."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    due_info = f"Срок задачи: {due_time}." if due_time else "Срок не задан."
    prompt = (
        f"Сейчас: {now_str}. {due_info}\n"
        f'Пользователь хочет напоминание: "{text}"\n'
        "Вычисли точное время напоминания в формате YYYY-MM-DD HH:MM.\n"
        "Если написано 'за 30 минут' — вычти 30 минут из срока задачи.\n"
        "Если написано конкретное время — используй его.\n"
        "Ответь ТОЛЬКО строкой YYYY-MM-DD HH:MM или словом null."
    )
    try:
        result = ai_complete([{"role": "user", "content": prompt}]).strip()
        if result.lower() == "null" or not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result):
            return None
        datetime.strptime(result[:16], "%Y-%m-%d %H:%M")
        return result[:16]
    except Exception:
        return None


# ══════════════════════════════════════════════
# БЛОК 4в: НАСТРОЙКА НАПОМИНАНИЙ ПО УМОЛЧАНИЮ
# ══════════════════════════════════════════════

async def cmd_remindset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /remindset — настроить время напоминания по умолчанию.
    Можно сразу: /remindset 60  (за 60 минут)
    """
    user_id = update.effective_user.id
    args = context.args
    current = db_get_remind_default(user_id)

    if args and args[0].isdigit():
        minutes = int(args[0])
        if minutes < 0 or minutes > 10080:  # max 1 неделя
            await update.message.reply_text("❌ Укажи число от 0 до 10080 (минут).")
            return ConversationHandler.END
        db_set_remind_default(user_id, minutes)
        label = _minutes_label(minutes)
        await update.message.reply_text(
            f"✅ Напоминание по умолчанию установлено: *{label}* до срока.\n\n"
            f"_Применится ко всем новым задачам с указанным временем._",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    # Показываем быстрые варианты + текущее значение
    keyboard = [
        [
            InlineKeyboardButton("5 мин", callback_data="remind_set:5"),
            InlineKeyboardButton("15 мин", callback_data="remind_set:15"),
            InlineKeyboardButton("30 мин", callback_data="remind_set:30"),
        ],
        [
            InlineKeyboardButton("1 час", callback_data="remind_set:60"),
            InlineKeyboardButton("2 часа", callback_data="remind_set:120"),
            InlineKeyboardButton("3 часа", callback_data="remind_set:180"),
        ],
        [
            InlineKeyboardButton("Пол-дня (12ч)", callback_data="remind_set:720"),
            InlineKeyboardButton("1 день", callback_data="remind_set:1440"),
        ],
        [InlineKeyboardButton("✏️ Своё значение", callback_data="remind_set:custom")],
    ]
    await update.message.reply_text(
        f"🔔 *Настройка напоминаний по умолчанию*\n\n"
        f"Сейчас: *{_minutes_label(current)}* до срока\n\n"
        "Выбери новое значение или введи своё:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )
    return REMIND_ENTER_MINUTES


async def remindset_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    value = query.data.split(":")[1]
    user_id = update.effective_user.id

    if value == "custom":
        await query.message.reply_text(
            "✏️ Введи количество минут (например: *45* или *90*):",
            parse_mode="Markdown",
        )
        return REMIND_ENTER_MINUTES

    minutes = int(value)
    db_set_remind_default(user_id, minutes)
    await query.message.reply_text(
        f"✅ Напоминание по умолчанию: *{_minutes_label(minutes)}* до срока.",
        parse_mode="Markdown",
    )
    return ConversationHandler.END


async def remindset_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = update.effective_user.id

    if not text.isdigit():
        await update.message.reply_text("❌ Введи целое число минут, например: 45")
        return REMIND_ENTER_MINUTES

    minutes = int(text)
    if minutes < 0 or minutes > 10080:
        await update.message.reply_text("❌ Укажи число от 0 до 10080.")
        return REMIND_ENTER_MINUTES

    db_set_remind_default(user_id, minutes)
    await update.message.reply_text(
        f"✅ Напоминание по умолчанию: *{_minutes_label(minutes)}* до срока.\n\n"
        "_Применится ко всем новым задачам._",
        parse_mode="Markdown",
    )
    return ConversationHandler.END


def _minutes_label(minutes: int) -> str:
    """Человекочитаемое представление минут."""
    if minutes == 0:
        return "без напоминания"
    if minutes < 60:
        return f"{minutes} мин"
    hours = minutes // 60
    mins = minutes % 60
    if mins == 0:
        return f"{hours} ч"
    return f"{hours} ч {mins} мин"




async def check_reminders(app: Application):
    for task in db_get_pending_reminders():
        try:
            due_str = task["due_time"][:16] if task["due_time"] else "скоро"
            await app.bot.send_message(
                chat_id=task["user_id"],
                text=(
                    f"🔔 *Напоминание!*\n\n"
                    f"📌 {task['text']}\n"
                    f"⏰ Срок: {due_str}\n\n"
                    f"Когда выполнишь: /done {task['id']}"
                ),
                parse_mode="Markdown",
            )
            db_mark_reminded(task["id"])
        except Exception as e:
            logger.error(f"Ошибка напоминания {task['id']}: {e}")


async def send_morning_digest(app: Application):
    for user_id in db_all_user_ids():
        tasks = db_get_today_tasks(user_id)
        if not tasks:
            continue
        lines = [f"☀️ *Доброе утро! Задачи на сегодня ({len(tasks)} шт.):*\n"]
        for t in tasks:
            time_str = f"  {t['due_time'][11:16]}" if t["due_time"] else ""
            lines.append(f"• {t['text']}{time_str}")
        lines.append("\n_Хорошего дня! 💪_")
        try:
            await app.bot.send_message(
                chat_id=user_id, text="\n".join(lines), parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Ошибка дайджеста {user_id}: {e}")


# ══════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════

def main():
    init_db()
    logger.info(f"Запуск | AI: {AI_PROVIDER} | STT: {STT_PROVIDER}")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # ConversationHandler: редактирование задач
    edit_handler = ConversationHandler(
        entry_points=[CommandHandler("edit", cmd_edit)],
        states={
            EDIT_CHOOSE_FIELD: [
                CallbackQueryHandler(edit_callback_select, pattern=r"^edit_select:"),
                CallbackQueryHandler(edit_callback_field, pattern=r"^edit_field:"),
            ],
            EDIT_ENTER_VALUE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, edit_receive_value),
            ],
        },
        fallbacks=[CommandHandler("cancel", edit_cancel)],
        per_message=False,
    )

    # ConversationHandler: настройка напоминаний
    remindset_handler = ConversationHandler(
        entry_points=[CommandHandler("remindset", cmd_remindset)],
        states={
            REMIND_ENTER_MINUTES: [
                CallbackQueryHandler(remindset_callback, pattern=r"^remind_set:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, remindset_custom),
            ],
        },
        fallbacks=[CommandHandler("cancel", edit_cancel)],
        per_message=False,
    )

    app.add_handler(edit_handler)
    app.add_handler(remindset_handler)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("tasks", cmd_tasks))
    app.add_handler(CommandHandler("done", cmd_done))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_reminders, "interval", minutes=1, args=[app])
    scheduler.add_job(
        send_morning_digest, "cron",
        hour=DIGEST_HOUR, minute=DIGEST_MINUTE, args=[app]
    )
    scheduler.start()

    logger.info("Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
