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
try:
    import pytz
    def now_local(tz_str="Europe/Moscow"):
        tz = pytz.timezone(tz_str)
        return datetime.now(tz).replace(tzinfo=None)
except ImportError:
    def now_local(tz_str="Europe/Moscow"):
        import time
        offset = {"Europe/Moscow": 3, "Europe/Kaliningrad": 2, "Europe/Samara": 4,
                  "Asia/Yekaterinburg": 5, "Asia/Omsk": 6, "Asia/Krasnoyarsk": 7,
                  "Asia/Irkutsk": 8, "Asia/Yakutsk": 9, "Asia/Vladivostok": 10,
                  "Asia/Magadan": 11, "Asia/Kamchatka": 12}.get(tz_str, 3)
        return datetime.utcnow() + timedelta(hours=offset)

import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup, Update
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
REMIND_ENTER_MINUTES = 10
# Состояния для настроек
SETTINGS_MAIN = 20
SETTINGS_DIGEST_HOUR = 21
SETTINGS_TIMEZONE = 22

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
            "sampleRateHertz": "48000",
        },
        data=audio_data,
        timeout=30,
    )
    logger.info(f"SpeechKit response: {resp.status_code} {resp.text[:200]}")
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
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite/latest",
            "completionOptions": {"stream": False, "temperature": 0.6, "maxTokens": "2000"},
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


# ══════════════════════════════════════════════
# БЛОК: РУССКИЙ ПАРСЕР ВРЕМЕНИ
# Понимает любые формы: "через 10 минут", "послезавтра в 9 утра",
# "в следующую пятницу", "15 декабря в 18:00" и т.д.
# ══════════════════════════════════════════════

WEEKDAYS_RU = {
    "понедельник": 0, "понедельника": 0, "понедельнику": 0,
    "вторник": 1, "вторника": 1, "вторнику": 1,
    "среда": 2, "среду": 2, "среды": 2, "среде": 2,
    "четверг": 3, "четверга": 3, "четвергу": 3,
    "пятница": 4, "пятницу": 4, "пятницы": 4, "пятнице": 4,
    "суббота": 5, "субботу": 5, "субботы": 5, "субботе": 5,
    "воскресенье": 6, "воскресенья": 6, "воскресенью": 6, "воскресенье": 6,
}

MONTHS_RU = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4,
    "ма": 5, "май": 5, "мая": 5, "июн": 6, "июл": 7,
    "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}

NUM_WORDS_RU = {
    "один": 1, "одну": 1, "одной": 1, "одного": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2,
    "три": 3, "трёх": 3, "трех": 3, "трём": 3,
    "четыре": 4, "четырёх": 4, "четырех": 4,
    "пять": 5, "пяти": 5, "шесть": 6, "шести": 6,
    "семь": 7, "семи": 7, "восемь": 8, "восьми": 8,
    "девять": 9, "девяти": 9, "десять": 10, "десяти": 10,
    "пол": 0.5, "полчаса": 0.5, "полчас": 0.5,
    "четверть": 0.25,
    "час": 1, "часа": 1, "часов": 1, "часу": 1,
    "минут": 1, "минуты": 1, "минуту": 1, "минутку": 1,
    "сутки": 24, "суток": 24,
    "неделю": 7, "недели": 7, "неделя": 7, "неделей": 7,
}


def _extract_number(word: str) -> Optional[float]:
    """Извлекает число из слова — цифрой или словом."""
    if word.isdigit():
        return float(word)
    return NUM_WORDS_RU.get(word.lower())


def _parse_time_of_day(text: str) -> Optional[tuple]:
    """Парсит время из строки. Возвращает (hour, minute) или None."""
    text = text.lower()

    # Полночь / полдень
    if any(w in text for w in ["полночь", "полночи", "00:00"]):
        return (0, 0)
    if any(w in text for w in ["полдень", "полудня", "12:00"]):
        return (12, 0)

    # Утро/день/вечер/ночь без времени
    time_of_day_defaults = {
        "утр": 9, "утром": 9,
        "днём": 14, "днем": 14, "дня": 14,
        "вечер": 19, "вечером": 19,
        "ноч": 21, "ночью": 21,
    }

    # HH:MM
    m = re.search(r"(\d{1,2}):(\d{2})", text)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        # Коррекция: "3 дня" → 15:00
        if h < 12 and any(w in text for w in ["дня", "днём", "днем", "вечер", "вечером"]):
            h += 12
        if h < 7 and any(w in text for w in ["утр", "утром"]):
            pass  # оставляем как есть для "в 6 утра"
        return (h, mn)

    # "в X часов", "в X утра/вечера"
    m = re.search(r"в\s+(\d{1,2})\s*(час|утр|дня|вечер|ноч|:00)?", text)
    if m:
        h = int(m.group(1))
        suffix = (m.group(2) or "").lower()
        if h < 12 and any(s in suffix for s in ["дня", "вечер", "ноч"]):
            h += 12
        return (h, 0)

    # "X часов" без "в"
    m = re.search(r"(\d{1,2})\s*час", text)
    if m:
        h = int(m.group(1))
        if h < 12 and any(w in text for w in ["дня", "вечер", "ночи"]):
            h += 12
        return (h, 0)

    # Дефолт по времени суток
    for key, hour in time_of_day_defaults.items():
        if key in text:
            return (hour, 0)

    return None


def parse_russian_datetime(text: str) -> Optional[str]:
    """
    Парсит русское описание времени и возвращает "YYYY-MM-DD HH:MM" или None.
    Понимает:
    - через N минут/часов/дней
    - сегодня/завтра/послезавтра в HH:MM
    - в понедельник/пятницу в HH:MM
    - 15 декабря в 18:00
    - утром/вечером/в полдень
    - следующую неделю
    """
    now = now_local()
    text_lower = text.lower()

    # ── Относительное время: "через N минут/часов/дней" ──
    m = re.search(
        r"через\s+(\d+|\w+)\s*(минут|минуты|минуту|минутку|час|часа|часов|час|день|дня|дней|неделю|недели)",
        text_lower
    )
    if m:
        val_str = m.group(1)
        unit = m.group(2)
        val = _extract_number(val_str)
        if val is None:
            val = 1
        if "минут" in unit or "минут" in unit:
            dt = now + timedelta(minutes=val)
        elif "час" in unit:
            dt = now + timedelta(hours=val)
        elif "недел" in unit:
            dt = now + timedelta(weeks=val)
        else:  # дней
            dt = now + timedelta(days=val)
        return dt.strftime("%Y-%m-%d %H:%M")

    # ── Полчаса / четверть часа ──
    if re.search(r"через\s+полчас|через\s+пол\s*час", text_lower):
        return (now + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")
    if re.search(r"через\s+четверть", text_lower):
        return (now + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")

    # ── Базовая дата ──
    base_date = None

    if "послезавтра" in text_lower:
        base_date = now + timedelta(days=2)
    elif "сегодня" in text_lower:
        base_date = now
    elif "завтра" in text_lower:
        base_date = now + timedelta(days=1)
    elif re.search(r"следующ\w+\s+недел", text_lower):
        base_date = now + timedelta(weeks=1)

    # День недели
    if base_date is None:
        for ru_day, weekday_num in WEEKDAYS_RU.items():
            if re.search(rf"\b{ru_day}\b", text_lower):
                days_ahead = weekday_num - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                # "следующий понедельник" → +7
                if "следующ" in text_lower:
                    days_ahead += 7
                base_date = now + timedelta(days=days_ahead)
                break

    # Конкретная дата: "15 декабря", "1 января"
    if base_date is None:
        for month_key, month_num in MONTHS_RU.items():
            m = re.search(rf"(\d{{1,2}})\s*{month_key}\w*(?:\s+(\d{{4}}))?", text_lower)
            if m:
                day = int(m.group(1))
                year = int(m.group(2)) if m.group(2) else now.year
                try:
                    base_date = datetime(year, month_num, day)
                    if base_date < now and not m.group(2):
                        base_date = datetime(year + 1, month_num, day)
                except ValueError:
                    pass
                break

    # ── Время суток ──
    time_tuple = _parse_time_of_day(text_lower)

    if base_date is not None:
        if time_tuple:
            h, mn = time_tuple
            result = base_date.replace(hour=h, minute=mn, second=0, microsecond=0)
        else:
            # Нет времени — ставим 9:00 для будущих дат
            result = base_date.replace(hour=9, minute=0, second=0, microsecond=0)
        return result.strftime("%Y-%m-%d %H:%M")

    # Только время без даты — считаем сегодня, если уже прошло — завтра
    if time_tuple:
        h, mn = time_tuple
        result = now.replace(hour=h, minute=mn, second=0, microsecond=0)
        if result <= now:
            result += timedelta(days=1)
        return result.strftime("%Y-%m-%d %H:%M")

    return None


def parse_task(text: str, remind_default: int = 30) -> dict:
    now = now_local()
    now_str = now.strftime("%Y-%m-%d %H:%M")
    prompt = (
        f"Сейчас: {now_str}.\n"
        f'Пользователь написал: "{text}"\n\n'
        "Твоя задача — извлечь структуру. Ответь ТОЛЬКО JSON без пояснений и markdown:\n"
        "{\n"
        '  "task": "краткое описание что нужно сделать",\n'
        '  "due_time": "YYYY-MM-DD HH:MM" или null,\n'
        '  "remind_minutes_before": целое число или 0\n'
        "}\n\n"
        "ВАЖНЫЕ ПРАВИЛА:\n"
        "1. Если написано \'через X минут\' или \'через X часов\' — due_time = сейчас + это время\n"
        "2. Если написано \'напомни через X минут\' без срока — due_time = сейчас + X минут, remind_minutes_before = 0\n"
        "3. Если написано \'напомни за X минут до\' — remind_minutes_before = X\n"
        "4. Если срока нет вообще — due_time = null, remind_minutes_before = 0\n"
        f"5. Если есть срок но не сказано за сколько напомнить — remind_minutes_before = {remind_default}\n\n"
        "Примеры:\n"
        f'- "напомни позвонить маме через 10 минут" → due_time="{(now + __import__("datetime").timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M")}", remind_minutes_before=0\n'
        '- "встреча завтра в 10:00, напомни за час" → due_time=завтра 10:00, remind_minutes_before=60\n'
        '- "купить молоко" → due_time=null, remind_minutes_before=0\n'
        '- "позвони врачу в пятницу в 15:00" → due_time=ближайшая пятница 15:00, remind_minutes_before=30'
    )
    try:
        raw = ai_complete([{"role": "user", "content": prompt}])
        raw = re.sub(r"```json|```", "", raw).strip()
        # Берём только JSON если есть лишний текст
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
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
            remind_default  INTEGER DEFAULT 30,
            digest_hour     INTEGER DEFAULT 8,
            digest_minute   INTEGER DEFAULT 0,
            digest_enabled  INTEGER DEFAULT 1,
            timezone        TEXT    DEFAULT 'Europe/Moscow'
        )
    """)
    # Добавляем новые колонки если их нет (для существующих БД)
    for col, definition in [
        ("digest_hour",    "INTEGER DEFAULT 8"),
        ("digest_minute",  "INTEGER DEFAULT 0"),
        ("digest_enabled", "INTEGER DEFAULT 1"),
        ("timezone",       "TEXT DEFAULT 'Europe/Moscow'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE user_settings ADD COLUMN {col} {definition}")
        except Exception:
            pass
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



def db_get_user_settings(user_id: int) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        "remind_default": 30,
        "digest_hour": 8,
        "digest_minute": 0,
        "digest_enabled": 1,
        "timezone": "Europe/Moscow",
    }


def db_update_user_settings(user_id: int, **fields):
    keys = ", ".join(fields.keys())
    placeholders = ", ".join("?" * len(fields))
    updates = ", ".join(f"{k} = excluded.{k}" for k in fields)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        f"""INSERT INTO user_settings (user_id, {keys}) VALUES (?, {placeholders})
            ON CONFLICT(user_id) DO UPDATE SET {updates}""",
        [user_id] + list(fields.values()),
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
    now = now_local().strftime("%Y-%m-%d %H:%M")
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
    today = now_local().strftime("%Y-%m-%d")
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


def main_keyboard() -> ReplyKeyboardMarkup:
    """Постоянная клавиатура внизу чата."""
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("📋 Мои задачи"), KeyboardButton("➕ Добавить задачу")],
            [KeyboardButton("☀️ Задачи на сегодня"), KeyboardButton("⚙️ Настройки")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Напиши задачу или выбери действие...",
    )

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ai_name = "GigaChat (Сбер)" if AI_PROVIDER == "gigachat" else "YandexGPT (Яндекс)"
    stt_name = "Vosk (локально)" if STT_PROVIDER == "vosk" else "Yandex SpeechKit"
    await update.message.reply_text(
        "👋 *Привет! Я твой AI-помощник по задачам.*\n\n"
        "Напиши или скажи голосом что нужно сделать:\n"
        "  • «Позвони врачу завтра в 15:00»\n"
        "  • «Встреча в пятницу в 10, напомни за час»\n"
        "  • «Купить продукты»\n\n"
        "Или используй кнопки внизу 👇\n\n"
        f"_🤖 {ai_name}  |  🎙 {stt_name}_",
        parse_mode="Markdown",
        reply_markup=main_keyboard(),
    )


async def _show_settings(message, user_id: int):
    """Показывает настройки с кнопками — используется и из команды и из callback."""
    s = db_get_user_settings(user_id)
    ai_name = "GigaChat (Сбер)" if AI_PROVIDER == "gigachat" else "YandexGPT (Яндекс)"
    stt_name = "Vosk (локально)" if STT_PROVIDER == "vosk" else "Yandex SpeechKit"
    digest_status = "✅ включён" if s["digest_enabled"] else "❌ выключен"
    digest_time = f"{int(s['digest_hour']):02d}:{int(s['digest_minute']):02d}"
    tz = s.get("timezone", "Europe/Moscow")

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"🔔 Напоминание: {_minutes_label(s['remind_default'])}",
            callback_data="settings:remind"
        )],
        [InlineKeyboardButton(
            f"☀️ Дайджест: {digest_status}",
            callback_data="settings:digest_toggle"
        )],
        [InlineKeyboardButton(
            f"🕐 Время дайджеста: {digest_time}",
            callback_data="settings:digest_time"
        )],
        [InlineKeyboardButton(
            f"🌍 Часовой пояс: {tz}",
            callback_data="settings:timezone"
        )],
    ])
    text = (
        f"⚙️ *Настройки*\n\n"
        f"🤖 AI: {ai_name}\n"
        f"🎙 Речь: {stt_name}\n"
        f"🔔 Напоминание: {_minutes_label(s['remind_default'])}\n"
        f"☀️ Дайджест: {digest_status} в {digest_time}\n"
        f"🌍 Часовой пояс: {tz}\n\n"
        "Нажми кнопку чтобы изменить:"
    )
    await message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _show_settings(update.message, update.effective_user.id)


async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tasks = db_get_active_tasks(update.effective_user.id)
    if not tasks:
        await update.message.reply_text(
            "✅ Нет активных задач — всё чисто!",
            reply_markup=main_keyboard(),
        )
        return
    await update.message.reply_text(
        f"📋 *Активные задачи ({len(tasks)} шт.):*",
        parse_mode="Markdown",
        reply_markup=main_keyboard(),
    )
    for t in tasks:
        due = f"\n⏰ {t['due_time'][:16]}" if t["due_time"] else ""
        remind = f"\n🔔 {t['remind_at'][:16]}" if t.get("remind_at") and not t["reminded"] else ""
        text = f"📌 *{t['text']}*{due}{remind}"
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Выполнено", callback_data=f"done:{t['id']}"),
            InlineKeyboardButton("✏️ Изменить", callback_data=f"edit_select:{t['id']}"),
        ]])
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)


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



async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Задачи на сегодня."""
    user_id = update.effective_user.id
    tasks = db_get_today_tasks(user_id)
    if not tasks:
        await update.message.reply_text(
            "✅ На сегодня задач нет!",
            reply_markup=main_keyboard(),
        )
        return
    await update.message.reply_text(
        f"📅 *Задачи на сегодня ({len(tasks)} шт.):*",
        parse_mode="Markdown",
        reply_markup=main_keyboard(),
    )
    for t in tasks:
        time_str = f"\n⏰ {t['due_time'][11:16]}" if t["due_time"] else ""
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Выполнено", callback_data=f"done:{t['id']}"),
            InlineKeyboardButton("✏️ Изменить", callback_data=f"edit_select:{t['id']}"),
        ]])
        await update.message.reply_text(
            f"📌 *{t['text']}*{time_str}",
            parse_mode="Markdown",
            reply_markup=kb,
        )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    # Обработка кнопок постоянной клавиатуры
    if text == "📋 Мои задачи":
        await cmd_tasks(update, context)
        return
    if text == "☀️ Задачи на сегодня":
        await cmd_today(update, context)
        return
    if text == "⚙️ Настройки":
        await cmd_status(update, context)
        return
    if text == "➕ Добавить задачу":
        await update.message.reply_text(
            "Напиши или скажи голосом что нужно сделать.\nНапример: «Позвонить маме завтра в 18:00»"
        )
        return

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
        # Обрабатываем текст напрямую, не меняя объект сообщения
        if detect_intent(text) == "TASK":
            await _process_task(update, text)
        else:
            await update.message.reply_chat_action("typing")
            reply = await chat_reply(update.effective_user.id, text)
            await update.message.reply_text(reply)
    except Exception as e:
        logger.error(f"Ошибка транскрипции: {e}")
        await update.message.reply_text("❌ Ошибка распознавания. Напиши задачу текстом.")


def _format_due(due_time: str) -> str:
    """Красиво форматирует дату/время срока."""
    try:
        dt = datetime.strptime(due_time, "%Y-%m-%d %H:%M")
        today = now_local().date()
        tomorrow = today + timedelta(days=1)
        weekdays = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
        weekday = weekdays[dt.weekday()]
        time_str = dt.strftime("%H:%M")
        if dt.date() == today:
            return f"сегодня в {time_str}"
        elif dt.date() == tomorrow:
            return f"завтра в {time_str}"
        else:
            return f"{dt.day:02d}.{dt.month:02d} ({weekday}) в {time_str}"
    except Exception:
        return due_time


def _format_remind(remind_at: str, due_time: str) -> str:
    """Показывает за сколько напомним."""
    try:
        due_dt = datetime.strptime(due_time, "%Y-%m-%d %H:%M")
        rem_dt = datetime.strptime(remind_at, "%Y-%m-%d %H:%M")
        delta_min = int((due_dt - rem_dt).total_seconds() / 60)
        if delta_min < 60:
            label = f"за {delta_min} мин до срока"
        elif delta_min % 60 == 0:
            label = f"за {delta_min // 60} ч до срока"
        else:
            label = f"за {delta_min // 60} ч {delta_min % 60} мин до срока"
        rem_time = rem_dt.strftime("%H:%M")
        return f"{rem_time} ({label})"
    except Exception:
        return remind_at


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

    if due_time:
        due_fmt = _format_due(due_time)
        remind_fmt = _format_remind(remind_at, due_time) if remind_at else None
        msg = (
            f"✅ *Задача добавлена!*\n"
            f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
            f"📋 {task_text}\n"
            f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
            f"📅 *Срок:* {due_fmt}\n"
        )
        if remind_fmt:
            msg += f"🔔 *Напомню:* {remind_fmt}"
    else:
        msg = (
            f"✅ *Задача добавлена!*\n"
            f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
            f"📋 {task_text}\n"
            f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
            f"📅 _Без конкретного срока_"
        )

    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Выполнено", callback_data=f"done:{task_id}"),
        InlineKeyboardButton("✏️ Изменить", callback_data=f"edit_select:{task_id}"),
    ]])
    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=kb)


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


# ══════════════════════════════════════════════
# БЛОК 4г: НАСТРОЙКИ (дайджест, часовой пояс)
# ══════════════════════════════════════════════

TIMEZONES_RU = {
    "Москва (UTC+3)":       "Europe/Moscow",
    "Калининград (UTC+2)":  "Europe/Kaliningrad",
    "Самара (UTC+4)":       "Europe/Samara",
    "Екатеринбург (UTC+5)": "Asia/Yekaterinburg",
    "Омск (UTC+6)":         "Asia/Omsk",
    "Красноярск (UTC+7)":   "Asia/Krasnoyarsk",
    "Иркутск (UTC+8)":      "Asia/Irkutsk",
    "Якутск (UTC+9)":       "Asia/Yakutsk",
    "Владивосток (UTC+10)": "Asia/Vladivostok",
    "Магадан (UTC+11)":     "Asia/Magadan",
    "Камчатка (UTC+12)":    "Asia/Kamchatka",
}


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    action = query.data.split(":")[1]

    if action == "remind":
        await query.message.reply_text("\U0001f447 Выбери время напоминания по умолчанию:")
        await cmd_remindset_msg(query.message, user_id)
        return SETTINGS_MAIN

    elif action == "digest_toggle":
        s = db_get_user_settings(user_id)
        new_val = 0 if s["digest_enabled"] else 1
        db_update_user_settings(user_id, digest_enabled=new_val)
        status = "\u2705 включён" if new_val else "\u274c выключен"
        await query.answer(f"Дайджест {status}", show_alert=True)
        await _show_settings(query.message, user_id)
        return SETTINGS_MAIN

    elif action == "digest_time":
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("6:00",  callback_data="digest_hour:6"),
                InlineKeyboardButton("7:00",  callback_data="digest_hour:7"),
                InlineKeyboardButton("8:00",  callback_data="digest_hour:8"),
            ],
            [
                InlineKeyboardButton("9:00",  callback_data="digest_hour:9"),
                InlineKeyboardButton("10:00", callback_data="digest_hour:10"),
                InlineKeyboardButton("11:00", callback_data="digest_hour:11"),
            ],
            [InlineKeyboardButton("\u270f\ufe0f Своё время", callback_data="digest_hour:custom")],
        ])
        await query.message.reply_text(
            "\u2600\ufe0f Выбери время утреннего дайджеста:",
            reply_markup=keyboard,
        )
        return SETTINGS_DIGEST_HOUR

    elif action == "timezone":
        rows = []
        keys = list(TIMEZONES_RU.keys())
        for i in range(0, len(keys), 2):
            row = [InlineKeyboardButton(keys[i], callback_data=f"tz:{TIMEZONES_RU[keys[i]]}")]
            if i + 1 < len(keys):
                row.append(InlineKeyboardButton(keys[i+1], callback_data=f"tz:{TIMEZONES_RU[keys[i+1]]}"))
            rows.append(row)
        await query.message.reply_text(
            "\U0001f30d Выбери свой часовой пояс:",
            reply_markup=InlineKeyboardMarkup(rows),
        )
        return SETTINGS_TIMEZONE

    return SETTINGS_MAIN


async def digest_hour_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    val = query.data.split(":")[1]
    if val == "custom":
        await query.message.reply_text(
            "\u270f\ufe0f Введи время в формате ЧЧ:ММ, например: *07:30*",
            parse_mode="Markdown",
        )
        return SETTINGS_DIGEST_HOUR
    hour = int(val)
    db_update_user_settings(user_id, digest_hour=hour, digest_minute=0)
    await query.message.reply_text(f"\u2705 Дайджест будет приходить в *{hour:02d}:00*", parse_mode="Markdown")
    await _show_settings(query.message, user_id)
    return SETTINGS_MAIN


async def digest_time_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    import re as _re
    m = _re.match(r"^(\d{1,2}):(\d{2})$", text)
    if not m:
        await update.message.reply_text("\u274c Неверный формат. Введи время как *08:30*", parse_mode="Markdown")
        return SETTINGS_DIGEST_HOUR
    h, mn = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mn <= 59):
        await update.message.reply_text("\u274c Некорректное время. Попробуй ещё раз:")
        return SETTINGS_DIGEST_HOUR
    db_update_user_settings(user_id, digest_hour=h, digest_minute=mn)
    await update.message.reply_text(f"\u2705 Дайджест будет приходить в *{h:02d}:{mn:02d}*", parse_mode="Markdown")
    await _show_settings(update.message, user_id)
    return ConversationHandler.END


async def timezone_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    tz = query.data.split(":")[1]
    db_update_user_settings(user_id, timezone=tz)
    label = next((k for k, v in TIMEZONES_RU.items() if v == tz), tz)
    await query.message.reply_text(f"\u2705 Часовой пояс: *{label}*", parse_mode="Markdown")
    await _show_settings(query.message, user_id)
    return SETTINGS_MAIN


async def done_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    task_id = int(query.data.split(":")[1])
    user_id = update.effective_user.id
    if db_complete_task(task_id, user_id):
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(f"\u2705 Выполнено! Молодец \U0001f4aa")
    else:
        await query.answer("\u274c Задача не найдена", show_alert=True)


async def cmd_remindset_msg(message, user_id: int):
    current = db_get_remind_default(user_id)
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("5 мин",  callback_data="remind_set:5"),
            InlineKeyboardButton("15 мин", callback_data="remind_set:15"),
            InlineKeyboardButton("30 мин", callback_data="remind_set:30"),
        ],
        [
            InlineKeyboardButton("1 час",  callback_data="remind_set:60"),
            InlineKeyboardButton("2 часа", callback_data="remind_set:120"),
            InlineKeyboardButton("3 часа", callback_data="remind_set:180"),
        ],
        [
            InlineKeyboardButton("12 часов", callback_data="remind_set:720"),
            InlineKeyboardButton("1 день",   callback_data="remind_set:1440"),
        ],
        [InlineKeyboardButton("\u270f\ufe0f Своё значение", callback_data="remind_set:custom")],
    ])
    await message.reply_text(
        f"\U0001f514 *Напоминание по умолчанию*\n\nСейчас: *{_minutes_label(current)}*\n\nВыбери:",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


async def check_reminders(app: Application):
    for task in db_get_pending_reminders():
        try:
            due_str = task["due_time"][:16] if task["due_time"] else "скоро"
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Выполнено", callback_data=f"done:{task['id']}"),
                InlineKeyboardButton("✏️ Изменить", callback_data=f"edit_select:{task['id']}"),
            ]])
            due_fmt = _format_due(task["due_time"]) if task["due_time"] else "скоро"
            now = now_local()
            try:
                due_dt = datetime.strptime(task["due_time"], "%Y-%m-%d %H:%M")
                mins_left = int((due_dt - now).total_seconds() / 60)
                if mins_left <= 0:
                    time_left = "⚡️ уже сейчас!"
                elif mins_left < 60:
                    time_left = f"через {mins_left} мин"
                elif mins_left < 1440:
                    time_left = f"через {mins_left // 60} ч {mins_left % 60} мин"
                else:
                    time_left = f"через {mins_left // 1440} дн."
            except Exception:
                time_left = ""

            reminder_text = (
                f"🔔 *Напоминание*\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📋 *{task['text']}*\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📅 *Срок:* {due_fmt}\n"
                f"⏳ *Осталось:* {time_left}"
            )
            await app.bot.send_message(
                chat_id=task["user_id"],
                text=reminder_text,
                parse_mode="Markdown",
                reply_markup=keyboard,
            )
            db_mark_reminded(task["id"])
        except Exception as e:
            logger.error(f"Ошибка напоминания {task['id']}: {e}")


async def send_morning_digest(app: Application):
    now = now_local()
    for user_id in db_all_user_ids():
        s = db_get_user_settings(user_id)
        # Пропускаем если дайджест выключен
        if not s.get("digest_enabled", 1):
            continue
        # Отправляем только если сейчас час дайджеста этого пользователя
        if now.hour != int(s.get("digest_hour", DIGEST_HOUR)) or now.minute != int(s.get("digest_minute", DIGEST_MINUTE)):
            continue
        tasks = db_get_today_tasks(user_id)
        if not tasks:
            continue
        try:
            await app.bot.send_message(
                chat_id=user_id,
                text=(
                    f"☀️ *Доброе утро!*\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📅 Сегодня {now_local().strftime('%d.%m.%Y')}\n"
                    f"📋 Задач на сегодня: *{len(tasks)}*\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"_Хорошего и продуктивного дня! 💪_"
                ),
                parse_mode="Markdown",
            )
            for i, t in enumerate(tasks, 1):
                time_str = f"\n🕐 *Время:* {t['due_time'][11:16]}" if t["due_time"] else "\n📅 _Без конкретного времени_"
                kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton("✅ Выполнено", callback_data=f"done:{t['id']}"),
                    InlineKeyboardButton("✏️ Изменить", callback_data=f"edit_select:{t['id']}"),
                ]])
                await app.bot.send_message(
                    chat_id=user_id,
                    text=(
                        f"*{i}.* {t['text']}"
                        f"{time_str}"
                    ),
                    parse_mode="Markdown",
                    reply_markup=kb,
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

    # ConversationHandler: настройки
    settings_handler = ConversationHandler(
        entry_points=[
            CommandHandler("settings", cmd_status),
            CallbackQueryHandler(settings_callback, pattern=r"^settings:"),
        ],
        states={
            SETTINGS_MAIN: [
                CallbackQueryHandler(settings_callback, pattern=r"^settings:"),
                CallbackQueryHandler(digest_hour_callback, pattern=r"^digest_hour:"),
                CallbackQueryHandler(timezone_callback, pattern=r"^tz:"),
            ],
            SETTINGS_DIGEST_HOUR: [
                CallbackQueryHandler(digest_hour_callback, pattern=r"^digest_hour:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, digest_time_custom),
            ],
            SETTINGS_TIMEZONE: [
                CallbackQueryHandler(timezone_callback, pattern=r"^tz:"),
            ],
        },
        fallbacks=[CommandHandler("cancel", edit_cancel)],
        per_message=False,
    )

    app.add_handler(edit_handler)
    app.add_handler(remindset_handler)
    app.add_handler(settings_handler)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("tasks", cmd_tasks))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("done", cmd_done))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CallbackQueryHandler(done_callback, pattern=r"^done:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_reminders, "interval", minutes=1, args=[app])
    # Дайджест каждую минуту — у каждого пользователя своё время
    scheduler.add_job(send_morning_digest, "interval", minutes=1, args=[app])
    scheduler.start()

    logger.info("Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
