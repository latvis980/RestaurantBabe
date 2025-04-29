# telegram_bot.py — enhanced conversational interface with preference collection and confirmation updates
import telebot
import logging
import time
import traceback
import asyncio
import os
from typing import Dict, Any, Optional

from agents.langchain_orchestrator import LangChainOrchestrator
import config
from utils.debug_utils import dump_chain_state
from utils.database import initialize_db, tables, engine
from sqlalchemy import insert
from openai import OpenAI

# -------------------------------------------------
# CONFIGURATION & GLOBALS
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# DB initialisation
initialize_db(config)

USER_PREFS_TABLE = tables.get(config.DB_TABLE_USER_PREFS)
USER_SEARCHES_TABLE = tables.get(config.DB_TABLE_SEARCHES)

# in‑memory user context
user_state: Dict[int, Dict[str, Any]] = {}  # {stage, last_query, prefs{raw}, pref_candidate}

bot = telebot.TeleBot(BOT_TOKEN)
orchestrator = LangChainOrchestrator(config)

# -------------------------------------------------
# HELPER UTILITIES
# -------------------------------------------------

def classify_intent(message: str, history: str = "") -> str:
    """Return one of {search, follow_up, chat}."""
    prompt = (
        "You are an intent classifier for a restaurant search assistant.\n"
        "Intents: search, follow_up, chat.\n\n"
        f"History: {history}\nUser: {message}\nRespond with one token."
    )
    try:
        rsp = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            max_tokens=1, temperature=0
        )
        intent = rsp.choices[0].message.content.strip().lower()
        return intent if intent in {"search", "follow_up", "chat"} else "search"
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}")
        return "search"

def detect_new_preference(text: str, existing_raw: str) -> Optional[str]:
    """Return a new preference phrase to propose storing, or None."""
    prompt = (
        "Existing preferences: " + (existing_raw or "<none>") + "\n"
        "User message: " + text + "\n\n"
        "If the user expresses a NEW standing food preference (e.g. vegetarian, gluten‑free, moderate price) "
        "that is not obviously already in existing preferences, output that preference phrase (max 4 words). "
        "If none, output 'none'."
    )
    try:
        rsp = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            max_tokens=4, temperature=0
        )
        candidate = rsp.choices[0].message.content.strip().lower()
        if candidate == "none" or not candidate:
            return None
        # quick dedup check
        return candidate if candidate not in existing_raw.lower() else None
    except Exception as e:
        logger.debug(f"Preference detection failed: {e}")
        return None

def store_preferences(user_id: int, raw_text: str):
    if USER_PREFS_TABLE is None:
        return
    with engine.begin() as conn:
        conn.execute(
            insert(USER_PREFS_TABLE).values(
                _id=str(user_id),
                data={"prefs_text": raw_text},
                timestamp=time.time()
            ).on_conflict_do_update(
                index_elements=[USER_PREFS_TABLE.c._id],
                set_={"data": {"prefs_text": raw_text}, "timestamp": time.time()}
            )
        )
    user_state.setdefault(user_id, {}).setdefault("prefs", {})["raw"] = raw_text

def ask_for_preferences(message):
    bot.reply_to(
        message,
        "Чтобы подобрать рестораны точнее, расскажите об атмосфере, ценовом диапазоне, кухнях, "
        "или ограничениях (например, вегетарианское, без глютена), которые вы предпочитаете.")

def ask_to_store_pref(message, cand: str):
    bot.reply_to(
        message,
        f"Хотите, чтобы я запомнил ваш предпочтение '{cand}' как постоянное? Напишите 'да' или 'нет'.")

def merge_with_last_query(user_id: int, new_msg: str) -> str:
    prev = user_state.get(user_id, {}).get("last_query", "")
    return f"{prev}\nFollow‑up: {new_msg}"

# -------------------------------------------------
# BOT HANDLERS
# -------------------------------------------------

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    uid = message.from_user.id
    user_state[uid] = {"stage": "awaiting_first_query"}
    bot.reply_to(
        message,
        "Привет! Я помогу найти классные рестораны. Опишите, что вы ищете — например, "
        "'современная португальская кухня в Лиссабоне'.")

@bot.message_handler(func=lambda m: True)
def handle_message(message):
    uid = message.from_user.id
    text = message.text.strip()
    state = user_state.setdefault(uid, {"stage": "awaiting_first_query"})

    try:
        # 0) Waiting for yes/no about storing preference
        if state.get("stage") == "awaiting_pref_confirm":
            if text.lower() in {"да", "yes", "y"}:
                cand = state.pop("pref_candidate", "")
                existing = state.get("prefs", {}).get("raw", "")
                new_raw = (existing + ", " + cand).strip(", ") if existing else cand
                store_preferences(uid, new_raw)
                bot.reply_to(message, "Готово! Предпочтение сохранено.")
            else:
                bot.reply_to(message, "Хорошо, не буду сохранять это предпочтение.")
            state["stage"] = "ready"
            return

        # 1) Waiting for initial preferences
        if state.get("stage") == "awaiting_prefs":
            store_preferences(uid, text)
            pending_query = state.pop("pending_query", "")
            state["stage"] = "ready"
            full_q = f"{pending_query}\nUser preferences: {text}"
            process_and_respond(full_q, message)
            return

        # 2) First query ever
        if state.get("stage") == "awaiting_first_query":
            state["pending_query"] = text
            state["stage"] = "awaiting_prefs"
            ask_for_preferences(message)
            return

        # 3) Established user – detect potential new preference
        existing_raw = state.get("prefs", {}).get("raw", "")
        cand_pref = detect_new_preference(text, existing_raw)
        if cand_pref:
            state["pref_candidate"] = cand_pref
            state["stage"] = "awaiting_pref_confirm"  # ask after we send results

        # classify intent
        intent = classify_intent(text, state.get("last_query", ""))
        if intent == "chat":
            bot.reply_to(message, "Понимаю! Чем могу помочь? Расскажите, что ищете.")
            if state.get("stage") == "awaiting_pref_confirm":
                ask_to_store_pref(message, state["pref_candidate"])
            return
        elif intent == "follow_up":
            query = merge_with_last_query(uid, text)
        else:
            query = text

        process_and_respond(query, message)

        if state.get("stage") == "awaiting_pref_confirm":
            ask_to_store_pref(message, state["pref_candidate"])
    except Exception as e:
        logger.error(f"handle_message error: {e}")
        logger.error(traceback.format_exc())
        bot.reply_to(message, "Извините, произошла ошибка. Попробуйте ещё раз.")

# -------------------------------------------------

def process_and_respond(query: str, message):
    uid = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")
    interim = bot.reply_to(message, "Ищу подходящие варианты…")
    start = time.time()
    try:
        result = orchestrator.process_query(query)
        user_state[uid]["last_query"] = query
        if USER_SEARCHES_TABLE is not None and isinstance(result, dict):
            with engine.begin() as conn:
                conn.execute(insert(USER_SEARCHES_TABLE).values(
                    _id=str(uid) + "-" + str(int
