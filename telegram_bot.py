# telegram_bot.py — enhanced conversational interface with preference collection and confirmation updates
"""
Key additions compared with the previous version:
• Detects possible *new* standing preferences mid‑conversation (e.g. “а вегетарианское?”)
• Confirms with the user before saving (“Сохранить ‘вегетарианское’ как постоянное?”)
• Saves/updates those preferences in the new `user_prefs` table only after explicit consent.
"""

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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

initialize_db(config)
USER_PREFS_TABLE = tables.get(config.DB_TABLE_USER_PREFS)
USER_SEARCHES_TABLE = tables.get(config.DB_TABLE_SEARCHES)

# in‑memory conversation context
user_state: Dict[int, Dict[str, Any]] = {}

bot = telebot.TeleBot(BOT_TOKEN)
orchestrator = LangChainOrchestrator(config)

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------

def classify_intent(message: str, history: str = "") -> str:
    """Classify into search / follow_up / chat."""
    prompt = (
        "You are an intent classifier for a restaurant search assistant.\n"
        "Intents: search, follow_up, chat.\n\n"
        f"History: {history}\nUser: {message}\nRespond with one token."
    )
    try:
        rsp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
        )
        intent = rsp.choices[0].message.content.strip().lower()
        return intent if intent in {"search", "follow_up", "chat"} else "search"
    except Exception as e:
        logger.warning("Intent classification failed – defaulting to 'search': %s", e)
        return "search"


def detect_new_preference(text: str, existing_raw: str) -> Optional[str]:
    """If message contains a new standing preference, return keyword, else None."""
    prompt = (
        "Existing preferences: " + (existing_raw or "<none>") + "\n"
        "User message: " + text + "\n\n"
        "If the user expresses a NEW *standing* food preference (e.g. vegetarian, gluten‑free, moderate price) that "
        "is not already in existing preferences, output that single short phrase (max 4 words). If no new preference "
        "is detected, output 'none'."
    )
    try:
        rsp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0,
        )
        candidate = rsp.choices[0].message.content.strip().lower()
        if candidate == "none" or not candidate:
            return None
        return candidate if candidate not in existing_raw.lower() else None
    except Exception as e:
        logger.debug("Preference detection failed: %s", e)
        return None


def store_preferences(user_id: int, raw_text: str):
    """Insert or update the user's standing preferences in DB and memory."""
    if USER_PREFS_TABLE is None:
        return
    with engine.begin() as conn:
        conn.execute(
            insert(USER_PREFS_TABLE)
            .values(_id=str(user_id), data={"prefs_text": raw_text}, timestamp=time.time())
            .on_conflict_do_update(
                index_elements=[USER_PREFS_TABLE.c._id],
                set_={"data": {"prefs_text": raw_text}, "timestamp": time.time()},
            )
        )
    user_state.setdefault(user_id, {}).setdefault("prefs", {})["raw"] = raw_text


def ask_for_preferences(message):
    bot.reply_to(
        message,
        "Чтобы я могла точнее посоветовать вам рестораны, расскажите, какие рестораны в каком стиле вы любите и есть ли у вас какие-либо ограничения. Например, вы предпочитаете вегетарианские рестораны или всегда ходите только в рестораны, где нет туристов?",
    )


def ask_to_store_pref(message, cand: str):
    bot.reply_to(
        message,
        f"Хотите, чтобы я запомнил предпочтение ‘{cand}’ как постоянное? Напишите ‘да’ или ‘нет’.",
    )


def merge_with_last_query(user_id: int, follow_up: str) -> str:
    prev = user_state.get(user_id, {}).get("last_query", "")
    return f"{prev}\nFollow‑up: {follow_up}"

# -------------------------------------------------
# BOT HANDLERS
# -------------------------------------------------

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    uid = message.from_user.id
    user_state[uid] = {"stage": "awaiting_first_query"}
    bot.reply_to(
        message,
        "Привет! Я ИИ-ассистент про прозвищу Restaurant Babe и я умею находить самые вкусные, самые модные, самые классные рестораны, кафе, пекарни, бары и кофейни по всему миру. Напишите, что вы ищете, например: \n\n'Модные места для бранча в Лиссабоне с необычными блюдами'\n Или 'Любимые севичерии местных жителей в Лиме'\nИли 'Где самый вкусный плов в Ташкенте?'\n\n Я наведу справки у  знакомых ресторанных экспертов, пролистаю колонки гастрономических критиков — и выдам лучшие рекомендации. \n\n Это может занять пару минут, потому что ищу я очень внимательно и тщательно проверяю результаты. Но никаких случайных мест в моем списке не будет. \n\nНачнем?",
    )


@bot.message_handler(func=lambda m: True)
def handle_message(message):
    uid = message.from_user.id
    text = message.text.strip()
    state = user_state.setdefault(uid, {"stage": "awaiting_first_query"})

    try:
        # 0) Yes/No about saving new preference
        if state.get("stage") == "awaiting_pref_confirm":
            if text.lower() in {"да", "yes", "y"}:
                cand = state.pop("pref_candidate", "")
                existing = state.get("prefs", {}).get("raw", "")
                new_raw = (existing + ", " + cand).strip(", ") if existing else cand
                store_preferences(uid, new_raw)
                bot.reply_to(message, "Готово — предпочтение сохранено.")
            else:
                bot.reply_to(message, "Хорошо, не сохраняю.")
            state["stage"] = "ready"
            return

        # 1) Still waiting for initial preferences (first time)
        if state.get("stage") == "awaiting_prefs":
            store_preferences(uid, text)
            pending_query = state.pop("pending_query", "")
            state["stage"] = "ready"
            full_query = f"{pending_query}\nUser preferences: {text}"
            process_and_respond(full_query, message)
            return

        # 2) First ever query
        if state.get("stage") == "awaiting_first_query":
            state["pending_query"] = text
            state["stage"] = "awaiting_prefs"
            ask_for_preferences(message)
            return

        # 3) Existing user — maybe a new standing preference?
        existing_raw = state.get("prefs", {}).get("raw", "")
        cand_pref = detect_new_preference(text, existing_raw)
        if cand_pref:
            state["pref_candidate"] = cand_pref
            state["stage"] = "awaiting_pref_confirm"  # we’ll ask after results

        # Intent classification
        intent = classify_intent(text, state.get("last_query", ""))
        if intent == "chat":
            bot.reply_to(message, "Понимаю! Чем могу помочь?")
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
        logger.error("handle_message error: %s", e)
        logger.error(traceback.format_exc())
        bot.reply_to(message, "Извините, произошла ошибка. Попробуйте ещё раз.")


# -------------------------------------------------
# CORE SEARCH AND RESPONSE
# -------------------------------------------------

def process_and_respond(query: str, message):
    uid = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")
    interim = bot.reply_to(message, "Ищу подходящие варианты…")
    start_ts = time.time()

    try:
        result = orchestrator.process_query(query)
        user_state[uid]["last_query"] = query

        # Persist search
        if USER_SEARCHES_TABLE is not None and isinstance(result, dict):
            with engine.begin() as conn:
                conn.execute(
                    insert(USER_SEARCHES_TABLE).values(
                        _id=str(uid) + "-" + str(int(time.time() * 1000)),
                        data={"query": query, "result": result},
                        timestamp=time.time(),
                    )
                )

        resp_text = (
            result.get("telegram_text", "Извините, ничего не найдено.")
            if isinstance(result, dict)
            else str(result)
        )

        # Clean up placeholder
        try:
            bot.delete_message(message.chat.id, interim.message_id)
        except Exception:
            pass

        bot.send_message(message.chat.id, resp_text, parse_mode="HTML")
        dump_chain_state("telegram_response_sent", {"processing_time": time.time() - start_ts})

    except Exception as e:
        logger.error("process_and_respond error: %s", e)
        try:
            bot.delete_message(message.chat.id, interim.message_id)
        except Exception:
            pass
        bot.reply_to(message, "Извините, поиск не удался. Попробуйте чуть позже.")


# -------------------------------------------------
# CLEAN SHUTDOWN
# -------------------------------------------------

def shutdown():
    logger.info("Shutting down…")
    from utils.async_utils import wait_for_pending_tasks

    try:
        asyncio.run(wait_for_pending_tasks())
    except RuntimeError:
        pass


# -------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------

def main():
    import atexit

    atexit.register(shutdown)
    logger.info("Telegram bot is running with preference‑confirmation features…")
    bot.infinity_polling()


if __name__ == "__main__":
    main()
