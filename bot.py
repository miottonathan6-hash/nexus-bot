"""ATLAS — bot Telegram (version corrigée).

Corrections par rapport à la version initiale :
- appel OpenAI asynchrone avec timeout (le bot ne se fige plus pendant la réponse) ;
- historique de conversation borné et sauvegardé sur disque ;
- réponses longues découpées (limite Telegram de 4096 caractères) ;
- gestionnaire d'erreurs global + clic répété sur le bouton géré ;
- erreurs techniques dans les logs, message générique pour l'utilisateur ;
- variables d'environnement validées au démarrage ;
- handlers limités aux messages privés.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# --- Configuration ----------------------------------------------------------

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MEMORY_FILE = Path(os.environ.get("MEMORY_FILE", "conversation_memory.json"))

MAX_TURNS = 20          # nombre de messages conservés en plus du prompt système
TELEGRAM_LIMIT = 4000   # marge sous la limite réelle de 4096 caractères
OPENAI_TIMEOUT = 45     # secondes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("atlas")


def _read_founder_id() -> int:
    raw = os.environ.get("FOUNDER_ID", "0").strip()
    try:
        return int(raw)
    except ValueError:
        logger.error("FOUNDER_ID doit être un nombre entier, reçu: %r", raw)
        return 0


FOUNDER_ID = _read_founder_id()

openai_client = (
    AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT) if OPENAI_API_KEY else None
)

SYSTEM_PROMPT = (
    "Tu es ATLAS, le Gérant Autonome du projet NEXUS Santé. "
    f"Ton fondateur est Nando (ID: {FOUNDER_ID}). "
    "Objectif: 100k€/mois avec 30% de marge nette en vendant des agents IA "
    "aux cliniques privées. Sois chirurgical, orienté résultats, utilise des emojis. "
    "Appelle-le toujours 'Fondateur Nando'."
)

# --- Mémoire de conversation ------------------------------------------------

conversation_memory: dict[int, list[dict]] = {}
_memory_lock = asyncio.Lock()


def load_memory() -> None:
    """Recharge l'historique sauvegardé, s'il existe."""
    if not MEMORY_FILE.exists():
        return
    try:
        raw = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        conversation_memory.update({int(k): v for k, v in raw.items()})
        logger.info("Mémoire rechargée: %d conversation(s).", len(conversation_memory))
    except (json.JSONDecodeError, ValueError, OSError):
        logger.exception("Mémoire illisible, on repart de zéro.")


def save_memory() -> None:
    try:
        tmp = MEMORY_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({str(k): v for k, v in conversation_memory.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(MEMORY_FILE)
    except OSError:
        logger.exception("Échec de la sauvegarde de la mémoire.")


def reset_history(user_id: int) -> list[dict]:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation_memory[user_id] = history
    return history


def get_history(user_id: int) -> list[dict]:
    history = conversation_memory.get(user_id)
    if not history:
        return reset_history(user_id)
    # Le prompt système est régénéré au cas où il aurait changé depuis la sauvegarde.
    history[0] = {"role": "system", "content": SYSTEM_PROMPT}
    return history


def trim_history(user_id: int) -> None:
    history = conversation_memory[user_id]
    if len(history) > MAX_TURNS + 1:
        conversation_memory[user_id] = [history[0]] + history[-MAX_TURNS:]


# --- Utilitaires ------------------------------------------------------------


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Découpe un texte trop long pour Telegram, de préférence sur un saut de ligne."""
    if not text:
        return ["(réponse vide)"]
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    chunks.append(text)
    return chunks


def is_founder(update: Update) -> bool:
    user = update.effective_user
    return bool(user) and user.id == FOUNDER_ID


# --- Handlers ---------------------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_founder(update):
        await update.message.reply_text("Accès refusé.")
        return
    async with _memory_lock:
        reset_history(update.effective_user.id)
        save_memory()
    keyboard = [[InlineKeyboardButton("📊 Rapport Statut", callback_data="rapport")]]
    await update.message.reply_text(
        "✅ NEXUS CONNECTÉ. Système opérationnel.",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_founder(update):
        return
    async with _memory_lock:
        reset_history(update.effective_user.id)
        save_memory()
    await update.message.reply_text("🧹 Mémoire effacée.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_founder(update) or not update.message or not update.message.text:
        return
    user_id = update.effective_user.id

    if openai_client is None:
        await update.message.reply_text("Erreur: clé OpenAI manquante.")
        return

    await update.effective_chat.send_action(action=ChatAction.TYPING)

    async with _memory_lock:
        history = get_history(user_id)
        history.append({"role": "user", "content": update.message.text})
        trim_history(user_id)
        messages = list(conversation_memory[user_id])

    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages
        )
    except asyncio.TimeoutError:
        logger.warning("Timeout OpenAI pour %s", user_id)
        await update.message.reply_text("⏳ Le modèle met trop de temps à répondre. Réessaie.")
        return
    except Exception:
        logger.exception("Appel OpenAI en échec")
        await update.message.reply_text("⚠️ Erreur système. Détails dans les logs.")
        return

    ai_reply = (response.choices[0].message.content or "").strip()

    async with _memory_lock:
        conversation_memory[user_id].append({"role": "assistant", "content": ai_reply})
        trim_history(user_id)
        save_memory()

    for chunk in split_message(ai_reply):
        await update.message.reply_text(chunk)


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    if query.from_user.id != FOUNDER_ID:
        await query.answer("Accès refusé.", show_alert=True)
        return
    await query.answer()

    if query.data != "rapport":
        return

    messages_count = max(len(conversation_memory.get(query.from_user.id, [])) - 1, 0)
    text = (
        "📊 Rapport\n"
        f"🧠 Modèle: {OPENAI_MODEL}\n"
        f"💬 Messages en mémoire: {messages_count}\n"
        "💰 Coût et marge: non connectés à une source de données réelle\n"
        f"🕒 {update.effective_message.date:%d/%m %H:%M} (heure du message)"
    )
    try:
        await query.edit_message_text(text=text)
    except BadRequest as exc:
        # Deuxième clic avec un texte identique : Telegram refuse la modification.
        if "not modified" not in str(exc).lower():
            raise
        await query.answer("Rapport déjà à jour.", show_alert=False)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Erreur non gérée", exc_info=context.error)


# --- Démarrage --------------------------------------------------------------


def main() -> None:
    missing = [
        name
        for name, value in (
            ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
            ("OPENAI_API_KEY", OPENAI_API_KEY),
        )
        if not value
    ]
    if FOUNDER_ID == 0:
        missing.append("FOUNDER_ID")
    if missing:
        print("ERREUR: variables d'environnement manquantes ou invalides: " + ", ".join(missing))
        return

    load_memory()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("reset", reset, filters=filters.ChatType.PRIVATE))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE, handle_message
        )
    )
    app.add_error_handler(error_handler)

    print("✅ Bot démarré...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
