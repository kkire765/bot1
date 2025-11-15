import os
import io
import re
import logging
import asyncio
from typing import Dict, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import arabic_reshaper
from bidi.algorithm import get_display
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

from aiohttp import web

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


# Deterministic language detection
DetectorFactory.seed = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("translator_bot")


# In-memory user preferences: chat_id -> target language code for Arabic reverse translations
USER_PREFS: Dict[int, str] = {}


def get_env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def shape_arabic(text: str) -> str:
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except Exception:
        return text


def get_font(size: int) -> ImageFont.FreeTypeFont:
    font_path = os.getenv("ARABIC_FONT_PATH")
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception as e:
            logger.warning("Failed to load font '%s': %s. Falling back to default.", font_path, e)
    # Fallback: default font (may not support Arabic shaping fully)
    return ImageFont.load_default()


def is_arabic_text(text: str) -> bool:
    try:
        lang = detect(text)
        return lang == "ar"
    except Exception:
        # Fallback: heuristic based on Unicode range
        return bool(re.search(r"[\u0600-\u06FF]", text))


def translate_text(text: str, target_lang: str) -> str:
    translator = GoogleTranslator(source="auto", target=target_lang)
    return translator.translate(text)


def autodetect_target(text: str, default_reverse: str = "en") -> str:
    return default_reverse if is_arabic_text(text) else "ar"


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "مرحبًا! أرسل نصًا وسأترجمه تلقائيًا:\n"
        "• إذا كان بالعربية → أترجمه للغة محددة (افتراضيًا الإنجليزية).\n"
        "• إذا كان بغير العربية → أترجمه للعربية.\n\n"
        "أوامر مفيدة:\n"
        "/to <lang> لتعيين لغة الهدف عند إدخال نص عربي (مثل en, fr, de).\n"
        "أرسل صورة تحتوي مفردات إنجليزية وسأعيدها مترجمة للعربية."
    )
    await update.message.reply_text(msg)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("أرسل نصًا أو صورة وسأتولى الترجمة. استخدم /to لتغيير اللغة الهدف عند النص العربي.")


async def cmd_set_target(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    args = context.args or []
    if not args:
        await update.message.reply_text("استخدم: /to <رمز_اللغة> مثل en أو fr أو de")
        return
    target = args[0].lower()
    # Validate via deep-translator supported languages
    try:
        supported = GoogleTranslator.get_supported_languages(as_dict=True)
        # supported is e.g. {"english": "en", "arabic": "ar", ...}
        codes = set(supported.values())
    except Exception:
        codes = {"ar", "en", "fr", "de", "es", "it", "tr", "ru", "zh-cn", "ja"}

    if target in codes:
        USER_PREFS[chat_id] = target
        await update.message.reply_text(f"تم تعيين اللغة الهدف للنص العربي إلى: {target}")
    else:
        await update.message.reply_text("رمز اللغة غير مدعوم. جرّب مثلًا: en, fr, de, es, it")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    default_rev = os.getenv("DEFAULT_AR_REVERSE_TARGET", "en").lower()
    target = USER_PREFS.get(chat_id, default_rev) if is_arabic_text(text) else "ar"

    try:
        translated = translate_text(text, target)
        await update.message.reply_text(translated)
    except Exception as e:
        logger.exception("Translation error: %s", e)
        await update.message.reply_text("حدث خطأ أثناء الترجمة. حاول لاحقًا.")


def _tesseract_available() -> bool:
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ensure_tesseract_cmd_from_env() -> None:
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def _translate_img_with_tesseract(img: Image.Image) -> Optional[Image.Image]:
    try:
        _ensure_tesseract_cmd_from_env()
        data = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.DICT)
        n = len(data["text"])
        if n == 0:
            return None

        # Prepare overlay
        base = img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        for i in range(n):
            text = data["text"][i].strip()
            conf = float(data["conf"][i]) if data["conf"][i] not in ("-1", "-1.0", "") else -1.0
            if not text or conf < 50:
                continue
            # Only translate alphabetic words to reduce noise
            if not re.search(r"[A-Za-z]", text):
                continue

            try:
                ar = translate_text(text, "ar")
            except Exception:
                ar = text

            ar_shaped = shape_arabic(ar)
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

            # Background rectangle with slight transparency
            draw.rectangle([(x, y), (x + w, y + h)], fill=(255, 255, 255, 180))

            # Fit font size to box width
            font_size = max(10, h - 2)
            font = get_font(font_size)
            tw, th = draw.textsize(ar_shaped, font=font)
            # Reduce font size if too wide
            while tw > w and font_size > 8:
                font_size -= 1
                font = get_font(font_size)
                tw, th = draw.textsize(ar_shaped, font=font)

            # Center text within box
            tx = x + (w - tw) / 2
            ty = y + (h - th) / 2
            draw.text((tx, ty), ar_shaped, font=font, fill=(0, 0, 0, 255))

        result = Image.alpha_composite(base, overlay).convert("RGB")
        return result
    except pytesseract.pytesseract.TesseractNotFoundError:
        logger.warning("Tesseract not found.")
        return None
    except Exception as e:
        logger.exception("Tesseract OCR failed: %s", e)
        return None


def _ocrspace_parse_and_overlay(img_bytes: bytes, img_size: Tuple[int, int]) -> Optional[Image.Image]:
    api_key = os.getenv("OCR_SPACE_API_KEY", "helloworld")
    url = "https://api.ocr.space/parse/image"
    try:
        resp = requests.post(
            url,
            data={
                "language": "eng",
                "isOverlayRequired": True,
                "OCREngine": 2,
            },
            files={"file": ("image.jpg", img_bytes)},
            headers={"apikey": api_key},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        parsed = data.get("ParsedResults", [{}])[0]
        overlay = parsed.get("TextOverlay", {})
        lines = overlay.get("Lines", [])
        if not lines:
            return None

        base = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        draw_ov = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(draw_ov)

        for line in lines:
            for word in line.get("Words", []):
                w_text = word.get("WordText", "").strip()
                if not w_text:
                    continue
                if not re.search(r"[A-Za-z]", w_text):
                    continue
                try:
                    ar = translate_text(w_text, "ar")
                except Exception:
                    ar = w_text
                ar_shaped = shape_arabic(ar)
                x = int(word.get("Left", 0))
                y = int(word.get("Top", 0))
                w = int(word.get("Width", 0))
                h = int(word.get("Height", 0))

                draw.rectangle([(x, y), (x + w, y + h)], fill=(255, 255, 255, 180))
                font_size = max(10, h - 2)
                font = get_font(font_size)
                tw, th = draw.textsize(ar_shaped, font=font)
                while tw > w and font_size > 8:
                    font_size -= 1
                    font = get_font(font_size)
                    tw, th = draw.textsize(ar_shaped, font=font)
                tx = x + (w - tw) / 2
                ty = y + (h - th) / 2
                draw.text((tx, ty), ar_shaped, font=font, fill=(0, 0, 0, 255))

        result = Image.alpha_composite(base, draw_ov).convert("RGB")
        return result
    except Exception as e:
        logger.exception("OCR.Space error: %s", e)
        return None


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if not update.message or not update.message.photo:
            return
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        b = await file.download_as_bytearray()

        img = Image.open(io.BytesIO(b)).convert("RGB")

        result_img: Optional[Image.Image] = None

        use_ocr = get_env_bool("ENABLE_OCR", True)
        preferred = os.getenv("PREFERRED_OCR", "tesseract").lower()

        if use_ocr:
            if preferred == "tesseract" and _tesseract_available():
                result_img = _translate_img_with_tesseract(img)
            if result_img is None:
                # Fallback to OCR.Space API (free key 'helloworld' has strict limits)
                result_img = _ocrspace_parse_and_overlay(b, img.size)

        if result_img is None:
            await update.message.reply_text("لم أتمكن من استخراج النص من الصورة. جرّب صورة أوضح أو فعّل Tesseract.")
            return

        out = io.BytesIO()
        result_img.save(out, format="JPEG", quality=90)
        out.seek(0)
        await update.message.reply_photo(out, caption="صورة مترجمة (إنجليزي → عربي)")
    except Exception as e:
        logger.exception("Photo handling failed: %s", e)
        await update.message.reply_text("حدث خطأ أثناء معالجة الصورة.")


async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "translator_bot"})


async def start_health_server(port: int) -> None:
    app = web.Application()
    app.add_routes([web.get("/health", health_handler), web.get("/", health_handler)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    logger.info("Health endpoint running on port %d", port)
    # Keep running indefinitely
    while True:
        await asyncio.sleep(3600)


async def main() -> None:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Environment variable BOT_TOKEN is required")

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("to", cmd_set_target))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    port = int(os.getenv("PORT", "8080"))

    await asyncio.gather(
        application.run_polling(close_loop=False, allowed_updates=Update.ALL_TYPES),
        start_health_server(port),
    )


if __name__ == "__main__":
    # Optional: set tesseract binary path if provided
    _ensure_tesseract_cmd_from_env()
    asyncio.run(main())
