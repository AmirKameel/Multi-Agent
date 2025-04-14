import os
import logging
import json
import requests
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
# Use a more thread-friendly Flask alternative
from quart import Quart

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get API endpoint from environment
API_BASE_URL = os.getenv("API_BASE_URL", "https://vectaraaa.onrender.com")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PORT = int(os.environ.get("PORT", 8080))

# Default approach is full_context
DEFAULT_APPROACH = "full_context"

# Initialize Quart app (async version of Flask)
app = Quart(__name__)

@app.route('/')
async def index():
    return 'Telegram Bot is running!'

@app.route('/health')
async def health():
    return {'status': 'healthy', 'message': 'Telegram bot is operational'}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    welcome_text = """مرحباً بك في نظام الاستعلام عن سياسات الشركة 👋

يمكنك استخدام هذا البوت للاستفسار عن سياسات الشركة وإجراءاتها. ما عليك سوى إرسال سؤالك، وسيقوم النظام بالبحث في وثائق السياسة وتقديم إجابة دقيقة.

الأوامر المتاحة:
/start - بدء استخدام البوت
/help - عرض المساعدة
/status - عرض حالة النظام

Welcome to the Company Policy Query System 👋

You can use this bot to inquire about company policies and procedures. Simply send your question, and the system will search the policy documents and provide an accurate answer.

Available commands:
/start - Start using the bot
/help - Display help
/status - Show system status"""

    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    help_text = """كيفية استخدام البوت:

1. اكتب سؤالك باللغة العربية أو الإنجليزية
2. انتظر الإجابة من النظام

الأوامر:
/start - بدء استخدام البوت
/help - عرض هذه المساعدة
/status - عرض حالة النظام

How to use the bot:

1. Type your question in Arabic or English
2. Wait for the system's response

Commands:
/start - Start using the bot
/help - Display this help
/status - Show system status"""

    await update.message.reply_text(help_text)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check the status of the backend system."""
    try:
        # Check health endpoint
        health_response = requests.get(f"{API_BASE_URL}/health")
        health_response.raise_for_status()
        
        # Get document info
        doc_info_response = requests.get(f"{API_BASE_URL}/document-info")
        doc_info_response.raise_for_status()
        doc_info = doc_info_response.json()
        
        if "status" in doc_info and doc_info["status"] == "No document uploaded yet":
            status_text = "✅ نظام الاستعلام يعمل\n❌ لم يتم تحميل أي وثيقة بعد\n\n✅ Query system is operational\n❌ No document uploaded yet"
        else:
            document = doc_info.get("document", {})
            title = document.get("title", "غير معروف / Unknown")
            chunk_count = doc_info.get("chunk_count", 0)
            
            status_text = f"""✅ نظام الاستعلام يعمل
📄 الوثيقة: {title}
🧩 عدد المقاطع: {chunk_count}

✅ Query system is operational
📄 Document: {title}
🧩 Chunk count: {chunk_count}"""
        
        await update.message.reply_text(status_text)
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        await update.message.reply_text(
            "❌ خطأ في الاتصال بنظام الاستعلام\n\n❌ Error connecting to the query system"
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages and query the API."""
    query_text = update.message.text
    
    # Send "typing" action
    await context.bot.send_chat_action(
        chat_id=update.effective_message.chat_id,
        action="typing"
    )
    
    # Let the user know we're processing
    processing_message = await update.message.reply_text(
        "جاري معالجة استفسارك...\n\nProcessing your query..."
    )
    
    try:
        # Prepare the request payload with default approach
        payload = {
            "query": query_text,
            "approach": DEFAULT_APPROACH
        }
        
        # Use requests in a non-blocking way
        response = await asyncio.to_thread(
            requests.post,
            f"{API_BASE_URL}/query/",
            json=payload
        )
        response.raise_for_status()  # Raise exception for non-200 responses
        
        result = response.json()
        
        answer = result.get("answer", "لم يتم العثور على إجابة / No answer found")
        processing_time = result.get("processing_time", 0)
        
        # Construct the response message without mentioning approach
        response_text = f"{answer}\n\n------\nالوقت: {processing_time:.2f} ثانية"
        
        # Delete the processing message and send the answer
        await context.bot.delete_message(
            chat_id=update.effective_message.chat_id,
            message_id=processing_message.message_id
        )
        
        # Split the message if it's too long
        if len(response_text) > 4000:
            chunks = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
            for chunk in chunks:
                await update.message.reply_text(chunk)
        else:
            await update.message.reply_text(response_text)
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        await context.bot.delete_message(
            chat_id=update.effective_message.chat_id,
            message_id=processing_message.message_id
        )
        await update.message.reply_text(
            "❌ حدث خطأ أثناء معالجة استفسارك. يرجى المحاولة مرة أخرى لاحقاً.\n\n❌ An error occurred while processing your query. Please try again later."
        )

async def setup_bot():
    """Setup the telegram bot application"""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("No TELEGRAM_BOT_TOKEN found in environment variables!")
        return None
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    
    # Add message handler for user questions
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    return application

@app.before_serving
async def on_startup():
    """Initialize bot before the web server starts"""
    app.bot_application = await setup_bot()
    if app.bot_application:
        # Start the bot
        await app.bot_application.initialize()
        await app.bot_application.start()
        await app.bot_application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Bot started successfully!")
    else:
        logger.error("Failed to initialize bot!")

@app.after_serving
async def on_shutdown():
    """Shutdown bot when the web server stops"""
    if hasattr(app, 'bot_application'):
        await app.bot_application.updater.stop()
        await app.bot_application.stop()
        await app.bot_application.shutdown()
        logger.info("Bot has been shut down")

# Alternative approach: For Render, we can use a webhook-based approach instead of polling
# This would be more reliable if polling doesn't work
if os.environ.get('WEBHOOK_MODE', 'false').lower() == 'true':
    WEBHOOK_URL = os.environ.get('WEBHOOK_URL', '')
    
    @app.route(f'/webhook', methods=['POST'])
    async def webhook():
        """Handle incoming webhook updates from Telegram"""
        if not hasattr(app, 'bot_application'):
            return {'error': 'Bot not initialized'}, 500
            
        request_data = await request.get_json()
        update = Update.de_json(request_data, app.bot_application.bot)
        await app.bot_application.process_update(update)
        return {'status': 'ok'}
    
    @app.before_serving
    async def setup_webhook():
        """Set up webhook instead of polling"""
        app.bot_application = await setup_bot()
        if app.bot_application and WEBHOOK_URL:
            await app.bot_application.bot.set_webhook(url=f"{WEBHOOK_URL}/webhook")
            logger.info(f"Webhook set to {WEBHOOK_URL}/webhook")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT)
