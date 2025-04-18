"""
Updated main application entry point for the Enhanced Restaurant Recommendation App.

This module initializes all components and starts the Telegram bot.
"""
import os
import argparse
import logging
import asyncio

import config
from openai_agent import RestaurantFormattingAgent
from enhanced_orchestrator import EnhancedRestaurantRecommender
from enhanced_telegram_bot import EnhancedRestaurantBot
from editor_agent import RestaurantEditorAgent

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Restaurant Recommendation App")

    # Add arguments
    parser.add_argument("--webhook", action="store_true", help="Run in webhook mode instead of polling")
    parser.add_argument("--webhook-url", type=str, help="Webhook URL (required for webhook mode)")
    parser.add_argument("--webhook-path", type=str, default="/webhook", help="Webhook path")
    parser.add_argument("--port", type=int, default=8443, help="Port for webhook server")
    parser.add_argument("--no-tracing", action="store_true", help="Disable LangSmith tracing")
    parser.add_argument("--max-results", type=int, help="Maximum number of restaurant results to return")

    return parser.parse_args()

async def run_webhook_mode(args):
    """Run the application in webhook mode."""
    logger.info("Starting enhanced application in webhook mode")

    # Override max results if specified in args
    if args.max_results:
        config.PERPLEXITY_MAX_RESULTS = args.max_results
        logger.info(f"Maximum results overridden to: {args.max_results}")

    # Initialize editor agent
    editor_agent = RestaurantEditorAgent()

    # Initialize formatting agent
    formatting_agent = RestaurantFormattingAgent()

    # Create enhanced recommender with tracing enabled/disabled according to args
    recommender = EnhancedRestaurantRecommender(enable_tracing=not args.no_tracing)

    # Create and run the Telegram bot
    bot = EnhancedRestaurantBot(recommender=recommender)

    webhook_url = args.webhook_url or config.TELEGRAM_WEBHOOK_URL
    if not webhook_url:
        logger.error("Webhook URL is required for webhook mode")
        return

    await bot.run_webhook(
        webhook_url=webhook_url,
        webhook_path=args.webhook_path,
        port=args.port
    )

    # Run forever
    await asyncio.Event().wait()

def run_polling_mode(args):
    """Run the application in polling mode."""
    logger.info("Starting enhanced application in polling mode")

    # Override max results if specified in args
    if args.max_results:
        config.PERPLEXITY_MAX_RESULTS = args.max_results
        logger.info(f"Maximum results overridden to: {args.max_results}")

    # Initialize editor agent
    editor_agent = RestaurantEditorAgent()

    # Initialize formatting agent
    formatting_agent = RestaurantFormattingAgent()

    # Create enhanced recommender with tracing enabled/disabled according to args
    recommender = EnhancedRestaurantRecommender(enable_tracing=not args.no_tracing)

    # Create and run the Telegram bot
    bot = EnhancedRestaurantBot(recommender=recommender)
    bot.run_polling()

def main():
    """Main entry point for the application."""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║  Enhanced Restaurant Recommendation App                ║
    ║  Powered by Perplexity, OpenAI Editor, and LangChain  ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    # Parse command line arguments
    args = parse_arguments()

    # Debug configuration
    print(f"Current environment settings:")
    print(f"PERPLEXITY_API_KEY set: {'Yes' if config.PERPLEXITY_API_KEY else 'No'}")
    print(f"PERPLEXITY_MODEL: {config.PERPLEXITY_MODEL}")
    print(f"PERPLEXITY_MAX_RESULTS: {config.PERPLEXITY_MAX_RESULTS}")
    print(f"OPENAI_API_KEY set: {'Yes' if config.OPENAI_API_KEY else 'No'}")
    print(f"OPENAI_MODEL: {config.OPENAI_MODEL}")

    # Verify required API keys
    try:
        # Ensure we have Perplexity and OpenAI keys
        if not config.PERPLEXITY_API_KEY:
            raise ValueError("PERPLEXITY_API_KEY is required")
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        if not config.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        config.validate_configuration()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        exit(1)

    # Set LangSmith tracing based on args
    if args.no_tracing:
        os.environ["LANGCHAIN_TRACING"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"

    # Run in webhook or polling mode
    if args.webhook:
        asyncio.run(run_webhook_mode(args))
    else:
        run_polling_mode(args)

if __name__ == "__main__":
    main()