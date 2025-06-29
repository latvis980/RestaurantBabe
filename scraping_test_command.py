# scraping_test_command.py
import telebot
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Any
import logging
from langchain_core.tracers.context import tracing_v2_enabled

logger = logging.getLogger("restaurant-recommender.scraping_test")

def add_test_scraping_command(bot: telebot.TeleBot, config, orchestrator):
    """
    Add the /test_scrape command to the Telegram bot.
    This command allows admins to test the scraping process and dump results.
    """

    @bot.message_handler(commands=['test_scrape'])
    def handle_test_scrape(message):
        """
        Handle the /test_scrape command.
        Format: /test_scrape your search query here
        Example: /test_scrape best ramen restaurants in Tokyo
        """

        # Check if user is admin (you can add admin chat IDs to config)
        if hasattr(config, 'ADMIN_CHAT_ID') and str(message.chat.id) != str(config.ADMIN_CHAT_ID):
            bot.reply_to(message, "❌ This command is only available for admins.")
            return

        # Extract the search query from the command
        command_parts = message.text.split(' ', 1)
        if len(command_parts) < 2:
            bot.reply_to(message, 
                "❌ Please provide a search query.\n"
                "Format: `/test_scrape your search query here`\n"
                "Example: `/test_scrape best ramen restaurants in Tokyo`", 
                parse_mode='Markdown')
            return

        search_query = command_parts[1].strip()

        # Send initial response
        status_message = bot.reply_to(message, 
            f"🔍 **Testing scraping process**\n"
            f"Query: `{search_query}`\n"
            f"Starting search and scraping...", 
            parse_mode='Markdown')

        try:
            # Run the scraping process
            with tracing_v2_enabled(project_name="restaurant-recommender-test"):
                logger.info(f"Starting test scraping for query: {search_query}")

                # Update status
                bot.edit_message_text(
                    f"🔍 **Testing scraping process**\n"
                    f"Query: `{search_query}`\n"
                    f"⏳ Searching for sources...", 
                    chat_id=status_message.chat.id,
                    message_id=status_message.message_id,
                    parse_mode='Markdown')

                # Get search results using the orchestrator's search agent
                search_agent = orchestrator.search_agent
                search_results = search_agent.search(search_query)

                # Update status
                bot.edit_message_text(
                    f"🔍 **Testing scraping process**\n"
                    f"Query: `{search_query}`\n"
                    f"✅ Found {len(search_results.get('results', []))} sources\n"
                    f"⏳ Scraping content...", 
                    chat_id=status_message.chat.id,
                    message_id=status_message.message_id,
                    parse_mode='Markdown')

                # Get scraping results using the orchestrator's scraper
                scraper_agent = orchestrator.scraper_agent
                scraping_results = []

                for result in search_results.get('results', []):
                    try:
                        scraped_data = scraper_agent.scrape_url(result['url'])
                        if scraped_data and scraped_data.strip():
                            scraping_results.append({
                                'url': result['url'],
                                'title': result.get('title', 'No title'),
                                'description': result.get('description', 'No description'),
                                'scraped_content': scraped_data,
                                'scraped_at': datetime.now().isoformat()
                            })
                    except Exception as scrape_error:
                        logger.error(f"Error scraping {result['url']}: {scrape_error}")
                        scraping_results.append({
                            'url': result['url'],
                            'title': result.get('title', 'No title'),
                            'description': result.get('description', 'No description'),
                            'scraped_content': None,
                            'error': str(scrape_error),
                            'scraped_at': datetime.now().isoformat()
                        })

                # Create the dump data exactly as it would go to list_analyser
                dump_data = {
                    'test_metadata': {
                        'query': search_query,
                        'timestamp': datetime.now().isoformat(),
                        'admin_chat_id': message.chat.id,
                        'total_sources_found': len(search_results.get('results', [])),
                        'successfully_scraped': len([r for r in scraping_results if r.get('scraped_content')]),
                        'failed_scrapes': len([r for r in scraping_results if not r.get('scraped_content')])
                    },
                    'raw_search_results': search_results,
                    'scraped_data': scraping_results,
                    'ready_for_list_analyser': prepare_for_list_analyser(scraping_results)
                }

                # Update final status
                successful_scrapes = len([r for r in scraping_results if r.get('scraped_content')])
                bot.edit_message_text(
                    f"🔍 **Testing scraping process**\n"
                    f"Query: `{search_query}`\n"
                    f"✅ Found {len(search_results.get('results', []))} sources\n"
                    f"✅ Successfully scraped: {successful_scrapes}\n"
                    f"📄 Preparing dump files...", 
                    chat_id=status_message.chat.id,
                    message_id=status_message.message_id,
                    parse_mode='Markdown')

                # Create temporary files for the dumps
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # JSON dump (complete data)
                json_filename = f"scraping_test_{timestamp}.json"
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as json_file:
                    json.dump(dump_data, json_file, indent=2, ensure_ascii=False)
                    json_temp_path = json_file.name

                # TXT dump (formatted for human reading)
                txt_filename = f"scraping_test_{timestamp}.txt"
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as txt_file:
                    txt_content = format_dump_as_text(dump_data)
                    txt_file.write(txt_content)
                    txt_temp_path = txt_file.name

                # Send the files
                with open(json_temp_path, 'rb') as json_file:
                    bot.send_document(
                        message.chat.id, 
                        json_file, 
                        caption=f"📄 **Complete scraping test results (JSON)**\n"
                                f"Query: `{search_query}`\n"
                                f"This contains the exact data that would go to list_analyser",
                        parse_mode='Markdown',
                        visible_file_name=json_filename
                    )

                with open(txt_temp_path, 'rb') as txt_file:
                    bot.send_document(
                        message.chat.id, 
                        txt_file, 
                        caption=f"📄 **Human-readable scraping test results (TXT)**\n"
                                f"Query: `{search_query}`\n"
                                f"Formatted for easy reading",
                        parse_mode='Markdown',
                        visible_file_name=txt_filename
                    )

                # Clean up temporary files
                os.unlink(json_temp_path)
                os.unlink(txt_temp_path)

                # Final success message
                bot.edit_message_text(
                    f"✅ **Scraping test completed!**\n"
                    f"Query: `{search_query}`\n"
                    f"📊 Results: {successful_scrapes}/{len(search_results.get('results', []))} successful\n"
                    f"📄 Files sent above ⬆️", 
                    chat_id=status_message.chat.id,
                    message_id=status_message.message_id,
                    parse_mode='Markdown')

                logger.info(f"Test scraping completed successfully for query: {search_query}")

        except Exception as e:
            logger.error(f"Error in test scraping: {e}")
            bot.edit_message_text(
                f"❌ **Scraping test failed**\n"
                f"Query: `{search_query}`\n"
                f"Error: `{str(e)}`", 
                chat_id=status_message.chat.id,
                message_id=status_message.message_id,
                parse_mode='Markdown')


def prepare_for_list_analyser(scraping_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare scraped data in the exact format that would be sent to list_analyser.
    This mimics what your main orchestrator would send.
    """
    prepared_data = []

    for result in scraping_results:
        if result.get('scraped_content'):  # Only include successfully scraped content
            prepared_item = {
                'source_url': result['url'],
                'source_title': result['title'],
                'source_description': result.get('description', ''),
                'content': result['scraped_content'],
                'metadata': {
                    'scraped_at': result.get('scraped_at'),
                    'content_length': len(result['scraped_content']),
                    'source_domain': extract_domain(result['url'])
                }
            }
            prepared_data.append(prepared_item)

    return prepared_data


def extract_domain(url: str) -> str:
    """Extract domain from URL for metadata."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except:
        return 'unknown'


def format_dump_as_text(dump_data: Dict[str, Any]) -> str:
    """Format the dump data as human-readable text."""

    metadata = dump_data.get('test_metadata', {})
    scraped_data = dump_data.get('scraped_data', [])

    txt_content = f"""RESTAURANT SCRAPING TEST RESULTS
{'='*50}

QUERY: {metadata.get('query', 'Unknown')}
TIMESTAMP: {metadata.get('timestamp', 'Unknown')}
TOTAL SOURCES FOUND: {metadata.get('total_sources_found', 0)}
SUCCESSFULLY SCRAPED: {metadata.get('successfully_scraped', 0)}
FAILED SCRAPES: {metadata.get('failed_scrapes', 0)}

{'='*50}
SCRAPED CONTENT (Ready for list_analyser)
{'='*50}

"""

    for i, result in enumerate(scraped_data, 1):
        txt_content += f"\n{'-'*30} SOURCE {i} {'-'*30}\n"
        txt_content += f"URL: {result.get('url', 'Unknown')}\n"
        txt_content += f"TITLE: {result.get('title', 'No title')}\n"
        txt_content += f"DESCRIPTION: {result.get('description', 'No description')}\n"
        txt_content += f"SCRAPED AT: {result.get('scraped_at', 'Unknown')}\n"

        if result.get('error'):
            txt_content += f"ERROR: {result.get('error')}\n"
            txt_content += "CONTENT: [SCRAPING FAILED]\n"
        elif result.get('scraped_content'):
            content = result.get('scraped_content', '')
            txt_content += f"CONTENT LENGTH: {len(content)} characters\n"
            txt_content += f"CONTENT:\n{content}\n"
        else:
            txt_content += "CONTENT: [NO CONTENT]\n"

    txt_content += f"\n{'='*50}\n"
    txt_content += "END OF SCRAPING TEST RESULTS\n"
    txt_content += f"{'='*50}\n"

    return txt_content