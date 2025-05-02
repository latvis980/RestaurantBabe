# Add this to telegram_bot.py

# ---------------------------------------------------------------------------
# ADMIN FEATURES
# ---------------------------------------------------------------------------
import os
from sqlalchemy import text
from urllib.parse import urlparse

# Admin ID - set this to your Telegram user ID for security
ADMIN_IDS = [int(id.strip()) for id in os.environ.get("ADMIN_IDS", "").split(",") if id.strip()]

# Admin state to track conversation flow
admin_state = {}

def is_admin(user_id):
    """Check if the user is an admin"""
    return user_id in ADMIN_IDS

# Admin command handlers
@bot.message_handler(commands=["admin"])
def handle_admin(msg):
    """Show admin commands menu"""
    user_id = msg.from_user.id

    if not is_admin(user_id):
        bot.reply_to(msg, "❌ This command is only available to administrators.")
        return

    menu_text = """
<b>Admin Commands:</b>

/sources [city] - Manage sources for a specific city
/add_admin [user_id] - Add a new admin (Super admin only)
/stats - View system statistics
    """

    bot.reply_to(msg, menu_text, parse_mode="HTML")

@bot.message_handler(commands=["sources"])
def handle_sources_command(msg):
    """Handle the sources command to list sources for a city"""
    user_id = msg.from_user.id

    if not is_admin(user_id):
        bot.reply_to(msg, "❌ This command is only available to administrators.")
        return

    # Get the city from command arguments
    command_args = msg.text.split(maxsplit=1)

    if len(command_args) < 2:
        # No city specified, ask for it
        bot.reply_to(msg, "Please specify a city, e.g., <code>/sources Paris</code>", parse_mode="HTML")
        return

    city = command_args[1].strip()
    show_sources_for_city(msg.chat.id, city)

def show_sources_for_city(chat_id, city):
    """Show sources for a specific city"""
    # Create sanitized table name for city-specific sources
    table_name = f"sources_{city.lower().replace(' ', '_').replace('-', '_')}"

    # Check if the table exists
    try:
        with engine.begin() as conn:
            # Try to query the table
            result = conn.execute(text(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')"))
            table_exists = result.scalar()

            if not table_exists:
                bot.send_message(chat_id, f"❌ No sources table found for city: {city}")
                return

            # Get all sources from the table
            result = conn.execute(text(f"SELECT data FROM {table_name}"))
            sources_data = result.fetchall()

            if not sources_data:
                bot.send_message(chat_id, f"No sources found for {city}. Add some with /add_source {city} [url] [name]")
                return

            # Extract sources from the data
            sources = []
            for row in sources_data:
                data = row[0]
                if isinstance(data, dict):
                    if "sources" in data:
                        sources.extend(data["sources"])
                    else:
                        sources.append(data)

            # Format and send the sources list
            if not sources:
                bot.send_message(chat_id, f"No sources found for {city}. Add some with /add_source {city} [url] [name]")
                return

            # Set the admin state to track the current city
            admin_state[chat_id] = {"action": "viewing_sources", "city": city, "sources": sources}

            # Create the sources message
            sources_text = f"<b>Sources for {city}:</b>\n\n"

            for i, source in enumerate(sources, 1):
                name = source.get("name", "Unnamed Source")
                url = source.get("url", "No URL")
                source_type = source.get("type", "Unknown Type")

                sources_text += f"{i}. <b>{name}</b>\n"
                sources_text += f"   URL: {url}\n"
                sources_text += f"   Type: {source_type}\n\n"

            sources_text += "\nCommands:\n"
            sources_text += f"/add_source {city} [url] [name] - Add a new source\n"
            sources_text += f"/delete_source {city} [number] - Delete a source by number\n"

            bot.send_message(chat_id, sources_text, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error showing sources for {city}: {e}")
        bot.send_message(chat_id, f"❌ Error retrieving sources for {city}: {str(e)}")

@bot.message_handler(commands=["add_source"])
def handle_add_source(msg):
    """Handle adding a new source for a city"""
    user_id = msg.from_user.id

    if not is_admin(user_id):
        bot.reply_to(msg, "❌ This command is only available to administrators.")
        return

    # Parse command: /add_source [city] [url] [name]
    parts = msg.text.split(maxsplit=3)

    if len(parts) < 3:
        bot.reply_to(msg, "Usage: /add_source [city] [url] [name (optional)]")
        return

    city = parts[1].strip()
    url = parts[2].strip()
    name = parts[3].strip() if len(parts) > 3 else None

    # Validate URL
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            bot.reply_to(msg, "❌ Invalid URL. Please provide a complete URL including http:// or https://")
            return
    except Exception:
        bot.reply_to(msg, "❌ Invalid URL format.")
        return

    # If no name is provided, extract from domain
    if not name:
        domain = urlparse(url).netloc
        name = domain.replace("www.", "").split(".")[0].capitalize()

    # Add the source
    add_source_to_city(msg.chat.id, city, url, name)

def add_source_to_city(chat_id, city, url, name):
    """Add a new source to the city database"""
    # Create sanitized table name for city-specific sources
    table_name = f"sources_{city.lower().replace(' ', '_').replace('-', '_')}"

    try:
        # Determine source type based on URL or just use "Local Publication" as default
        domain = urlparse(url).netloc.lower()

        source_type = "Local Publication"
        if "blog" in domain or "blogger" in domain:
            source_type = "Food Blog"
        elif "guide" in domain or "michelin" in domain:
            source_type = "Food Guide"
        elif "news" in domain or "times" in domain or "post" in domain:
            source_type = "News Publication"

        # Create new source object
        new_source = {
            "name": name,
            "url": url,
            "type": source_type,
            "city": city,
            "language": "en"  # Default to English, you might want to detect this
        }

        with engine.begin() as conn:
            # Check if table exists
            result = conn.execute(text(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')"))
            table_exists = result.scalar()

            if not table_exists:
                # Create table if it doesn't exist
                conn.execute(text(f"""
                CREATE TABLE {table_name} (
                    _id VARCHAR PRIMARY KEY,
                    data JSONB,
                    timestamp FLOAT
                )
                """))

                # Insert initial record with an array of sources
                conn.execute(text(f"""
                INSERT INTO {table_name} (_id, data, timestamp)
                VALUES ('city_sources', '{"city": "{city}", "sources": []}', {time.time()})
                """))

            # Get current data
            result = conn.execute(text(f"SELECT data FROM {table_name} WHERE _id = 'city_sources'"))
            data = result.fetchone()

            if data:
                current_data = data[0]
                if "sources" in current_data:
                    # Add new source to existing sources
                    current_data["sources"].append(new_source)

                    # Update the record
                    conn.execute(text(f"""
                    UPDATE {table_name}
                    SET data = :data, timestamp = :timestamp
                    WHERE _id = 'city_sources'
                    """), {"data": current_data, "timestamp": time.time()})
                else:
                    # Create sources array
                    current_data["sources"] = [new_source]

                    # Update the record
                    conn.execute(text(f"""
                    UPDATE {table_name}
                    SET data = :data, timestamp = :timestamp
                    WHERE _id = 'city_sources'
                    """), {"data": current_data, "timestamp": time.time()})
            else:
                # Insert new record
                conn.execute(text(f"""
                INSERT INTO {table_name} (_id, data, timestamp)
                VALUES ('city_sources', :data, :timestamp)
                """), {"data": {"city": city, "sources": [new_source]}, "timestamp": time.time()})

        bot.send_message(chat_id, f"✅ Added source: <b>{name}</b> for {city}", parse_mode="HTML")

        # Refresh the sources list
        show_sources_for_city(chat_id, city)

    except Exception as e:
        logger.error(f"Error adding source for {city}: {e}")
        bot.send_message(chat_id, f"❌ Error adding source: {str(e)}")

@bot.message_handler(commands=["delete_source"])
def handle_delete_source(msg):
    """Handle deleting a source for a city"""
    user_id = msg.from_user.id

    if not is_admin(user_id):
        bot.reply_to(msg, "❌ This command is only available to administrators.")
        return

    # Parse command: /delete_source [city] [number]
    parts = msg.text.split()

    if len(parts) < 3:
        bot.reply_to(msg, "Usage: /delete_source [city] [number]")
        return

    city = parts[1].strip()

    try:
        source_index = int(parts[2]) - 1  # Convert to zero-based index
        delete_source_from_city(msg.chat.id, city, source_index)
    except ValueError:
        bot.reply_to(msg, "❌ Invalid source number. Please provide a valid number.")

def delete_source_from_city(chat_id, city, source_index):
    """Delete a source from the city database by index"""
    # Create sanitized table name for city-specific sources
    table_name = f"sources_{city.lower().replace(' ', '_').replace('-', '_')}"

    try:
        with engine.begin() as conn:
            # Check if table exists
            result = conn.execute(text(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')"))
            table_exists = result.scalar()

            if not table_exists:
                bot.send_message(chat_id, f"❌ No sources table found for city: {city}")
                return

            # Get current data
            result = conn.execute(text(f"SELECT data FROM {table_name} WHERE _id = 'city_sources'"))
            data = result.fetchone()

            if not data:
                bot.send_message(chat_id, f"❌ No sources found for {city}")
                return

            current_data = data[0]

            if "sources" not in current_data or not current_data["sources"]:
                bot.send_message(chat_id, f"❌ No sources found for {city}")
                return

            sources = current_data["sources"]

            if source_index < 0 or source_index >= len(sources):
                bot.send_message(chat_id, f"❌ Invalid source number. Valid range is 1-{len(sources)}")
                return

            # Get the source that will be deleted
            deleted_source = sources[source_index]
            deleted_name = deleted_source.get("name", "Unnamed Source")

            # Remove the source
            current_data["sources"].pop(source_index)

            # Update the database
            conn.execute(text(f"""
            UPDATE {table_name}
            SET data = :data, timestamp = :timestamp
            WHERE _id = 'city_sources'
            """), {"data": current_data, "timestamp": time.time()})

            bot.send_message(chat_id, f"✅ Deleted source: <b>{deleted_name}</b> from {city}", parse_mode="HTML")

            # Refresh the sources list
            show_sources_for_city(chat_id, city)

    except Exception as e:
        logger.error(f"Error deleting source for {city}: {e}")
        bot.send_message(chat_id, f"❌ Error deleting source: {str(e)}")

@bot.message_handler(commands=["add_admin"])
def handle_add_admin(msg):
    """Add a new admin"""
    user_id = msg.from_user.id

    # Only existing admins can add new admins
    if not is_admin(user_id):
        bot.reply_to(msg, "❌ This command is only available to administrators.")
        return

    # Parse command: /add_admin [user_id]
    parts = msg.text.split()

    if len(parts) < 2:
        bot.reply_to(msg, "Usage: /add_admin [user_id]")
        return

    try:
        new_admin_id = int(parts[1].strip())

        # Add the new admin ID to the environment variable
        admin_ids_str = os.environ.get("ADMIN_IDS", "")
        current_admin_ids = [int(id.strip()) for id in admin_ids_str.split(",") if id.strip()]

        if new_admin_id in current_admin_ids:
            bot.reply_to(msg, f"User ID {new_admin_id} is already an admin.")
            return

        current_admin_ids.append(new_admin_id)

        # Update the global ADMIN_IDS list
        global ADMIN_IDS
        ADMIN_IDS = current_admin_ids

        # Note: In a production environment, you would need to update
        # the actual environment variable on your hosting platform
        bot.reply_to(msg, f"✅ Added user ID {new_admin_id} as an admin.\n\n⚠️ Note: This change is temporary until the bot restarts. Update your environment variables to make it permanent.")

    except ValueError:
        bot.reply_to(msg, "❌ Invalid user ID. Please provide a valid numeric ID.")

@bot.message_handler(commands=["stats"])
def handle_stats(msg):
    """Show system statistics"""
    user_id = msg.from_user.id

    if not is_admin(user_id):
        bot.reply_to(msg, "❌ This command is only available to administrators.")
        return

    try:
        with engine.begin() as conn:
            # Count total cities with sources
            result = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name LIKE 'sources_%'
            """))
            cities_count = result.scalar()

            # Count total restaurants
            result = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name LIKE 'restaurants_%'
            """))
            restaurant_tables_count = result.scalar()

            # Count total searches
            result = conn.execute(text(f"SELECT COUNT(*) FROM {config.DB_TABLE_SEARCHES}"))
            searches_count = result.scalar()

            # Count total processes
            result = conn.execute(text(f"SELECT COUNT(*) FROM {config.DB_TABLE_PROCESSES}"))
            processes_count = result.scalar()

            stats_text = f"""
<b>System Statistics:</b>

📍 Cities with sources: {cities_count}
🍽 Restaurant tables: {restaurant_tables_count}
🔍 Total searches: {searches_count}
⚙️ Total processes: {processes_count}
            """

            bot.reply_to(msg, stats_text, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        bot.reply_to(msg, f"❌ Error getting statistics: {str(e)}")