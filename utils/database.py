# utils/database.py
import logging
import json
import uuid
import time
from sqlalchemy import create_engine, MetaData, Table, Column, String, JSON, Float, select
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# Database connection and tables
meta = MetaData()
engine = None
tables = {}

def initialize_db(config):
    """Initialize database connection and tables"""
    global engine, tables

    if engine is not None:
        return  # Already initialized

    try:
        # Create database engine
        engine = create_engine(config.DATABASE_URL)

        # Define tables with _id as primary key (matching existing schema)
        tables[config.DB_TABLE_SOURCES] = Table(
            config.DB_TABLE_SOURCES, meta,
            Column('_id', String, primary_key=True),
            Column('data', JSON),
            Column('timestamp', Float)
        )

        tables[config.DB_TABLE_SEARCHES] = Table(
            config.DB_TABLE_SEARCHES, meta,
            Column('_id', String, primary_key=True),
            Column('data', JSON),
            Column('timestamp', Float)
        )

        tables[config.DB_TABLE_PROCESSES] = Table(
            config.DB_TABLE_PROCESSES, meta,
            Column('_id', String, primary_key=True),
            Column('data', JSON),
            Column('timestamp', Float)
        )

        tables[config.DB_TABLE_RESTAURANTS] = Table(
            config.DB_TABLE_RESTAURANTS, meta,
            Column('_id', String, primary_key=True),
            Column('data', JSON),
            Column('timestamp', Float)
        )

        # Create tables if they don't exist
        meta.create_all(engine)
        logger.info("Database tables initialized")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def ensure_city_table(city, config, table_type="restaurants"):
    """
    Ensure a city-specific table exists (can be for restaurants or sources)

    Args:
        city (str): The city name
        config: Configuration object with database settings
        table_type (str): Type of table ('restaurants' or 'sources')

    Returns:
        str: The name of the table
    """
    global engine, tables, meta

    if engine is None:
        initialize_db(config)

    # Create a sanitized table name
    table_name = f"{table_type}_{city.lower().replace(' ', '_').replace('-', '_')}"

    # Check if table is already in our registry
    if table_name in tables:
        return table_name

    # Define the table if it doesn't exist
    try:
        # Create the table definition
        tables[table_name] = Table(
            table_name, meta,
            Column('_id', String, primary_key=True),
            Column('data', JSON),
            Column('timestamp', Float)
        )

        # Create the table in the database if it doesn't exist
        tables[table_name].create(engine, checkfirst=True)
        logger.info(f"Created city-specific {table_type} table: {table_name}")

        return table_name
    except Exception as e:
        logger.error(f"Error creating {table_type} table {table_name}: {e}")
        # Fall back to the general table
        if table_type == "restaurants":
            return config.DB_TABLE_RESTAURANTS
        else:
            return config.DB_TABLE_SOURCES

def save_data(table_name, data_dict, config):
    """Save data to the specified table"""
    global engine, tables

    if engine is None:
        initialize_db(config)

    # Check if this is a city-specific table and ensure it exists
    if table_name.startswith("restaurants_"):
        city = table_name.replace("restaurants_", "").replace("_", " ")
        table_name = ensure_city_table(city, config)
    elif table_name.startswith("sources_"):
        city = table_name.replace("sources_", "").replace("_", " ")
        table_name = ensure_city_table(city, config, table_type="sources")

    try:
        # Generate a unique ID if not provided
        doc_id = data_dict.get('id', str(uuid.uuid4()))

        # Create record to insert
        record = {
            '_id': doc_id,  # Use _id instead of id
            'data': data_dict,
            'timestamp': data_dict.get('timestamp', time.time())
        }

        # Insert into database
        with engine.connect() as conn:
            table = tables[table_name]
            insert_stmt = table.insert().values(**record)
            conn.execute(insert_stmt)
            conn.commit()

        return doc_id

    except SQLAlchemyError as e:
        logger.error(f"Error saving to database: {e}")
        # Continue execution even if database saving fails
        return None

def update_data(table_name, filter_query, update_data, config):
    """
    Update data in a database table

    Args:
        table_name (str): Table name
        filter_query (dict): Filter to identify the record to update
        update_data (dict): Data to update
        config: Configuration object with database settings
    """
    global engine, tables
    if engine is None:
        initialize_db(config)

    # Check if this is a city-specific table and ensure it exists
    if table_name.startswith("restaurants_"):
        city = table_name.replace("restaurants_", "").replace("_", " ")
        table_name = ensure_city_table(city, config)
    elif table_name.startswith("sources_"):
        city = table_name.replace("sources_", "").replace("_", " ")
        table_name = ensure_city_table(city, config, table_type="sources")

    try:
        table = tables[table_name]

        # We need to convert the filter_query to work with the JSON structure
        # Since data is stored in the 'data' JSON column
        with engine.connect() as conn:
            # First, find the record to update
            if 'domain' in filter_query:
                stmt = select(table).where(table.c.data['domain'].as_string() == filter_query.get('domain'))
            elif 'location' in filter_query:
                stmt = select(table).where(table.c.data['location'].as_string() == filter_query.get('location'))
            elif 'city' in filter_query:  # Added city filter
                stmt = select(table).where(table.c.data['city'].as_string() == filter_query.get('city'))
            elif 'name' in filter_query and 'address' in filter_query:
                # For restaurant lookups by name and address
                stmt = select(table).where(
                    (table.c.data['name'].as_string() == filter_query.get('name')) &
                    (table.c.data['address'].as_string() == filter_query.get('address'))
                )
            else:
                # For other query types, this would need to be expanded
                logger.error(f"Unsupported filter query: {filter_query}")
                return False

            result = conn.execute(stmt).fetchone()

            if result:
                # Record exists, update it
                record_id = result[0]  # Get the _id

                # Update the data
                update_stmt = table.update().where(table.c._id == record_id).values(
                    data=update_data,
                    timestamp=update_data.get('evaluated_at', 0) or update_data.get('timestamp', 0) or time.time()
                )

                conn.execute(update_stmt)
                conn.commit()
                return True
            else:
                # Record not found
                logger.warning(f"No record found for update: {filter_query}")
                return False

    except SQLAlchemyError as e:
        logger.error(f"Error updating database: {e}")
        return False

def find_data(table_name, query, config):
    """Find data in the specified table"""
    global engine, tables
    if engine is None:
        initialize_db(config)

    # Check if this is a city-specific table and ensure it exists
    if table_name.startswith("restaurants_"):
        city = table_name.replace("restaurants_", "").replace("_", " ")
        table_name = ensure_city_table(city, config)
    elif table_name.startswith("sources_"):
        city = table_name.replace("sources_", "").replace("_", " ")
        table_name = ensure_city_table(city, config, table_type="sources")

    try:
        table = tables[table_name]
        # Convert query to SQL
        with engine.connect() as conn:
            # Build the query based on the provided filter
            if 'location' in query:
                stmt = select(table.c.data).where(table.c.data['location'].as_string() == query.get('location'))
            elif 'domain' in query:
                stmt = select(table.c.data).where(table.c.data['domain'].as_string() == query.get('domain'))
            elif 'city' in query:
                # For finding all restaurants in a city
                stmt = select(table.c.data).where(table.c.data['city'].as_string() == query.get('city'))
            elif 'name' in query and 'address' in query:
                # For restaurant lookups by name and address
                stmt = select(table.c.data).where(
                    (table.c.data['name'].as_string() == query.get('name')) &
                    (table.c.data['address'].as_string() == query.get('address'))
                )
            else:
                # For other query types, this would need to be expanded
                logger.error(f"Unsupported query: {query}")
                return None

            result = conn.execute(stmt).fetchone()
            if result:
                return result[0]
            return None
    except SQLAlchemyError as e:
        logger.error(f"Error querying database: {e}")
        return None

def find_all_data(table_name, query, config, limit=100):
    """Find multiple records in the specified table"""
    global engine, tables
    if engine is None:
        initialize_db(config)

    # Check if this is a city-specific table and ensure it exists
    if table_name.startswith("restaurants_"):
        city = table_name.replace("restaurants_", "").replace("_", " ")
        table_name = ensure_city_table(city, config)
    elif table_name.startswith("sources_"):
        city = table_name.replace("sources_", "").replace("_", " ")
        table_name = ensure_city_table(city, config, table_type="sources")

    try:
        table = tables[table_name]

        # Convert query to SQL
        with engine.connect() as conn:
            # Build the query based on the provided filter
            if 'city' in query:
                stmt = select(table.c.data).where(
                    table.c.data['city'].as_string() == query.get('city')
                ).limit(limit)
            elif 'location' in query:
                stmt = select(table.c.data).where(
                    table.c.data['location'].as_string() == query.get('location')
                ).limit(limit)
            elif 'domain' in query:
                stmt = select(table.c.data).where(
                    table.c.data['domain'].as_string() == query.get('domain')
                ).limit(limit)
            else:
                # For other query types
                logger.error(f"Unsupported query for find_all: {query}")
                return []

            results = conn.execute(stmt).fetchall()
            if results:
                return [result[0] for result in results]
            return []
    except SQLAlchemyError as e:
        logger.error(f"Error in find_all query: {e}")
        return []