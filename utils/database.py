# Modified version of database.py - keeping necessary tables
import logging
import json
import uuid
import time
from sqlalchemy import create_engine, MetaData, Table, Column, String, JSON, Float

logger = logging.getLogger(__name__)

# Globals
meta = MetaData()
engine = None
tables = {}

def initialize_db(config):
    """Initialize database connection and necessary tables (idempotent)."""
    global engine, tables

    if engine is not None:
        return  # already initialized

    try:
        # Create engine
        engine = create_engine(config.DATABASE_URL)

        # Define tables we want to keep
        core_tables = {
            config.DB_TABLE_USER_PREFS: [('_id', String, True), ('data', JSON), ('timestamp', Float)],
            config.DB_TABLE_SEARCHES: [('_id', String, True), ('data', JSON), ('timestamp', Float)],
            config.DB_TABLE_PROCESSES: [('_id', String, True), ('data', JSON), ('timestamp', Float)],
        }

        # Create table definitions
        for name, cols in core_tables.items():
            if name in tables:
                continue
            tables[name] = Table(
                name,
                meta,
                *[Column(col_name, col_type, primary_key=is_pk) for col_name, col_type, is_pk in
                  [(c[0], c[1], c[2] if len(c) == 3 else False) for c in cols]]
            )

        # Create tables if not exists
        meta.create_all(engine)
        logger.info("Database tables initialized (user_prefs, searches, processes).")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

# Keep save_data and find_data functions for general use
def save_data(table_name, data_dict, config):
    """Save data to the specified table with upsert behavior for duplicates."""
    global engine, tables
    if engine is None:
        initialize_db(config)

    try:
        doc_id = data_dict.get('id', str(uuid.uuid4()))
        record = {
            '_id': doc_id,
            'data': data_dict,
            'timestamp': data_dict.get('timestamp', time.time())
        }

        # Use an upsert operation instead of plain insert
        with engine.begin() as conn:
            # Check if record exists
            stmt = tables[table_name].select().where(tables[table_name].c._id == doc_id)
            existing = conn.execute(stmt).fetchone()

            if existing:
                # Update existing record
                update_stmt = tables[table_name].update().where(tables[table_name].c._id == doc_id).values(
                    data=data_dict,
                    timestamp=record['timestamp']
                )
                conn.execute(update_stmt)
            else:
                # Insert new record
                conn.execute(tables[table_name].insert().values(**record))

        return doc_id
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        return None

def find_data(table_name, query, config):
    """Find data in the specified table based on a query."""
    global engine, tables
    if engine is None:
        initialize_db(config)

    try:
        table = tables[table_name]
        with engine.begin() as conn:
            if 'user_id' in query:
                stmt = table.select().where(table.c._id == str(query['user_id']))
            else:
                logger.error(f"Unsupported query: {query}")
                return None

            row = conn.execute(stmt).fetchone()
            return row[1] if row else None
    except Exception as e:
        logger.error(f"Error querying database: {e}")
        return None