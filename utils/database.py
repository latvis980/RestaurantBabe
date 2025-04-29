# utils/database.py – now with user preferences table support
import logging
import json
import uuid
import time
from sqlalchemy import create_engine, MetaData, Table, Column, String, JSON, Float, select
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# -------------------------------------------------
# GLOBALS
# -------------------------------------------------
meta = MetaData()
engine = None  # will be initialised by initialize_db()
tables = {}

# -------------------------------------------------
# INITIALISATION
# -------------------------------------------------

def initialize_db(config):
    """Initialise database connection and core tables (idempotent)."""
    global engine, tables

    if engine is not None:
        return  # already initialised

    try:
        # 1) Engine
        engine = create_engine(config.DATABASE_URL)

        # 2) Core tables (one‑to‑one with config names)
        core_defs = {
            config.DB_TABLE_SOURCES: [('_id', String, True), ('data', JSON), ('timestamp', Float)],
            config.DB_TABLE_SEARCHES: [('_id', String, True), ('data', JSON), ('timestamp', Float)],
            config.DB_TABLE_PROCESSES: [('_id', String, True), ('data', JSON), ('timestamp', Float)],
            config.DB_TABLE_RESTAURANTS: [('_id', String, True), ('data', JSON), ('timestamp', Float)],
            # NEW: per‑user standing preferences
            config.DB_TABLE_USER_PREFS: [('_id', String, True), ('data', JSON), ('timestamp', Float)],
        }

        for name, cols in core_defs.items():
            if name in tables:
                continue
            tables[name] = Table(
                name,
                meta,
                *[Column(col_name, col_type, primary_key=is_pk) for col_name, col_type, is_pk in
                  [(c[0], c[1], c[2] if len(c) == 3 else False) for c in cols]]
            )

        # 3) Create all missing tables
        meta.create_all(engine)
        logger.info("Database tables initialised (including user_prefs).")

    except Exception as e:
        logger.error(f"Error initialising database: {e}")
        raise

# -------------------------------------------------
# UTILITY FUNCTIONS (unchanged below except for minor comments)
# -------------------------------------------------

def ensure_city_table(city, config, table_type="restaurants"):
    """Ensure a city‑specific table exists."""
    global engine, tables, meta

    if engine is None:
        initialize_db(config)

    table_name = f"{table_type}_{city.lower().replace(' ', '_').replace('-', '_')}"

    if table_name in tables:
        return table_name

    try:
        tables[table_name] = Table(
            table_name,
            meta,
            Column('_id', String, primary_key=True),
            Column('data', JSON),
            Column('timestamp', Float)
        )
        tables[table_name].create(engine, checkfirst=True)
        logger.info(f"Created city‑specific {table_type} table: {table_name}")
        return table_name
    except Exception as e:
        logger.error(f"Error creating {table_type} table {table_name}: {e}")
        return config.DB_TABLE_RESTAURANTS if table_type == "restaurants" else config.DB_TABLE_SOURCES

# --- save_data / update_data / find_data / find_all_data ---
# Implementation below is unchanged; it automatically supports the new table
# because they work generically over the 'tables' dict.

# (Full original implementations are kept below for completeness.)


def save_data(table_name, data_dict, config):
    """Save data to the specified table."""
    global engine, tables
    if engine is None:
        initialize_db(config)

    # city‑specific convenience
    if table_name.startswith("restaurants_"):
        city = table_name.replace("restaurants_", "").replace("_", " ")
        table_name = ensure_city_table(city, config)
    elif table_name.startswith("sources_"):
        city = table_name.replace("sources_", "").replace("_", " ")
        table_name = ensure_city_table(city, config, table_type="sources")

    try:
        doc_id = data_dict.get('id', str(uuid.uuid4()))
        record = {
            '_id': doc_id,
            'data': data_dict,
            'timestamp': data_dict.get('timestamp', time.time())
        }
        with engine.begin() as conn:
            conn.execute(tables[table_name].insert().values(**record))
        return doc_id
    except SQLAlchemyError as e:
        logger.error(f"Error saving to database: {e}")
        return None


def update_data(table_name, filter_query, update_data, config):
    # (original logic unchanged)
    global engine, tables
    if engine is None:
        initialize_db(config)

    # convenience wrappers for city tables
    if table_name.startswith("restaurants_"):
        city = table_name.replace("restaurants_", "").replace("_", " ")
        table_name = ensure_city_table(city, config)
    elif table_name.startswith("sources_"):
        city = table_name.replace("sources_", "").replace("_", " ")
        table_name = ensure_city_table(city, config, table_type="sources")

    try:
        table = tables[table_name]
        with engine.begin() as conn:
            # same selection logic as before ... (kept intact)
            if 'domain' in filter_query:
                stmt = select(table).where(table.c.data['domain'].as_string() == filter_query['domain'])
            elif 'location' in filter_query:
                stmt = select(table).where(table.c.data['location'].as_string() == filter_query['location'])
            elif 'city' in filter_query:
                stmt = select(table).where(table.c.data['city'].as_string() == filter_query['city'])
            elif 'name' in filter_query and 'address' in filter_query:
                stmt = select(table).where(
                    (table.c.data['name'].as_string() == filter_query['name']) &
                    (table.c.data['address'].as_string() == filter_query['address'])
                )
            else:
                logger.error(f"Unsupported filter query: {filter_query}")
                return False

            row = conn.execute(stmt).fetchone()
            if row:
                record_id = row[0]
                update_stmt = table.update().where(table.c._id == record_id).values(
                    data=update_data,
                    timestamp=update_data.get('evaluated_at') or update_data.get('timestamp') or time.time()
                )
                conn.execute(update_stmt)
                return True
            logger.warning("No record found to update.")
            return False
    except SQLAlchemyError as e:
        logger.error(f"Error updating database: {e}")
        return False


def find_data(table_name, query, config):
    # (original logic unchanged)
    global engine, tables
    if engine is None:
        initialize_db(config)

    if table_name.startswith("restaurants_"):
        city = table_name.replace("restaurants_", "").replace("_", " ")
        table_name = ensure_city_table(city, config)
    elif table_name.startswith("sources_"):
        city = table_name.replace("sources_", "").replace("_", " ")
        table_name = ensure_city_table(city, config, table_type="sources")

    try:
        table = tables[table_name]
        with engine.begin() as conn:
            if 'location' in query:
                stmt = select(table.c.data).where(table.c.data['location'].as_string() == query['location'])
            elif 'domain' in query:
                stmt = select(table.c.data).where(table.c.data['domain'].as_string() == query['domain'])
            elif 'city' in query:
                stmt = select(table.c.data).where(table.c.data['city'].as_string() == query['city'])
            elif 'name' in query and 'address' in query:
                stmt = select(table.c.data).where(
                    (table.c.data['name'].as_string() == query['name']) &
                    (table.c.data['address'].as_string() == query['address'])
                )
            else:
                logger.error(f"Unsupported query: {query}")
                return None
            row = conn.execute(stmt).fetchone()
            return row[0] if row else None
    except SQLAlchemyError as e:
        logger.error(f"Error querying database: {e}")
        return None


def find_all_data(table_name, query, config, limit=100):
    # (original logic unchanged)
    global engine, tables
    if engine is None:
        initialize_db(config)

    if table_name.startswith("restaurants_"):
        city = table_name.replace("restaurants_", "").replace("_", " ")
        table_name = ensure_city_table(city, config)
    elif table_name.startswith("sources_"):
        city = table_name.replace("sources_", "").replace("_", " ")
        table_name = ensure_city_table(city, config, table_type="sources")

    try:
        table = tables[table_name]
        with engine.begin() as conn:
            if 'city' in query:
                stmt = select(table.c.data).where(table.c.data['city'].as_string() == query['city']).limit(limit)
            elif 'location' in query:
                stmt = select(table.c.data).where(table.c.data['location'].as_string() == query['location']).limit(limit)
            elif 'domain' in query:
                stmt = select(table.c.data).where(table.c.data['domain'].as_string() == query['domain']).limit(limit)
            else:
                logger.error(f"Unsupported query for find_all: {query}")
                return []
            rows = conn.execute(stmt).fetchall()
            return [r[0] for r in rows]
    except SQLAlchemyError as e:
        logger.error(f"Error in find_all query: {e}")
        return []
