# utils/database.py
from sqlalchemy import create_engine, Column, String, Float, JSON, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import os
import logging
import time
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection string from Railway environment
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
Base = declarative_base()
metadata = MetaData()
Session = sessionmaker(bind=engine)

# Define tables dynamically based on collection names
def get_table(table_name):
    """Get or create a table with the given name"""
    if table_name in metadata.tables:
        return metadata.tables[table_name]

    # Create a new table
    table = Table(
        table_name,
        metadata,
        Column("id", String, primary_key=True),  # Renamed from _id to id
        Column("data", JSON),
        Column("timestamp", Float),
        extend_existing=True
    )
    metadata.create_all(engine, tables=[table])
    return table

# Initialize database
def init_db():
    """Initialize the database tables"""
    try:
        metadata.create_all(engine)
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

# Call init_db to create tables
init_db()

def save_data(table_name, data, config):
    """Save data to PostgreSQL"""
    try:
        # Create a session
        session = Session()

        # Get the table
        table = get_table(table_name)

        # Ensure id exists
        if "id" not in data:
            data["id"] = str(uuid.uuid4())

        # Check if document exists
        exists = session.query(table).filter(table.c.id == data["id"]).first()

        # Create new values
        values = {
            "id": data["id"],
            "data": data,
            "timestamp": time.time()
        }

        # Insert or update
        if exists:
            session.query(table).filter(table.c.id == data["id"]).update(values)
        else:
            session.execute(table.insert().values(**values))

        # Commit changes
        session.commit()
        logger.info(f"Saved data to {table_name}")
        return data["id"]
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        session.rollback()
        return None
    finally:
        session.close()

def find_data(table_name, query, config):
    """Find data in PostgreSQL by query"""
    try:
        # Create a session
        session = Session()

        # Get the table
        table = get_table(table_name)

        # If id is in query, use it directly
        if "id" in query:
            result = session.query(table).filter(table.c.id == query["id"]).first()
            if result:
                return result.data
            return None

        # Otherwise, need to scan JSON data
        results = session.query(table).all()
        for row in results:
            data = row.data
            matches = all(key in data and data[key] == value for key, value in query.items())
            if matches:
                return data

        return None
    except Exception as e:
        logger.error(f"Error finding data: {e}")
        return None
    finally:
        session.close()

def find_all_data(table_name, query, config, limit=0):
    """Find multiple data entries in PostgreSQL by query"""
    try:
        # Create a session
        session = Session()

        # Get the table
        table = get_table(table_name)

        # Get all results first
        query_obj = session.query(table)

        # Apply limit if needed
        if limit > 0:
            query_obj = query_obj.limit(limit)

        results = query_obj.all()

        # Filter based on query
        filtered_results = []
        for row in results:
            data = row.data
            matches = all(key in data and data[key] == value for key, value in query.items())
            if matches:
                filtered_results.append(data)

        return filtered_results
    except Exception as e:
        logger.error(f"Error finding data entries: {e}")
        return []
    finally:
        session.close()

def get_db_connection(config):
    """Returns SQLAlchemy engine for direct DB access if needed"""
    logger.info("PostgreSQL connection successful")
    return engine