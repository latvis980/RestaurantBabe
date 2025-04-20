# utils/database.py
from sqlalchemy import create_engine, Column, String, Float, JSON, Text, MetaData, Table
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

# Define tables dynamically based on collections
def get_table(collection_name):
    """Get or create a table for the collection"""
    if collection_name in metadata.tables:
        return metadata.tables[collection_name]

    # Create a new table
    table = Table(
        collection_name,
        metadata,
        Column("_id", String, primary_key=True),
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

def save_to_mongodb(collection_name, data, config):
    """Save data to PostgreSQL (MongoDB compatibility function)"""
    try:
        # Create a session
        session = Session()

        # Get the table
        table = get_table(collection_name)

        # Ensure _id exists
        if "_id" not in data:
            data["_id"] = str(uuid.uuid4())

        # Check if document exists
        exists = session.query(table).filter(table.c._id == data["_id"]).first()

        # Create new values
        values = {
            "_id": data["_id"],
            "data": data,
            "timestamp": time.time()
        }

        # Insert or update
        if exists:
            session.query(table).filter(table.c._id == data["_id"]).update(values)
        else:
            session.execute(table.insert().values(**values))

        # Commit changes
        session.commit()
        logger.info(f"Saved document to {collection_name}")
        return data["_id"]
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        session.rollback()
        return None
    finally:
        session.close()

def find_in_mongodb(collection_name, query, config):
    """Find a document in PostgreSQL (MongoDB compatibility function)"""
    try:
        # Create a session
        session = Session()

        # Get the table
        table = get_table(collection_name)

        # If _id is in query, use it directly
        if "_id" in query:
            result = session.query(table).filter(table.c._id == query["_id"]).first()
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
        logger.error(f"Error finding document: {e}")
        return None
    finally:
        session.close()

def find_many_in_mongodb(collection_name, query, config, limit=0):
    """Find multiple documents in PostgreSQL (MongoDB compatibility function)"""
    try:
        # Create a session
        session = Session()

        # Get the table
        table = get_table(collection_name)

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
        logger.error(f"Error finding documents: {e}")
        return []
    finally:
        session.close()

def get_mongodb_client(config):
    """Legacy compatibility function - returns None but logs success"""
    logger.info("PostgreSQL connection is being used instead of MongoDB")
    return None