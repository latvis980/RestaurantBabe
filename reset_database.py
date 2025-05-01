# reset_database.py
import os
import logging
from sqlalchemy import create_engine, inspect, MetaData

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_reset")

# Get database URL from environment or config
import config
DATABASE_URL = os.getenv("DATABASE_URL", config.DATABASE_URL)

def reset_database():
    """Drop all tables and reinitialize them."""
    try:
        logger.info("Connecting to database...")
        engine = create_engine(DATABASE_URL)

        # Create metadata object
        metadata = MetaData()

        # Bind metadata to engine
        metadata.reflect(bind=engine)

        # Drop all tables
        logger.info("Dropping all tables...")
        metadata.drop_all(engine)
        logger.info("All tables dropped successfully.")

        # Now you can reinitialize the database schema if needed
        logger.info("Reinitializing database schema...")
        from utils.database import initialize_db
        initialize_db(config)
        logger.info("Database reinitialized successfully.")

        return True
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting database reset process...")
    success = reset_database()
    if success:
        logger.info("Database reset completed successfully.")
    else:
        logger.error("Database reset failed.")