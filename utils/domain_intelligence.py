# utils/domain_intelligence.py
import logging
import json
import time
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer, DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Global variables for database connection
engine = None
domain_intelligence_table = None

def initialize_domain_intelligence_db(config):
    """Initialize the domain intelligence database table"""
    global engine, domain_intelligence_table

    if engine is not None:
        return  # Already initialized

    try:
        # Use existing database connection
        engine = create_engine(config.DATABASE_URL)

        # Create metadata
        meta = MetaData()

        # Define the domain intelligence table
        domain_intelligence_table = Table(
            'domain_intelligence',
            meta,
            Column('domain', String(255), primary_key=True),
            Column('complexity', String(50), nullable=False),
            Column('scraper_type', String(50), nullable=False),
            Column('cost', Float, nullable=False, default=0.0),
            Column('confidence', Float, nullable=False, default=0.5),
            Column('reasoning', Text),
            Column('success_count', Integer, default=0),
            Column('failure_count', Integer, default=0),
            Column('total_restaurants_found', Integer, default=0),
            Column('avg_content_length', Integer, default=0),
            Column('last_successful_scrape', DateTime),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('last_updated', DateTime, default=datetime.utcnow),
            Column('metadata', JSON)  # For storing additional flexible data
        )

        # Create table if it doesn't exist
        meta.create_all(engine)

        logger.info("Domain intelligence database table initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing domain intelligence database: {e}")
        raise

def save_domain_intelligence(domain: str, intelligence_data: Dict[str, Any], config) -> bool:
    """
    Save or update domain intelligence in the database

    Args:
        domain: Domain name (e.g., 'timeout.com')
        intelligence_data: Dictionary containing intelligence info
        config: App configuration

    Returns:
        bool: True if successful, False otherwise
    """
    global engine, domain_intelligence_table

    if engine is None:
        initialize_domain_intelligence_db(config)

    try:
        with engine.begin() as conn:
            # Check if domain already exists
            select_stmt = domain_intelligence_table.select().where(
                domain_intelligence_table.c.domain == domain
            )
            existing = conn.execute(select_stmt).fetchone()

            # Prepare data for database
            db_data = {
                'domain': domain,
                'complexity': intelligence_data.get('complexity', 'unknown'),
                'scraper_type': intelligence_data.get('scraper_type', 'enhanced_http'),
                'cost': float(intelligence_data.get('cost', 0.5)),
                'confidence': float(intelligence_data.get('confidence', 0.5)),
                'reasoning': intelligence_data.get('reasoning', ''),
                'success_count': int(intelligence_data.get('success_count', 0)),
                'failure_count': int(intelligence_data.get('failure_count', 0)),
                'total_restaurants_found': int(intelligence_data.get('total_restaurants_found', 0)),
                'avg_content_length': int(intelligence_data.get('avg_content_length', 0)),
                'last_updated': datetime.utcnow(),
                'metadata': intelligence_data.get('metadata', {})
            }

            # Set last_successful_scrape if this was a success
            if intelligence_data.get('was_successful', False):
                db_data['last_successful_scrape'] = datetime.utcnow()

            if existing:
                # Update existing record
                update_stmt = domain_intelligence_table.update().where(
                    domain_intelligence_table.c.domain == domain
                ).values(**db_data)
                conn.execute(update_stmt)
                logger.debug(f"Updated domain intelligence for {domain}")
            else:
                # Insert new record
                db_data['created_at'] = datetime.utcnow()
                insert_stmt = domain_intelligence_table.insert().values(**db_data)
                conn.execute(insert_stmt)
                logger.debug(f"Saved new domain intelligence for {domain}")

        return True

    except Exception as e:
        logger.error(f"Error saving domain intelligence for {domain}: {e}")
        return False

def load_all_domain_intelligence(config) -> List[Dict[str, Any]]:
    """
    Load all domain intelligence from the database

    Args:
        config: App configuration

    Returns:
        List of domain intelligence records
    """
    global engine, domain_intelligence_table

    if engine is None:
        initialize_domain_intelligence_db(config)

    try:
        with engine.begin() as conn:
            # Load domains updated within last 90 days with reasonable confidence
            cutoff_date = datetime.utcnow() - timedelta(days=90)

            select_stmt = domain_intelligence_table.select().where(
                (domain_intelligence_table.c.last_updated > cutoff_date) &
                (domain_intelligence_table.c.confidence > 0.3)
            ).order_by(domain_intelligence_table.c.confidence.desc())

            results = conn.execute(select_stmt).fetchall()

            domain_list = []
            for row in results:
                domain_data = {
                    'domain': row.domain,
                    'complexity': row.complexity,
                    'scraper_type': row.scraper_type,
                    'cost': row.cost,
                    'confidence': row.confidence,
                    'reasoning': row.reasoning,
                    'success_count': row.success_count,
                    'failure_count': row.failure_count,
                    'total_restaurants_found': row.total_restaurants_found,
                    'avg_content_length': row.avg_content_length,
                    'last_updated': row.last_updated.timestamp() if row.last_updated else time.time(),
                    'created_at': row.created_at.timestamp() if row.created_at else time.time(),
                    'metadata': row.metadata or {}
                }
                domain_list.append(domain_data)

            logger.info(f"Loaded {len(domain_list)} domain intelligence records from database")
            return domain_list

    except Exception as e:
        logger.error(f"Error loading domain intelligence: {e}")
        return []

def get_domain_intelligence(domain: str, config) -> Optional[Dict[str, Any]]:
    """
    Get intelligence for a specific domain

    Args:
        domain: Domain name
        config: App configuration

    Returns:
        Domain intelligence data or None if not found
    """
    global engine, domain_intelligence_table

    if engine is None:
        initialize_domain_intelligence_db(config)

    try:
        with engine.begin() as conn:
            select_stmt = domain_intelligence_table.select().where(
                domain_intelligence_table.c.domain == domain
            )
            result = conn.execute(select_stmt).fetchone()

            if result:
                return {
                    'complexity': result.complexity,
                    'scraper_type': result.scraper_type,
                    'cost': result.cost,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'success_count': result.success_count,
                    'failure_count': result.failure_count,
                    'timestamp': result.last_updated.timestamp() if result.last_updated else time.time(),
                    'metadata': result.metadata or {}
                }
            return None

    except Exception as e:
        logger.error(f"Error getting domain intelligence for {domain}: {e}")
        return None

def cleanup_old_domain_intelligence(config, days_old: int = 90, min_confidence: float = 0.3) -> int:
    """
    Clean up old or low-confidence domain intelligence records

    Args:
        config: App configuration
        days_old: Remove records older than this many days
        min_confidence: Remove records with confidence below this threshold

    Returns:
        Number of records deleted
    """
    global engine, domain_intelligence_table

    if engine is None:
        initialize_domain_intelligence_db(config)

    try:
        with engine.begin() as conn:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            # Delete old records with low confidence
            delete_stmt = domain_intelligence_table.delete().where(
                (domain_intelligence_table.c.last_updated < cutoff_date) &
                (domain_intelligence_table.c.confidence < min_confidence)
            )

            result = conn.execute(delete_stmt)
            deleted_count = result.rowcount

            logger.info(f"Cleaned up {deleted_count} old domain intelligence records")
            return deleted_count

    except Exception as e:
        logger.error(f"Error cleaning up domain intelligence: {e}")
        return 0

def get_domain_intelligence_stats(config) -> Dict[str, Any]:
    """
    Get statistics about the domain intelligence database

    Args:
        config: App configuration

    Returns:
        Dictionary with statistics
    """
    global engine, domain_intelligence_table

    if engine is None:
        initialize_domain_intelligence_db(config)

    try:
        with engine.begin() as conn:
            # Total domains - FIXED: Use proper SQLAlchemy COUNT
            total_count_result = conn.execute(
                domain_intelligence_table.select().with_only_columns(func.count())
            ).scalar()
            total_count = total_count_result or 0

            # High confidence domains (>0.8) - FIXED: Use COUNT instead of rowcount
            high_confidence_result = conn.execute(
                domain_intelligence_table.select().with_only_columns(func.count()).where(
                    domain_intelligence_table.c.confidence > 0.8
                )
            ).scalar()
            high_confidence_count = high_confidence_result or 0

            # Domains by complexity - FIXED: Use COUNT for each complexity type
            complexity_stats = {}
            for complexity in ['simple_html', 'moderate_js', 'heavy_js', 'specialized']:
                complexity_result = conn.execute(
                    domain_intelligence_table.select().with_only_columns(func.count()).where(
                        domain_intelligence_table.c.complexity == complexity
                    )
                ).scalar()
                complexity_stats[complexity] = complexity_result or 0

            # Recent activity (last 7 days) - FIXED: Use COUNT
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_updates_result = conn.execute(
                domain_intelligence_table.select().with_only_columns(func.count()).where(
                    domain_intelligence_table.c.last_updated > recent_cutoff
                )
            ).scalar()
            recent_updates = recent_updates_result or 0

            # Top performing domains
            top_domains = conn.execute(
                domain_intelligence_table.select().where(
                    domain_intelligence_table.c.confidence > 0.7
                ).order_by(
                    domain_intelligence_table.c.success_count.desc()
                ).limit(10)
            ).fetchall()

            return {
                'total_domains': total_count,
                'high_confidence_domains': high_confidence_count,
                'complexity_distribution': complexity_stats,
                'recent_updates_7days': recent_updates,
                'top_performing_domains': [
                    {
                        'domain': row.domain,
                        'complexity': row.complexity,
                        'confidence': row.confidence,
                        'success_count': row.success_count,
                        'total_restaurants': row.total_restaurants_found
                    }
                    for row in top_domains
                ],
                'cache_effectiveness': {
                    'high_confidence_rate': round((high_confidence_count / max(total_count, 1)) * 100, 1),
                    'recent_activity_rate': round((recent_updates / max(total_count, 1)) * 100, 1)
                }
            }

    except Exception as e:
        logger.error(f"Error getting domain intelligence stats: {e}")
        return {}

def export_domain_intelligence(config, file_path: str = None) -> str:
    """
    Export domain intelligence to JSON file for backup/analysis

    Args:
        config: App configuration
        file_path: Optional file path, will generate if not provided

    Returns:
        Path to exported file
    """
    if not file_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"domain_intelligence_export_{timestamp}.json"

    try:
        domains = load_all_domain_intelligence(config)

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_domains': len(domains),
            'domains': domains
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported {len(domains)} domain intelligence records to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error exporting domain intelligence: {e}")
        raise

def import_domain_intelligence(config, file_path: str) -> int:
    """
    Import domain intelligence from JSON file

    Args:
        config: App configuration
        file_path: Path to JSON file

    Returns:
        Number of domains imported
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)

        domains = import_data.get('domains', [])
        imported_count = 0

        for domain_data in domains:
            domain = domain_data.get('domain')
            if domain:
                # Convert back to internal format
                intelligence_data = {
                    'complexity': domain_data.get('complexity'),
                    'scraper_type': domain_data.get('scraper_type'),
                    'cost': domain_data.get('cost'),
                    'confidence': domain_data.get('confidence'),
                    'reasoning': domain_data.get('reasoning'),
                    'success_count': domain_data.get('success_count', 0),
                    'failure_count': domain_data.get('failure_count', 0),
                    'total_restaurants_found': domain_data.get('total_restaurants_found', 0),
                    'avg_content_length': domain_data.get('avg_content_length', 0),
                    'metadata': domain_data.get('metadata', {})
                }

                if save_domain_intelligence(domain, intelligence_data, config):
                    imported_count += 1

        logger.info(f"Imported {imported_count} domain intelligence records from {file_path}")
        return imported_count

    except Exception as e:
        logger.error(f"Error importing domain intelligence: {e}")
        raise