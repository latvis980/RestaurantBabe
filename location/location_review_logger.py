# location/location_review_logger.py
"""
Enhanced logging for Google Maps reviews in Langsmith

This module adds detailed review logging to make reviews visible in Langsmith traces.
"""

import logging
from typing import List, Dict, Any, Optional
from langsmith import traceable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ReviewLogEntry:
    """Structured review log entry for Langsmith"""
    venue_name: str
    review_text: str
    rating: Optional[int] = None
    author: Optional[str] = None
    time: Optional[str] = None

@traceable(
    run_type="chain",
    name="google_reviews_extraction",
    metadata={"component": "review_logger", "purpose": "visibility"}
)
def log_google_reviews_to_langsmith(
    venue_name: str,
    google_reviews: List[Dict],
    max_reviews: int = 5
) -> Dict[str, Any]:
    """
    Extract and log Google reviews in a Langsmith-visible format
    
    Args:
        venue_name: Name of the venue
        google_reviews: List of review dictionaries from Google Maps
        max_reviews: Maximum number of reviews to log
        
    Returns:
        Dictionary with formatted review data for Langsmith
    """
    if not google_reviews:
        logger.info(f"No reviews found for {venue_name}")
        return {
            "venue": venue_name,
            "review_count": 0,
            "reviews": []
        }
    
    # Extract and format reviews
    formatted_reviews = []
    for i, review in enumerate(google_reviews[:max_reviews], 1):
        if isinstance(review, dict):
            formatted_review = {
                "review_number": i,
                "text": review.get("text", "")[:300],  # First 300 chars
                "rating": review.get("rating"),
                "author": review.get("author_name", "Anonymous"),
                "time": review.get("relative_time_description", "Unknown time")
            }
            formatted_reviews.append(formatted_review)
            
            # Log to console for debugging
            logger.info(f"📝 Review {i} for {venue_name}: {formatted_review['text'][:100]}...")
    
    result = {
        "venue": venue_name,
        "review_count": len(google_reviews),
        "reviews_logged": len(formatted_reviews),
        "reviews": formatted_reviews
    }
    
    logger.info(f"✅ Logged {len(formatted_reviews)} reviews for {venue_name} to Langsmith")
    return result


@traceable(
    run_type="chain", 
    name="review_context_builder",
    metadata={"component": "review_logger", "step": "context_building"}
)
@traceable(
    run_type="chain", 
    name="review_context_builder",
    metadata={"component": "review_logger", "step": "context_building"}
)
def build_review_context_with_logging(
    venue_name: str,
    google_reviews: List[Dict],
    max_reviews: int = 5  # CHANGED from 3 to 5
) -> str:
    """
    Build review context string while logging individual reviews to Langsmith

    This function creates the review_context string that goes to the AI editor,
    but also logs each review separately for visibility.

    Args:
        venue_name: Name of the venue
        google_reviews: List of review dictionaries
        max_reviews: Number of reviews to include (now defaults to 5)

    Returns:
        Concatenated review context string
    """
    if not google_reviews:
        return ""

    review_texts = []
    for i, review in enumerate(google_reviews[:max_reviews], 1):
        if isinstance(review, dict) and 'text' in review:
            # INCREASED from 200 to 400 chars per review
            review_text = review['text'][:400]
            review_texts.append(review_text)

            # Log each review individually for Langsmith visibility
            logger.info(f"Review {i}/{max_reviews} for {venue_name}: {review_text[:150]}...")

    context = " | ".join(review_texts)
    logger.info(f"Built review context for {venue_name}: {len(context)} chars from {len(review_texts)} reviews")

    return context


@traceable(
    run_type="chain",
    name="reviews_before_ai_editor",
    metadata={"component": "review_logger", "checkpoint": "before_editor"}
)
def log_reviews_before_ai_processing(
    combined_venues: List[Any]
) -> Dict[str, Any]:
    """
    Log all reviews just before they go to the AI editor
    
    This checkpoint function shows exactly what review data is being
    sent to the AI for description generation.
    
    Args:
        combined_venues: List of CombinedVenueData objects
        
    Returns:
        Summary of all review data going to AI
    """
    summary = {
        "total_venues": len(combined_venues),
        "venues_with_reviews": 0,
        "total_review_chars": 0,
        "venue_reviews": []
    }
    
    for venue in combined_venues:
        has_reviews = bool(getattr(venue, 'review_context', ''))
        if has_reviews:
            summary["venues_with_reviews"] += 1
            review_context = venue.review_context
            summary["total_review_chars"] += len(review_context)
            
            venue_review_data = {
                "venue_name": venue.name,
                "review_context_length": len(review_context),
                "review_preview": review_context[:200] + "..." if len(review_context) > 200 else review_context,
                "has_media": venue.has_media_coverage,
                "media_publications": venue.media_publications
            }
            summary["venue_reviews"].append(venue_review_data)
            
            # Log each venue's reviews
            logger.info(f"📍 {venue.name}: {len(review_context)} chars of reviews")
    
    logger.info(f"🎯 Total: {summary['venues_with_reviews']}/{summary['total_venues']} venues have reviews")
    logger.info(f"📊 Total review context: {summary['total_review_chars']} characters")
    
    return summary
