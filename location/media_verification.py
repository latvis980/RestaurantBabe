# location/media_verification.py
"""
Media Verification Agent - STEPS 4 & 5

This implements Steps 4-5 of the location search flow:
- Step 4: Verify results in the media
- Step 5: Get brief restaurant descriptions from trusted media sources

Main purpose: Provide restaurant recommendations from professional guides and media,
not TripAdvisor and UGC sites.

How to verify in the media: run additional Brave searches constructed like:
restaurant + city + media. Filter by source reputation using search_agent.py logic.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from location.google_maps_search import VenueResult
from location.location_utils import LocationUtils

logger = logging.getLogger(__name__)

@dataclass
class VerifiedVenue:
    """Structure for media-verified venue results"""
    name: str
    address: str
    latitude: float
    longitude: float
    distance_km: float
    rating: Optional[float] = None
    place_id: Optional[str] = None
    description: str = ""
    media_sources: List[str] = None
    google_maps_url: str = ""

    def __post_init__(self):
        if self.media_sources is None:
            self.media_sources = []

class MediaVerificationAgent:
    """
    STEPS 4-5: Media verification and description extraction

    Purpose: Filter venues by media coverage and extract descriptions
    from trusted professional sources only.
    """

    def __init__(self, config):
        self.config = config

        # Initialize Brave search for media verification
        try:
            from agents.search_agent import BraveSearchAgent
            self.brave_search = BraveSearchAgent(config)
            logger.info("✅ Brave Search Agent loaded for media verification")
        except ImportError as e:
            logger.error(f"❌ Could not import BraveSearchAgent: {e}")
            self.brave_search = None

        # Media verification settings
        self.min_media_mentions = getattr(config, 'MIN_MEDIA_MENTIONS_REQUIRED', 1)
        self.trusted_media_domains = self._get_trusted_media_domains()

        logger.info("✅ Media Verification Agent initialized (Steps 4-5)")
        logger.info(f"🔧 Will verify against {len(self.trusted_media_domains)} trusted media domains")

    def _get_trusted_media_domains(self) -> List[str]:
        """
        Get list of trusted media domains for restaurant coverage

        Based on professional guides and media, not UGC sites.
        """
        return [
            # Professional restaurant guides
            'timeout.com',
            'eater.com', 
            'zagat.com',
            'michelin.com',
            'resy.com',
            'opentable.com',

            # Major food media
            'foodandwine.com',
            'bonappetit.com',
            'saveur.com',
            'seriouseats.com',
            'thrillist.com',

            # Quality travel/lifestyle media
            'cntraveler.com',
            'travelandleisure.com',
            'vogue.com',
            'gq.com',
            'wallpaper.com',

            # Regional quality publications
            'theinfatuation.com',
            'timecity.com',  # Time Out cities
            'secretnyc.co',
            'secretlondon.co',

            # International quality sources
            'telegraph.co.uk',
            'theguardian.com',
            'independent.co.uk',
            'lemonde.fr',
            'repubblica.it'
        ]

    async def verify_venues(
        self,
        venues: List[VenueResult],
        query: str,
        cancel_check_fn=None
    ) -> List[Dict[str, Any]]:
        """
        STEPS 4-5: Verify venues in media and extract descriptions

        Args:
            venues: List of VenueResult objects from Google Maps
            query: Original search query for context
            cancel_check_fn: Function to check if search should be cancelled

        Returns:
            List of verified venues with media sources and descriptions
        """
        try:
            logger.info(f"📰 STEPS 4-5: Media verification for {len(venues)} venues")

            if not self.brave_search:
                logger.warning("⚠️ Brave search not available, returning unverified venues")
                return self._convert_venues_to_dict(venues)

            verified_venues = []

            for venue in venues:
                if cancel_check_fn and cancel_check_fn():
                    logger.info("🚫 Media verification cancelled")
                    break

                # STEP 4: Verify in media
                media_verification = await self._verify_venue_in_media(venue, query)

                if media_verification['is_verified']:
                    # STEP 5: Get description from trusted sources
                    description = await self._extract_venue_description(
                        venue, media_verification['sources']
                    )

                    verified_venue = self._create_verified_venue(
                        venue, media_verification, description
                    )
                    verified_venues.append(verified_venue)

                    logger.info(f"✅ Verified: {venue.name} ({len(media_verification['sources'])} sources)")
                else:
                    logger.info(f"❌ Not verified: {venue.name} (no trusted media coverage)")

            logger.info(f"📊 STEPS 4-5 COMPLETE: {len(verified_venues)}/{len(venues)} venues verified")
            return verified_venues

        except Exception as e:
            logger.error(f"❌ Error in media verification: {e}")
            # Fallback: return original venues as dictionaries
            return self._convert_venues_to_dict(venues)

    async def _verify_venue_in_media(
        self, 
        venue: VenueResult, 
        query: str
    ) -> Dict[str, Any]:
        """
        STEP 4: Verify venue in trusted media sources

        Constructs searches like: restaurant + city + media
        """
        try:
            # Extract city from venue address
            city = self._extract_city_from_address(venue.address)

            # Construct media search query
            search_query = f"{venue.name} {city} restaurant review"

            logger.info(f"🔍 Media search: '{search_query}'")

            # Perform Brave search
            search_results = await self._perform_media_search(search_query)

            # Filter by trusted domains and evaluate source quality
            trusted_sources = self._filter_trusted_sources(search_results)

            verification_result = {
                'is_verified': len(trusted_sources) >= self.min_media_mentions,
                'sources': trusted_sources,
                'search_query': search_query,
                'total_results': len(search_results)
            }

            return verification_result

        except Exception as e:
            logger.error(f"❌ Error verifying {venue.name} in media: {e}")
            return {'is_verified': False, 'sources': [], 'search_query': '', 'total_results': 0}

    async def _perform_media_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform Brave search for media coverage
        """
        try:
            if not self.brave_search:
                return []

            # Use the correct method from BraveSearchAgent
            # search_parallel_batch expects a list of queries and destination
            search_queries = [query]
            destination = "media_search"  # Generic destination for media searches

            filtered_results = self.brave_search.search_parallel_batch(
                search_queries=search_queries,
                destination=destination,
                query_metadata={}  # Empty metadata for media searches
            )

            return filtered_results

        except Exception as e:
            logger.error(f"❌ Error in media search: {e}")
            return []

    def _filter_trusted_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter search results to only trusted media domains

        Adapts the filtering process from agents/search_agent.py
        """
        try:
            trusted_sources = []

            for result in search_results:
                url = result.get('url', '')
                domain = self._extract_domain(url)

                # Check if domain is in our trusted list
                if any(trusted_domain in domain for trusted_domain in self.trusted_media_domains):
                    # Additional quality check using search_agent logic
                    quality_score = result.get('quality_score', 0.0)

                    if quality_score >= 0.7:  # High quality threshold
                        trusted_sources.append({
                            'url': url,
                            'domain': domain,
                            'title': result.get('title', ''),
                            'snippet': result.get('snippet', ''),
                            'quality_score': quality_score
                        })

            return trusted_sources

        except Exception as e:
            logger.error(f"❌ Error filtering trusted sources: {e}")
            return []

    async def _extract_venue_description(
        self, 
        venue: VenueResult, 
        trusted_sources: List[Dict[str, Any]]
    ) -> str:
        """
        STEP 5: Extract brief restaurant description from trusted media sources

        Gets descriptions from professional guides and media only.
        """
        try:
            if not trusted_sources:
                return f"Restaurant in {self._extract_city_from_address(venue.address)}"

            # Take the best description from highest quality source
            best_source = max(trusted_sources, key=lambda x: x.get('quality_score', 0))
            snippet = best_source.get('snippet', '')

            if snippet and len(snippet) > 20:
                # Clean and format the description
                description = self._clean_description(snippet)
                return description

            # Fallback description
            city = self._extract_city_from_address(venue.address)
            return f"Restaurant recommended by {best_source.get('domain', 'professional guides')} in {city}"

        except Exception as e:
            logger.error(f"❌ Error extracting description for {venue.name}: {e}")
            return f"Restaurant in {self._extract_city_from_address(venue.address)}"

    def _clean_description(self, snippet: str) -> str:
        """
        Clean and format description from media snippet
        """
        try:
            # Remove common snippet artifacts
            description = snippet.strip()

            # Remove date references and other clutter
            import re
            description = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', '', description)
            description = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b', '', description)

            # Limit length and end at sentence boundary
            if len(description) > 200:
                description = description[:200]
                last_period = description.rfind('.')
                if last_period > 100:
                    description = description[:last_period + 1]

            return description.strip()

        except Exception:
            return snippet[:200] if snippet else ""

    def _create_verified_venue(
        self, 
        venue: VenueResult, 
        media_verification: Dict[str, Any], 
        description: str
    ) -> Dict[str, Any]:
        """
        Create verified venue dictionary with all required fields
        """
        sources = media_verification.get('sources', [])
        source_names = [source.get('domain', 'Unknown') for source in sources]

        return {
            'name': venue.name,
            'address': venue.address,
            'latitude': venue.latitude,
            'longitude': venue.longitude,
            'distance_km': venue.distance_km,
            'distance_text': LocationUtils.format_distance(venue.distance_km),
            'rating': venue.rating,
            'place_id': venue.place_id,
            'description': description,
            'sources': source_names,
            'media_verified': True,
            'media_sources_count': len(sources),
            'google_maps_url': venue.google_maps_url,
            'verification_query': media_verification.get('search_query', '')
        }

    def _convert_venues_to_dict(self, venues: List[VenueResult]) -> List[Dict[str, Any]]:
        """
        Convert VenueResult objects to dictionaries (fallback when verification fails)
        """
        return [
            {
                'name': venue.name,
                'address': venue.address,
                'latitude': venue.latitude,
                'longitude': venue.longitude,
                'distance_km': venue.distance_km,
                'distance_text': LocationUtils.format_distance(venue.distance_km),
                'rating': venue.rating,
                'place_id': venue.place_id,
                'description': f"Restaurant in {self._extract_city_from_address(venue.address)}",
                'sources': [],
                'media_verified': False,
                'google_maps_url': venue.google_maps_url
            }
            for venue in venues
        ]

    def _extract_city_from_address(self, address: str) -> str:
        """
        Extract city name from venue address
        """
        try:
            if not address:
                return "Unknown"

            # Simple extraction - split by comma and take appropriate part
            parts = [part.strip() for part in address.split(',')]

            if len(parts) >= 3:
                # Usually: street, city, state/country
                return parts[1]
            elif len(parts) == 2:
                return parts[0]
            else:
                return parts[0] if parts else "Unknown"

        except Exception:
            return "Unknown"

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain

        except Exception:
            return url.lower()

    def get_verification_stats(self) -> Dict[str, Any]:
        """
        Get statistics about media verification process
        """
        return {
            'trusted_domains_count': len(self.trusted_media_domains),
            'min_media_mentions': self.min_media_mentions,
            'trusted_domains': self.trusted_media_domains[:10],  # Show first 10
            'has_brave_search': self.brave_search is not None
        }