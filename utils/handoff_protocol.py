# utils/handoff_protocol.py
"""
Structured handoff protocol between AI Chat Layer (Supervisor) and Search Pipeline (Workers)

This replaces raw text passing with structured messages for:
- Clear communication
- Context management
- Destination change detection
- Type safety

UPDATED: Added supervisor_instructions, exclude_restaurants, modified_query, is_follow_up
for smarter follow-up handling with AI-driven filtering.

UPDATED: Added search_radius_km for dynamic distance-based searches.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum


class SearchType(Enum):
    """Search types - decided by supervisor, executed by workers"""
    CITY_SEARCH = "city_search"
    LOCATION_SEARCH = "location_search"  # Database first, then maps if insufficient
    LOCATION_MAPS_SEARCH = "location_maps_search"  # Maps only (for "more results" after DB)


class HandoffCommand(Enum):
    """Commands from supervisor to workers"""
    EXECUTE_SEARCH = "execute_search"
    CONTINUE_CONVERSATION = "continue_conversation"
    RESUME_WITH_DECISION = "resume_with_decision"


@dataclass
class SearchContext:
    """
    Structured context for search execution

    This is what gets passed from AI Chat Layer to the search pipeline
    """
    # Primary search parameters
    destination: str  # City, neighborhood, or location name
    cuisine: Optional[str] = None  # Type of cuisine

    # Search control
    search_type: SearchType = SearchType.CITY_SEARCH
    gps_coordinates: Optional[Tuple[float, float]] = None

    # Context management
    clear_previous_context: bool = False  # Signal to start fresh
    is_new_destination: bool = False  # Detected destination change

    # User query info (for reference, not accumulation)
    user_query: str = ""  # Current message only
    requirements: List[str] = None  # ["quality", "modern", "local"]
    preferences: Dict[str, Any] = None  # {"price": "moderate", "atmosphere": "casual"}

    # ==========================================================================
    # AI-generated follow-up context (for "more results" and modifications)
    # ==========================================================================
    supervisor_instructions: Optional[str] = None  
    # Natural language instructions from AI Chat Layer to downstream agents.
    # Example: "User saw 3 brunch spots but now wants LUNCH options, closer to location."

    exclude_restaurants: List[str] = None  
    # Restaurant names to exclude (already shown to user)

    modified_query: Optional[str] = None  
    # AI may modify the search query based on conversation context.
    # Example: Original was "brunch" but user said "lunch not brunch" -> modified to "lunch"

    is_follow_up: bool = False
    # Flag indicating this is a follow-up search (user asked for "more")

    # ==========================================================================
    # Dynamic search radius (AI-determined based on user input)
    # ==========================================================================
    search_radius_km: Optional[float] = None
    # Search radius in kilometers. Determined by AI based on user input:
    # - Default: 1.5 km when user says "nearby", "around", etc.
    # - Calculated: When user mentions "10 min walk", "within 1 km", etc.
    # - Reduced: When user asks for "closer" results after initial search
    # If None, downstream will use its default (1.5 km)

    # Metadata
    user_id: int = 0
    thread_id: str = ""

    def __post_init__(self):
        """Initialize mutable defaults"""
        if self.requirements is None:
            self.requirements = []
        if self.preferences is None:
            self.preferences = {}
        if self.exclude_restaurants is None:
            self.exclude_restaurants = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing between components"""
        data = asdict(self)
        # Convert enum to string for JSON compatibility
        data['search_type'] = self.search_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchContext':
        """Create from dictionary"""
        # Convert string back to enum
        if 'search_type' in data and isinstance(data['search_type'], str):
            data['search_type'] = SearchType(data['search_type'])
        return cls(**data)


@dataclass
class HandoffMessage:
    """Structured handoff between AI Chat Layer and orchestrator"""
    command: HandoffCommand
    reasoning: str = ""

    # Conversation
    conversation_response: Optional[str] = None

    # Search
    search_context: Optional[SearchContext] = None

    # Resume
    decision: Optional[str] = None
    thread_id: Optional[str] = None

    # GPS requirement flag (EXPLICIT)
    needs_gps: bool = False  # True if GPS coordinates required for location button

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'command': self.command.value,
            'search_context': self.search_context.to_dict() if self.search_context else None,
            'conversation_response': self.conversation_response,
            'needs_gps': self.needs_gps,
            'reasoning': self.reasoning
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HandoffMessage':
        """Create from dictionary"""
        if 'command' in data and isinstance(data['command'], str):
            data['command'] = HandoffCommand(data['command'])
        if 'search_context' in data and data['search_context']:
            data['search_context'] = SearchContext.from_dict(data['search_context'])
        return cls(**data)


# =============================================================================
# HELPER FUNCTIONS FOR CREATING HANDOFFS
# =============================================================================

def create_search_handoff(
    destination: str,
    cuisine: Optional[str],
    search_type: SearchType,
    user_query: str,
    user_id: int,
    thread_id: str,
    gps_coordinates: Optional[Tuple[float, float]] = None,
    requirements: List[str] = None,
    preferences: Dict[str, Any] = None,
    clear_previous: bool = False,
    is_new_destination: bool = False,
    reasoning: str = "",
    # Follow-up context parameters
    supervisor_instructions: Optional[str] = None,
    exclude_restaurants: List[str] = None,
    modified_query: Optional[str] = None,
    is_follow_up: bool = False,
    # Dynamic radius parameter
    search_radius_km: Optional[float] = None
) -> HandoffMessage:
    """Create a search handoff message from supervisor to search pipeline"""
    search_context = SearchContext(
        destination=destination,
        cuisine=cuisine,
        search_type=search_type,
        gps_coordinates=gps_coordinates,
        clear_previous_context=clear_previous,
        is_new_destination=is_new_destination,
        user_query=user_query,
        requirements=requirements or [],
        preferences=preferences or {},
        user_id=user_id,
        thread_id=thread_id,
        # Follow-up context
        supervisor_instructions=supervisor_instructions,
        exclude_restaurants=exclude_restaurants or [],
        modified_query=modified_query,
        is_follow_up=is_follow_up,
        # Dynamic radius
        search_radius_km=search_radius_km
    )

    return HandoffMessage(
        command=HandoffCommand.EXECUTE_SEARCH,
        search_context=search_context,
        reasoning=reasoning
    )


def create_conversation_handoff(response: str, reasoning: str = "", needs_gps: bool = False) -> HandoffMessage:
    """Create a conversation handoff (no search needed)"""
    return HandoffMessage(
        command=HandoffCommand.CONTINUE_CONVERSATION,
        conversation_response=response,
        reasoning=reasoning,
        needs_gps=needs_gps
    )


def create_resume_handoff(
    thread_id: str,
    decision: str = "accept",
    reasoning: str = "Resuming graph execution with user decision"
) -> HandoffMessage:
    """Create a handoff to resume paused graph execution"""
    return HandoffMessage(
        command=HandoffCommand.RESUME_WITH_DECISION,
        reasoning=reasoning,
        decision=decision,
        thread_id=thread_id
    )