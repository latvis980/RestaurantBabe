# utils/handoff_protocol.py
"""
Structured handoff protocol between AI Chat Layer (Supervisor) and Search Pipeline (Workers)

This replaces raw text passing with structured messages for:
- Clear communication
- Context management
- Destination change detection
- Type safety
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum


class SearchType(Enum):
    """Search types - decided by supervisor, executed by workers"""
    CITY_SEARCH = "city_search"
    LOCATION_SEARCH = "location_search"


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

    # Metadata
    user_id: int = 0
    thread_id: str = ""

    def __post_init__(self):
        """Initialize mutable defaults"""
        if self.requirements is None:
            self.requirements = []
        if self.preferences is None:
            self.preferences = {}

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

    # NEW: GPS requirement flag (EXPLICIT)
    needs_gps: bool = False  # True if GPS coordinates required for location button


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'command': self.command.value,
            'search_context': self.search_context.to_dict() if self.search_context else None,
            'conversation_response': self.conversation_response,
            'timestamp': self.timestamp,
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
    reasoning: str = ""
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
        thread_id=thread_id
    )

    return HandoffMessage(
        command=HandoffCommand.EXECUTE_SEARCH,
        search_context=search_context,
        reasoning=reasoning
        # timestamp will use default value 0.0 - no need to pass it
    )


def create_conversation_handoff(response: str, reasoning: str = "") -> HandoffMessage:
    """Create a conversation handoff (no search needed)"""
    return HandoffMessage(
        command=HandoffCommand.CONTINUE_CONVERSATION,
        conversation_response=response,
        reasoning=reasoning
        # timestamp will use default value 0.0 - no need to pass it
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
        # timestamp will use default value 0.0 - no need to pass it
    )

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Create search handoff
    handoff = create_search_handoff(
        destination="Lisbon",
        cuisine="Portuguese",
        search_type=SearchType.LOCATION_SEARCH,
        user_query="find good restaurants in Sao Bento",
        user_id=176556234,
        thread_id="chat_176556234_1234567890",
        requirements=["quality", "local"],
        clear_previous=True,
        is_new_destination=True,
        reasoning="User changed from Bermeo to Lisbon - starting fresh search"
    )

    print("Search Handoff:", handoff.to_dict())

    # Example 2: Create conversation handoff
    conv_handoff = create_conversation_handoff(
        response="Great! I'll find Portuguese restaurants in Sao Bento for you.",
        reasoning="Have all info needed, confirming before search"
    )

    print("\nConversation Handoff:", conv_handoff.to_dict())