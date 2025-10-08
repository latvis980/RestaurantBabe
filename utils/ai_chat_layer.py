# utils/ai_chat_layer.py
"""
COMPLETE V2: AI Chat Layer with Structured Handoff Protocol

Includes all original functionality:
- Session management
- Conversation history
- GPS coordinates
- Collected info tracking
- Memory-aware (ready for integration)

New features:
- Structured HandoffMessage output
- Destination change detection
- Context clearing signals
- No raw query accumulation
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from utils.handoff_protocol import (
    HandoffMessage, SearchContext, SearchType, HandoffCommand,
    create_search_handoff, create_conversation_handoff
)

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Internal conversation states"""
    GREETING = "greeting"
    COLLECTING_CUISINE = "collecting_cuisine"
    COLLECTING_LOCATION = "collecting_location"
    COLLECTING_PREFERENCES = "collecting_preferences"
    READY_TO_SEARCH = "ready_to_search"
    SHOWING_RESULTS = "showing_results"
    FOLLOW_UP = "follow_up"


class AIChatLayer:
    """
    COMPLETE V2: Supervisor agent with structured handoffs

    Responsibilities:
    - Manage conversation flow
    - Collect destination + cuisine
    - Decide WHEN to search
    - Detect destination changes
    - Return structured HandoffMessage

    NOT Responsible For:
    - Deciding HOW to search (city vs location)
    - Raw query accumulation
    - Detailed location analysis
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3,
            api_key=config.OPENAI_API_KEY
        )

        # User sessions
        self.user_sessions: Dict[int, Dict[str, Any]] = {}

        # Build prompts
        self._build_prompts()

        logger.info("✅ AI Chat Layer V2 (Complete) initialized")

    def _build_prompts(self):
        """Build AI prompts for conversation management"""

        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a conversation manager for a restaurant recommendation bot.

    Your job is to:
    1. Have natural conversations
    2. Collect necessary information (destination + cuisine)
    3. Decide WHEN enough info is gathered to search
    4. Detect when user changes destination

    REQUIRED FOR SEARCH:
    - Destination: city, neighborhood, or area name
    - Cuisine/Type: what food they want

    CONVERSATION STATES:
    - greeting: Initial interaction
    - collecting: Gathering information
    - ready_to_search: Have enough to search
    - showing_results: Just showed results

    DESTINATION CHANGE DETECTION:
    - If user mentions a NEW location different from current session
    - Signal: clear_previous_context = True

    Current Session:
    - Current destination: {current_destination}
    - Current cuisine: {current_cuisine}
    - Conversation state: {conversation_state}

    User's NEW message: {user_message}

    Respond with JSON. CRITICAL: Choose exactly ONE action:
    {{
        "action": "chat_response",
        "response_text": "natural conversational response",
        "destination": "extracted destination or keep current",
        "cuisine": "extracted cuisine or keep current",
        "requirements": ["quality", "local", "modern"],
        "is_new_destination": true,
        "new_state": "conversation_state",
        "confidence": 0.85,
        "reasoning": "why you chose this action"
    }}

    OR

    {{
        "action": "trigger_search",
        "response_text": "Great! Let me find those restaurants for you.",
        "destination": "city name",
        "cuisine": "food type",
        "requirements": ["quality"],
        "is_new_destination": false,
        "new_state": "ready_to_search",
        "confidence": 0.9,
        "reasoning": "Have both destination and cuisine"
    }}

    VALID ACTIONS (choose ONE):
    - "chat_response" - Continue conversation, need more info
    - "collect_info" - Ask for missing information
    - "trigger_search" - Ready to search (have destination + cuisine)

    DO NOT use pipes or multiple actions. Choose the SINGLE best action.

    Be conversational, warm, and natural. Don't interrogate users."""),
            ("human", "{user_message}")
        ])

    # =========================================================================
    # MAIN PROCESSING METHOD
    # =========================================================================

    async def process_message(
        self,
        user_id: int,
        user_message: str,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
        message_history: Optional[List[Dict[str, str]]] = None  # Fix: Changed from List[Dict[str, str]] to Optional
    ) -> HandoffMessage:
        """
        Process user message and return structured handoff

        Args:
            user_id: User ID
            user_message: Current user message
            gps_coordinates: Optional GPS coordinates
            thread_id: Thread ID
            message_history: Optional external message history

        Returns:
            HandoffMessage with command and context
        """
        try:
            if not thread_id:
                thread_id = f"chat_{user_id}_{int(time.time())}"

            # Get or create session
            session = self._get_or_create_session(user_id, thread_id)

            # Update GPS if provided
            if gps_coordinates:
                session['gps_coordinates'] = gps_coordinates

            # Add user message to history
            session['conversation_history'].append({
                'role': 'user',
                'message': user_message,
                'timestamp': time.time()
            })

            # Keep history manageable
            if len(session['conversation_history']) > 20:
                session['conversation_history'] = session['conversation_history'][-15:]

            # Format context for AI
            conversation_context = self._format_conversation_context(session)

            # Get current state safely with type checking
            current_state = session.get('state', ConversationState.GREETING)
            if isinstance(current_state, ConversationState):
                current_state_value = current_state.value
            else:
                current_state_value = str(current_state) if current_state else 'greeting'

            # Prepare prompt variables
            prompt_vars = {
                'conversation_context': conversation_context,
                'current_destination': session.get('current_destination') or 'None',
                'current_cuisine': session.get('current_cuisine') or 'None',
                'conversation_state': current_state_value,
                'user_message': user_message
            }

            # Get AI decision
            # Get AI decision
            response = await self.llm.ainvoke(
                self.conversation_prompt.format_messages(**prompt_vars)
            )

            # Parse response - handle both string and BaseMessage
            if hasattr(response, 'content'):
                response_content = response.content
                # Ensure it's a string (CRITICAL FIX)
                if not isinstance(response_content, str):
                    response_content = str(response_content)
            else:
                response_content = str(response)

            # Now response_content is guaranteed to be str
            try:
                decision = self._parse_ai_response(response_content)
            except json.JSONDecodeError:
                logger.error(f"JSON parse error: {response_content}")
                return create_conversation_handoff(
                    response="How can I help you find restaurants?",
                    reasoning="JSON parse error - using fallback"
                )

            logger.info(f"🤖 AI Decision: {decision.get('action')} - {decision.get('reasoning')}")

            # Extract info from decision
            destination = decision.get('destination', session.get('current_destination'))
            cuisine = decision.get('cuisine', session.get('current_cuisine'))
            is_new_destination = decision.get('is_new_destination', False)
            requirements = decision.get('requirements', [])
            preferences = decision.get('preferences', {})

            # Update session
            session['current_destination'] = destination
            session['current_cuisine'] = cuisine

            # Update state
            new_state_str = decision.get('new_state', 'greeting')
            try:
                session['state'] = ConversationState(new_state_str)
            except ValueError:
                session['state'] = ConversationState.GREETING

            # Add assistant response to history
            if decision.get('response_text'):
                session['conversation_history'].append({
                    'role': 'assistant',
                    'message': decision['response_text'],
                    'timestamp': time.time()
                })

            # Update collected info
            self._update_collected_info(session, destination, cuisine, requirements, preferences)

            # Route by action
            action = decision.get('action')

            if action in ['chat_response', 'collect_info']:
                # Continue conversation
                return create_conversation_handoff(
                    response=decision.get('response_text', "How can I help?"),
                    reasoning=decision.get('reasoning', '')
                )

            elif action == 'trigger_search':
                # Track search time
                session['last_search_time'] = time.time()

                # Determine search type hint
                search_type_hint = (
                    SearchType.LOCATION_SEARCH if gps_coordinates
                    else SearchType.CITY_SEARCH
                )

                # Create search handoff
                return create_search_handoff(
                    destination=destination or "unknown",  # Fix: Provide default
                    cuisine=cuisine,
                    search_type=search_type_hint,
                    user_query=user_message,
                    user_id=user_id,
                    thread_id=thread_id,
                    gps_coordinates=gps_coordinates,
                    requirements=requirements,
                    preferences=preferences,
                    clear_previous=is_new_destination,
                    is_new_destination=is_new_destination,
                    reasoning=decision.get('reasoning', '')
                )

            else:
                # Unknown action - fallback
                return create_conversation_handoff(
                    response=decision.get('response_text', "Let me help you find restaurants."),
                    reasoning=f"Unknown action: {action}"
                )

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return create_conversation_handoff(
                response="I'd be happy to help you find restaurants. What are you looking for?",
                reasoning=f"Error: {str(e)}"
            )

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def _get_or_create_session(self, user_id: int, thread_id: str) -> Dict[str, Any]:
        """Get or create user session"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'user_id': user_id,
                'thread_id': thread_id,
                'created_at': time.time(),
                'state': ConversationState.GREETING,
                'conversation_history': [],
                'current_destination': None,
                'current_cuisine': None,
                'collected_info': {
                    'location': None,
                    'cuisine': None,
                    'requirements': [],
                    'preferences': {}
                },
                'gps_coordinates': None,
                'last_search_time': None
            }
        return self.user_sessions[user_id]

    def _format_conversation_context(self, session: Dict[str, Any]) -> str:
        """Format conversation history for AI"""
        history = session.get('conversation_history', [])
        if not history:
            return "No previous conversation."

        formatted = []
        for msg in history[-6:]:  # Last 6 messages
            role = "User" if msg['role'] == 'user' else "Assistant"
            formatted.append(f"{role}: {msg['message']}")

        return "\n".join(formatted)

    def _update_collected_info(
        self, 
        session: Dict[str, Any],
        destination: Optional[str],
        cuisine: Optional[str],
        requirements: List[str],
        preferences: Dict[str, Any]
    ):
        """Update collected information in session"""
        collected = session['collected_info']

        if destination:
            collected['location'] = destination
        if cuisine:
            collected['cuisine'] = cuisine
        if requirements:
            collected['requirements'] = list(set(collected['requirements'] + requirements))
        if preferences:
            collected['preferences'].update(preferences)

    def _parse_ai_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI response, handle JSON formatting"""
        try:
            # Extract JSON from markdown if present
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content.strip()

            return json.loads(json_content)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Content: {response_content[:200]}")
            raise

    # =========================================================================
    # UTILITY METHODS (for compatibility with existing code)
    # =========================================================================

    def get_search_ready_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get collected information (for backward compatibility)

        NOTE: With structured handoffs, this is less needed since
        SearchContext contains all info. Kept for gradual migration.
        """
        session = self.user_sessions.get(user_id)
        if not session:
            return None

        return {
            'collected_info': session.get('collected_info', {}),
            'gps_coordinates': session.get('gps_coordinates'),
            'conversation_history': session.get('conversation_history', []),
            'current_destination': session.get('current_destination'),
            'current_cuisine': session.get('current_cuisine')
        }

    def clear_session(self, user_id: int):
        """Clear user session"""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            # Keep thread_id and created_at, clear rest
            session['current_destination'] = None
            session['current_cuisine'] = None
            session['state'] = ConversationState.GREETING
            session['collected_info'] = {
                'location': None,
                'cuisine': None,
                'requirements': [],
                'preferences': {}
            }
            session['last_search_time'] = None
            logger.info(f"🧹 Cleared session for user {user_id}")

    def get_session_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get current session info"""
        return self.user_sessions.get(user_id)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import asyncio

    class MockConfig:
        OPENAI_API_KEY = "sk-test"
        OPENAI_MODEL = "gpt-4o-mini"

    async def test_conversation():
        """Test conversation flow"""
        config = MockConfig()
        chat_layer = AIChatLayer(config)

        user_id = 176556234

        # Test 1: Initial request
        print("\n=== Test 1: Initial Request ===")
        handoff1 = await chat_layer.process_message(
            user_id=user_id,
            user_message="find restaurants in Bermeo, modern Basque cuisine"
        )
        print(f"Command: {handoff1.command.value}")
        if handoff1.search_context:
            print(f"Destination: {handoff1.search_context.destination}")
            print(f"Cuisine: {handoff1.search_context.cuisine}")

        # Test 2: Destination change
        print("\n=== Test 2: Destination Change ===")
        handoff2 = await chat_layer.process_message(
            user_id=user_id,
            user_message="actually, find restaurants in Lisbon instead"
        )
        print(f"Command: {handoff2.command.value}")
        if handoff2.search_context:
            print(f"Destination: {handoff2.search_context.destination}")
            print(f"Clear previous: {handoff2.search_context.clear_previous_context}")
            print(f"New destination: {handoff2.search_context.is_new_destination}")

    # asyncio.run(test_conversation())