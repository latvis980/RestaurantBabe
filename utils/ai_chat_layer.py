# utils/ai_chat_layer.py
"""
Enhanced AI Chat Layer for Restaurant Bot

This layer sits between the Telegram bot and the LangGraph agents, providing:
1. Intelligent conversation routing (chat vs search vs pipeline)
2. Memory-aware responses
3. Natural conversation flow with information collection
4. Context-aware decision making
5. Raw query accumulation across multiple messages

The AI Chat Layer decides when enough information is collected to trigger search.
"""

import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions the AI can take"""
    CHAT_RESPONSE = "chat_response"           # Just respond conversationally  
    COLLECT_INFO = "collect_info"             # Gather more information
    TRIGGER_CITY_SEARCH = "trigger_city_search"         # Launch city-wide search
    TRIGGER_LOCATION_SEARCH = "trigger_location_search" # Launch location-based search
    FOLLOW_UP_SEARCH = "follow_up_search"     # Continue existing search
    CLARIFY_REQUEST = "clarify_request"       # Ask for clarification
    SHOW_RECOMMENDATIONS = "show_recommendations"  # Show saved recommendations
    UPDATE_PREFERENCES = "update_preferences"     # Learn from conversation


class ConversationState(Enum):
    """Current state of the conversation"""
    GREETING = "greeting"
    COLLECTING_CUISINE = "collecting_cuisine"
    COLLECTING_LOCATION = "collecting_location"
    COLLECTING_PREFERENCES = "collecting_preferences"
    READY_TO_SEARCH = "ready_to_search"
    SHOWING_RESULTS = "showing_results"
    FOLLOW_UP = "follow_up"


@dataclass  
class ChatDecision:
    """Decision made by the AI chat layer"""
    action: ActionType
    response_text: Optional[str]
    search_params: Optional[Dict[str, Any]]
    search_type: Optional[str]  # "city_wide" or "location_based"
    confidence: float
    reasoning: str
    memory_updates: List[str]
    new_state: Optional[ConversationState]
    raw_query_accumulated: str


class AIChatLayer:
    """
    Intelligent chat layer that coordinates between user messages and restaurant services

    This layer provides:
    - Natural conversation flow
    - Information collection until ready to search
    - Intelligent routing to search pipelines
    - Context preservation across conversations
    - Raw query accumulation
    """

    def __init__(self, config):
        self.config = config

        # Initialize AI model for chat decisions
        self.llm = ChatOpenAI(
            model=getattr(config, 'AI_CHAT_LAYER_MODEL', 'gpt-4o-mini'),
            temperature=getattr(config, 'AI_CHAT_TEMPERATURE', 0.3),
            max_tokens=getattr(config, 'AI_CHAT_MAX_TOKENS', 1000),
            api_key=config.OPENAI_API_KEY
        )

        # Store conversation sessions per user
        self.user_sessions = {}  # user_id -> session_data

        # Build prompts
        self._build_prompts()

        logger.info("✅ Enhanced AI Chat Layer initialized")

    def _build_prompts(self):
        """Build AI prompts for conversation management"""

        # Main chat routing prompt
        self.chat_router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Restaurant Babe, an intelligent AI that helps users find amazing restaurants worldwide.

Your job is to have natural conversations with users and collect enough information to make great restaurant recommendations. You are warm, enthusiastic, and knowledgeable about food.

CONVERSATION FLOW RULES:
1. COLLECT INFORMATION: You need at least location + cuisine type before triggering a search
2. BE CONVERSATIONAL: Chat naturally, don't interrogate users
3. ACCUMULATE CONTEXT: Build up the full conversation context over multiple messages
4. DECIDE WHEN READY: Only trigger search when you have sufficient information

REQUIRED INFORMATION FOR SEARCH:
- LOCATION: City/area/neighborhood OR GPS coordinates OR location context
- CUISINE/TYPE: What kind of food they want

OPTIONAL INFORMATION (enhances search):
- Preferences: price range, atmosphere, specific requirements
- Occasion: date night, business lunch, family dinner
- Dietary restrictions: vegetarian, gluten-free, etc.

SEARCH TYPES:
- CITY_SEARCH: "best sushi in Tokyo" (city-wide recommendations)
- LOCATION_SEARCH: "pizza near me" or with GPS coordinates (nearby venues)

CONVERSATION STATES:
- greeting: Initial interaction
- collecting_cuisine: Need to know what food they want
- collecting_location: Need to know where they are/want to eat
- collecting_preferences: Have basics, gathering preferences
- ready_to_search: Have enough info to search
- showing_results: Just showed results, can follow up
- follow_up: User asking for more/different options

Current conversation context:
{conversation_context}

User's message: {user_message}

Current conversation state: {current_state}

Raw query accumulated so far: {raw_query_accumulated}

Respond with a JSON object:
{{
    "action": "chat_response|collect_info|trigger_city_search|trigger_location_search|follow_up_search",
    "response_text": "your warm, natural response",
    "search_params": {{"location": "...", "cuisine": "...", "requirements": "..."}},
    "search_type": "city_wide|location_based",
    "confidence": 0.85,
    "reasoning": "why you chose this action",
    "memory_updates": ["learned preference", "noted location"],
    "new_state": "conversation_state",
    "raw_query_accumulated": "complete conversation context for search"
}}

IMPORTANT: 
- Only trigger search when you have BOTH location and cuisine/food type
- Be conversational and warm in your responses
- Accumulate the full conversation in raw_query_accumulated
- For location searches, include GPS info if available"""),
            ("human", "{user_message}")
        ])

    # =====================================================================
    # MAIN PROCESSING FUNCTION
    # =====================================================================

    async def process_message(
        self, 
        user_id: int,
        user_message: str,
        gps_coordinates: Optional[Tuple[float, float]] = None,
        message_history: List[Dict[str, str]] = None
    ) -> ChatDecision:
        """
        Main function: Process user message and decide what action to take

        This is the core intelligence that replaces direct search routing
        with conversational information collection.
        """
        try:
            # Get or create user session
            session = self._get_or_create_session(user_id)

            # Add GPS coordinates if provided
            if gps_coordinates:
                session['gps_coordinates'] = gps_coordinates
                session['raw_query_accumulated'] += f" [GPS: {gps_coordinates}]"

            # Update conversation history
            session['conversation_history'].append({
                'role': 'user',
                'message': user_message,
                'timestamp': time.time()
            })

            # Prepare context for AI
            conversation_context = self._format_conversation_context(session)

            # Get AI decision
            ai_response = await self.llm.ainvoke(
                self.chat_router_prompt.format_messages(
                    conversation_context=conversation_context,
                    user_message=user_message,
                    current_state=session.get('state', 'greeting'),
                    raw_query_accumulated=session.get('raw_query_accumulated', '')
                )
            )

            # Parse AI response
            decision_data = self._parse_ai_response(ai_response.content)
            decision = self._create_chat_decision(decision_data, session, user_message)

            # Update session based on decision
            self._update_session_from_decision(session, decision, user_message)

            logger.info(f"AI Chat Decision for user {user_id}: {decision.action.value} - {decision.reasoning}")
            return decision

        except Exception as e:
            logger.error(f"Error in AI chat processing: {e}")
            return ChatDecision(
                action=ActionType.CHAT_RESPONSE,
                response_text="I'm here to help you find amazing restaurants! What are you looking for?",
                search_params=None,
                search_type=None,
                confidence=0.5,
                reasoning=f"Error fallback: {str(e)}",
                memory_updates=[],
                new_state=ConversationState.GREETING,
                raw_query_accumulated=user_message
            )

    def _get_or_create_session(self, user_id: int) -> Dict[str, Any]:
        """Get existing session or create new one for user"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'user_id': user_id,
                'state': ConversationState.GREETING,
                'conversation_history': [],
                'raw_query_accumulated': '',
                'collected_info': {
                    'location': None,
                    'cuisine': None,
                    'preferences': [],
                    'requirements': []
                },
                'gps_coordinates': None,
                'last_search_time': None,
                'created_at': time.time()
            }
        return self.user_sessions[user_id]

    def _format_conversation_context(self, session: Dict[str, Any]) -> str:
        """Format conversation history for AI prompt"""
        history = session.get('conversation_history', [])
        if not history:
            return "No previous conversation."

        formatted = []
        for msg in history[-6:]:  # Last 6 messages
            role = "User" if msg['role'] == 'user' else "Assistant"
            formatted.append(f"{role}: {msg['message']}")

        return "\n".join(formatted)

    def _parse_ai_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI response, handling potential JSON formatting issues"""
        try:
            # Try to extract JSON from response
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content.strip()

            return json.loads(json_content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Response content: {response_content}")

            # Fallback response
            return {
                "action": "chat_response",
                "response_text": "I'm here to help you find great restaurants! What are you looking for?",
                "search_params": {},
                "search_type": None,
                "confidence": 0.3,
                "reasoning": "JSON parse error - using fallback",
                "memory_updates": [],
                "new_state": "greeting",
                "raw_query_accumulated": ""
            }

    def _create_chat_decision(
        self, 
        decision_data: Dict[str, Any], 
        session: Dict[str, Any], 
        user_message: str
    ) -> ChatDecision:
        """Create ChatDecision object from AI response"""
        try:
            action_str = decision_data.get('action', 'chat_response')
            action = ActionType(action_str)
        except ValueError:
            logger.warning(f"Unknown action: {action_str}, defaulting to chat_response")
            action = ActionType.CHAT_RESPONSE

        try:
            state_str = decision_data.get('new_state', 'greeting')
            new_state = ConversationState(state_str)
        except ValueError:
            logger.warning(f"Unknown state: {state_str}, defaulting to greeting")
            new_state = ConversationState.GREETING

        # Handle raw query accumulation
        raw_query = decision_data.get('raw_query_accumulated', '')
        if not raw_query:
            # Fallback: accumulate from session
            existing = session.get('raw_query_accumulated', '')
            raw_query = f"{existing} {user_message}".strip()

        return ChatDecision(
            action=action,
            response_text=decision_data.get('response_text', ''),
            search_params=decision_data.get('search_params', {}),
            search_type=decision_data.get('search_type'),
            confidence=decision_data.get('confidence', 0.5),
            reasoning=decision_data.get('reasoning', ''),
            memory_updates=decision_data.get('memory_updates', []),
            new_state=new_state,
            raw_query_accumulated=raw_query
        )

    def _update_session_from_decision(
        self, 
        session: Dict[str, Any], 
        decision: ChatDecision, 
        user_message: str
    ) -> None:
        """Update session state based on AI decision"""
        # Update conversation state
        session['state'] = decision.new_state

        # Update raw query accumulation
        session['raw_query_accumulated'] = decision.raw_query_accumulated

        # Add assistant response to history
        if decision.response_text:
            session['conversation_history'].append({
                'role': 'assistant',
                'message': decision.response_text,
                'timestamp': time.time()
            })

        # Extract and store collected information
        if decision.search_params:
            collected_info = session['collected_info']
            if 'location' in decision.search_params:
                collected_info['location'] = decision.search_params['location']
            if 'cuisine' in decision.search_params:
                collected_info['cuisine'] = decision.search_params['cuisine']
            if 'requirements' in decision.search_params:
                collected_info['requirements'].append(decision.search_params['requirements'])

        # Track search timing
        if decision.action in [ActionType.TRIGGER_CITY_SEARCH, ActionType.TRIGGER_LOCATION_SEARCH]:
            session['last_search_time'] = time.time()

        # Keep conversation history manageable
        if len(session['conversation_history']) > 20:
            session['conversation_history'] = session['conversation_history'][-15:]

    def get_search_ready_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get collected information when ready to search"""
        session = self.user_sessions.get(user_id)
        if not session:
            return None

        return {
            'raw_query': session.get('raw_query_accumulated', ''),
            'collected_info': session.get('collected_info', {}),
            'gps_coordinates': session.get('gps_coordinates'),
            'conversation_history': session.get('conversation_history', [])
        }

    def clear_session(self, user_id: int) -> None:
        """Clear user session (for testing or reset)"""
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        logger.info(f"Cleared session for user {user_id}")