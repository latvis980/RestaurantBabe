# utils/ai_chat_layer.py
"""
AI Chat Layer for Restaurant Bot

This layer sits between the Telegram bot and the LangGraph agents, providing:
1. Intelligent conversation routing (chat vs search vs pipeline)
2. Memory-aware responses
3. Natural conversation flow
4. Context-aware decision making

The AI Chat Layer eliminates automated messages and provides contextual,
intelligent responses while coordinating with the restaurant search pipeline.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from ai_memory_system import (
    AIMemorySystem, ConversationState, UserPreferences, 
    RestaurantMemory, ConversationPattern
)

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions the AI can take"""
    CHAT_RESPONSE = "chat_response"           # Just respond conversationally
    TRIGGER_SEARCH = "trigger_search"         # Launch restaurant search
    FOLLOW_UP_SEARCH = "follow_up_search"     # Continue existing search
    CLARIFY_REQUEST = "clarify_request"       # Ask for clarification
    SHOW_RECOMMENDATIONS = "show_recommendations"  # Show saved recommendations
    UPDATE_PREFERENCES = "update_preferences"     # Learn from conversation


@dataclass  
class ChatDecision:
    """Decision made by the AI chat layer"""
    action: ActionType
    response_text: Optional[str]
    search_params: Optional[Dict[str, Any]]
    confidence: float
    reasoning: str
    memory_updates: List[str]


class AIChatLayer:
    """
    Intelligent chat layer that coordinates between user messages and restaurant services

    This layer provides:
    - Natural conversation flow
    - Memory-aware responses  
    - Intelligent routing to search pipelines
    - Context preservation across conversations
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

        # Initialize memory system
        self.memory_system = AIMemorySystem(config)

        # Build prompts
        self._build_prompts()

        logger.info("✅ AI Chat Layer initialized")

    def _build_prompts(self):
        """Build AI prompts for different chat scenarios"""

        # Main chat routing prompt
        self.chat_router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Restaurant Babe, an intelligent AI that helps users find amazing restaurants worldwide.

Your personality:
- Warm, enthusiastic, and knowledgeable about food
- Remember past conversations and preferences
- Keep responses natural and engaging
- Match the user's communication style

User Context:
{user_context}

Recent conversation:
{recent_messages}

Based on the user's message and context, decide what action to take:

1. CHAT_RESPONSE: Just have a conversational response (greetings, questions about food, etc.)
2. TRIGGER_SEARCH: User wants to find restaurants (clear search intent)
3. FOLLOW_UP_SEARCH: User wants more options from current search
4. CLARIFY_REQUEST: Need more information to help them
5. SHOW_RECOMMENDATIONS: User asks about past recommendations
6. UPDATE_PREFERENCES: Learn from what they said about preferences

Respond with a JSON object:
{{
    "action": "action_type",
    "response_text": "natural response text",
    "search_params": {{"city": "...", "cuisine": "...", "requirements": "..."}},
    "confidence": 0.85,
    "reasoning": "why you chose this action",
    "memory_updates": ["preference learned", "city noted"]
}}

IMPORTANT: Always provide a natural, conversational response_text that fits your warm, enthusiastic personality."""),
            ("human", "{user_message}")
        ])

    # =====================================================================
    # MAIN PROCESSING FUNCTION
    # =====================================================================

    async def process_message(
        self, 
        user_id: int,
        thread_id: str,
        user_message: str,
        message_history: List[Dict[str, str]] = None
    ) -> ChatDecision:
        """
        Main function: Process user message and decide what action to take

        This is the core intelligence of the chat layer that replaces
        automated search messages with contextual AI responses.
        """
        try:
            logger.info(f"🧠 AI Chat Layer processing message for user {user_id}")

            # Get comprehensive user context from memory
            user_context = await self.memory_system.get_user_context(user_id, thread_id)

            # Learn from this message (update preferences if applicable)
            current_city = user_context.get("current_city")
            await self.memory_system.learn_preferences_from_message(
                user_id, user_message, current_city
            )

            # Make decision using AI
            decision = await self._make_chat_decision(
                user_message, user_context, message_history or []
            )

            # Update session state based on decision
            await self._update_session_state(user_id, thread_id, decision, user_context)

            logger.info(f"🎯 AI Decision: {decision.action.value} (confidence: {decision.confidence})")
            return decision

        except Exception as e:
            logger.error(f"❌ Error in AI chat layer: {e}")
            # Fallback response
            return ChatDecision(
                action=ActionType.CHAT_RESPONSE,
                response_text="I'm having a bit of trouble right now. Could you try asking again?",
                search_params=None,
                confidence=0.1,
                reasoning="Error fallback",
                memory_updates=[]
            )

    async def _make_chat_decision(
        self,
        user_message: str,
        user_context: Dict[str, Any],
        message_history: List[Dict[str, str]]
    ) -> ChatDecision:
        """Use AI to decide what action to take based on user message and context"""
        try:
            # Format recent messages for context
            recent_messages = self._format_message_history(message_history)

            # Create prompt with context
            prompt = self.chat_router_prompt.format(
                user_context=json.dumps(user_context, indent=2),
                recent_messages=recent_messages,
                user_message=user_message
            )

            # Get AI decision
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            # Parse JSON response
            try:
                decision_data = json.loads(response.content)
            except json.JSONDecodeError:
                # If AI doesn't return valid JSON, create a fallback decision
                logger.warning("AI returned invalid JSON, creating fallback decision")
                return self._create_fallback_decision(user_message, user_context)

            # Create ChatDecision object
            decision = ChatDecision(
                action=ActionType(decision_data.get("action", "chat_response")),
                response_text=decision_data.get("response_text", "I'm here to help you find great restaurants!"),
                search_params=decision_data.get("search_params"),
                confidence=float(decision_data.get("confidence", 0.5)),
                reasoning=decision_data.get("reasoning", "AI decision"),
                memory_updates=decision_data.get("memory_updates", [])
            )

            return decision

        except Exception as e:
            logger.error(f"Error making AI decision: {e}")
            return self._create_fallback_decision(user_message, user_context)

    def _create_fallback_decision(self, user_message: str, user_context: Dict[str, Any]) -> ChatDecision:
        """Create a simple fallback decision when AI fails"""
        # Simple keyword-based fallback
        message_lower = user_message.lower()

        # Look for restaurant search keywords
        search_keywords = ["restaurant", "food", "eat", "hungry", "dinner", "lunch", "breakfast"]
        if any(keyword in message_lower for keyword in search_keywords):
            return ChatDecision(
                action=ActionType.TRIGGER_SEARCH,
                response_text="I'd love to help you find a great restaurant! Let me search for options based on what you're looking for.",
                search_params={"query": user_message},
                confidence=0.6,
                reasoning="Keyword-based fallback search",
                memory_updates=[]
            )
        else:
            return ChatDecision(
                action=ActionType.CHAT_RESPONSE,
                response_text="Hi! I'm Restaurant Babe, and I love helping people find amazing restaurants. What kind of food are you in the mood for?",
                search_params=None,
                confidence=0.7,
                reasoning="General chat fallback",
                memory_updates=[]
            )

    def _format_message_history(self, message_history: List[Dict[str, str]]) -> str:
        """Format message history for AI context"""
        if not message_history:
            return "No recent conversation history."

        # Take last 5 messages for context
        recent = message_history[-5:]
        formatted = []

        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    async def _update_session_state(
        self,
        user_id: int,
        thread_id: str,
        decision: ChatDecision,
        user_context: Dict[str, Any]
    ) -> None:
        """Update session state based on AI decision"""
        try:
            # Map decision to conversation state
            state_mapping = {
                ActionType.CHAT_RESPONSE: ConversationState.CASUAL_CHAT,
                ActionType.TRIGGER_SEARCH: ConversationState.SEARCHING,
                ActionType.FOLLOW_UP_SEARCH: ConversationState.SEARCHING,
                ActionType.CLARIFY_REQUEST: ConversationState.SEARCHING,
                ActionType.SHOW_RECOMMENDATIONS: ConversationState.PRESENTING_RESULTS,
                ActionType.UPDATE_PREFERENCES: ConversationState.CASUAL_CHAT
            }

            new_state = state_mapping.get(decision.action, ConversationState.IDLE)

            # Get current context and update
            current_context = user_context.get("session_context", {})

            # Add decision info to context
            current_context["last_decision"] = {
                "action": decision.action.value,
                "confidence": decision.confidence,
                "timestamp": datetime.now().isoformat()
            }

            # If it's a search decision, store search params
            if decision.search_params:
                current_context["pending_search"] = decision.search_params

            # Update session state
            await self.memory_system.set_session_state(
                user_id, thread_id, new_state, current_context
            )

        except Exception as e:
            logger.error(f"Error updating session state: {e}")

    # =====================================================================
    # UTILITY FUNCTIONS
    # =====================================================================

    async def get_user_memory_summary(self, user_id: int) -> Dict[str, Any]:
        """Get a summary of user's memory for debugging/display"""
        try:
            stats = await self.memory_system.get_memory_stats(user_id)
            preferences = await self.memory_system.get_user_preferences(user_id)

            return {
                "memory_stats": stats,
                "preferences": preferences.to_dict(),
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {"status": "error", "error": str(e)}

    async def handle_city_change(self, user_id: int, thread_id: str, new_city: str) -> bool:
        """Handle when user changes to a different city"""
        try:
            return await self.memory_system.set_current_city(user_id, thread_id, new_city)
        except Exception as e:
            logger.error(f"Error handling city change: {e}")
            return False

    async def save_restaurant_recommendation(
        self,
        user_id: int,
        restaurant_name: str,
        city: str,
        cuisine: str,
        source: str = "search"
    ) -> bool:
        """Save a restaurant recommendation to user's memory"""
        try:
            memory = RestaurantMemory(
                restaurant_name=restaurant_name,
                city=city,
                cuisine=cuisine,
                recommended_date=datetime.now().isoformat(),
                user_feedback=None,
                rating_given=None,
                notes=None,
                source=source
            )

            return await self.memory_system.save_restaurant_memory(user_id, memory)
        except Exception as e:
            logger.error(f"Error saving restaurant recommendation: {e}")
            return False

    async def get_user_restaurant_history(
        self, 
        user_id: int, 
        city: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get user's restaurant recommendation history"""
        try:
            memories = await self.memory_system.get_restaurant_memories(user_id, city)
            return [memory.to_dict() for memory in memories]
        except Exception as e:
            logger.error(f"Error getting restaurant history: {e}")
            return []

    # =====================================================================
    # SPECIALIZED RESPONSE HANDLERS
    # =====================================================================

    async def handle_greeting(self, user_id: int, thread_id: str) -> ChatDecision:
        """Handle greeting messages with personalized response"""
        try:
            preferences = await self.memory_system.get_user_preferences(user_id)

            # Personalize greeting based on known preferences
            if preferences.preferred_cities:
                city_text = f"I remember you like exploring restaurants in {', '.join(preferences.preferred_cities[:2])}!"
            else:
                city_text = "I'd love to learn about your favorite dining spots!"

            response_text = f"Hey there! 🍸 Great to see you again. {city_text} What kind of culinary adventure are you in the mood for today?"

            return ChatDecision(
                action=ActionType.CHAT_RESPONSE,
                response_text=response_text,
                search_params=None,
                confidence=0.9,
                reasoning="Personalized greeting",
                memory_updates=[]
            )
        except Exception as e:
            logger.error(f"Error handling greeting: {e}")
            return ChatDecision(
                action=ActionType.CHAT_RESPONSE,
                response_text="Hello! I'm Restaurant Babe, and I'm excited to help you discover amazing restaurants! What are you in the mood for?",
                search_params=None,
                confidence=0.8,
                reasoning="Fallback greeting",
                memory_updates=[]
            )

    async def handle_more_options_request(
        self, 
        user_id: int, 
        thread_id: str
    ) -> ChatDecision:
        """Handle requests for more restaurant options"""
        try:
            # Get session context to see if there's a current search
            _, session_context = await self.memory_system.get_session_state(user_id, thread_id)

            if "pending_search" in session_context:
                # Continue with existing search parameters
                search_params = session_context["pending_search"]

                return ChatDecision(
                    action=ActionType.FOLLOW_UP_SEARCH,
                    response_text="Let me find some more great options for you! 🔍",
                    search_params=search_params,
                    confidence=0.9,
                    reasoning="Continue existing search",
                    memory_updates=[]
                )
            else:
                # No current search, ask for clarification
                return ChatDecision(
                    action=ActionType.CLARIFY_REQUEST,
                    response_text="I'd love to find more options for you! What type of restaurant or cuisine are you looking for?",
                    search_params=None,
                    confidence=0.8,
                    reasoning="Need search parameters",
                    memory_updates=[]
                )
        except Exception as e:
            logger.error(f"Error handling more options request: {e}")
            return ChatDecision(
                action=ActionType.CLARIFY_REQUEST,
                response_text="I'd be happy to help you find more restaurants! What are you looking for?",
                search_params=None,
                confidence=0.7,
                reasoning="Error fallback",
                memory_updates=[]
            )