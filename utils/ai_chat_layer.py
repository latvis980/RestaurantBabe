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

from utils.ai_memory_system import (
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
            model=config.OPENAI_MODEL,
            temperature=0.3,  # Slightly creative but focused
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
- Friendly, knowledgeable, and enthusiastic about food
- You remember user preferences and past conversations
- You provide natural, conversational responses
- You're proactive in helping users discover great dining experiences

CRITICAL: You must decide what action to take based on the user's message and context.

Available Actions:
1. CHAT_RESPONSE: Respond conversationally (greetings, thanks, general chat)
2. TRIGGER_SEARCH: Start a new restaurant search (user wants recommendations)
3. FOLLOW_UP_SEARCH: Continue/modify existing search (ask for more, different cuisine, etc.)
4. CLARIFY_REQUEST: Ask for clarification (vague request)
5. SHOW_RECOMMENDATIONS: Show previous recommendations for a city
6. UPDATE_PREFERENCES: Learn user preferences without taking other action

User Context:
{user_context}

Guidelines:
- If user greets you or chats casually → CHAT_RESPONSE
- If user asks for restaurant recommendations → TRIGGER_SEARCH or FOLLOW_UP_SEARCH
- If request is vague or missing key info → CLARIFY_REQUEST  
- If user asks about previous recommendations → SHOW_RECOMMENDATIONS
- If user mentions preferences (cuisine, dietary needs) → UPDATE_PREFERENCES + other action
- Always be memory-aware: reference past conversations naturally

Respond with JSON only:
{{
    "action": "ACTION_TYPE",
    "response_text": "Your natural response to the user",
    "search_params": {{"city": "...", "cuisine": "...", "query": "..."}},
    "confidence": 0.85,
    "reasoning": "Why you chose this action",
    "memory_updates": ["preference learned", "city noted"]
}}"""),
            ("human", "User message: {user_message}")
        ])

        # Search refinement prompt
        self.search_refiner_prompt = ChatPromptTemplate.from_messages([
            ("system", """You help refine restaurant search parameters based on user context and preferences.

User Context: {user_context}
Current Search State: {search_context}

Extract and structure search parameters from the user's message:
- City/location (required)
- Cuisine type (optional)
- Price range (optional) 
- Ambiance/vibe (optional)
- Special requirements (optional)

Consider the user's:
- Past preferences
- Current city (if set)
- Previous searches
- Dietary restrictions

Respond with JSON only:
{{
    "city": "extracted or inferred city",
    "cuisine": "specific cuisine or null",
    "query": "natural language query for search",
    "filters": {{"budget": "...", "ambiance": "...", "dietary": [...]}},
    "confidence": 0.9,
    "needs_clarification": false,
    "clarification_question": null
}}"""),
            ("human", "User message: {user_message}")
        ])

        # Conversation response prompt
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Restaurant Babe, responding conversationally to the user.

Your personality:
- Warm, enthusiastic, and knowledgeable about food
- Remember past conversations and preferences
- Keep responses natural and engaging
- Match the user's communication style

User Context:
{user_context}

Recent conversation:
{recent_messages}

Respond naturally as Restaurant Babe. Reference user's preferences, past recommendations, or context when relevant. Keep the tone conversational and helpful."""),
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
                reasoning=f"Error fallback: {str(e)}",
                memory_updates=[]
            )

    async def _make_chat_decision(
        self, 
        user_message: str, 
        user_context: Dict[str, Any],
        message_history: List[Dict[str, str]]
    ) -> ChatDecision:
        """Use AI to decide what action to take"""
        try:
            # Format context for prompt
            context_str = json.dumps(user_context, indent=2)

            # Get AI decision
            response = await self.llm.ainvoke(
                self.chat_router_prompt.format_messages(
                    user_context=context_str,
                    user_message=user_message
                )
            )

            # Parse JSON response
            decision_data = json.loads(response.content)

            # Create ChatDecision object
            decision = ChatDecision(
                action=ActionType(decision_data["action"]),
                response_text=decision_data.get("response_text"),
                search_params=decision_data.get("search_params"),
                confidence=decision_data.get("confidence", 0.5),
                reasoning=decision_data.get("reasoning", ""),
                memory_updates=decision_data.get("memory_updates", [])
            )

            # If it's a search action, refine the search parameters
            if decision.action in [ActionType.TRIGGER_SEARCH, ActionType.FOLLOW_UP_SEARCH]:
                refined_params = await self._refine_search_parameters(
                    user_message, user_context
                )
                decision.search_params = refined_params

            return decision

        except Exception as e:
            logger.error(f"Error making chat decision: {e}")
            # Fallback to clarification
            return ChatDecision(
                action=ActionType.CLARIFY_REQUEST,
                response_text="I'd love to help you find great restaurants! Could you tell me what city you're looking for and what type of cuisine interests you?",
                search_params=None,
                confidence=0.3,
                reasoning=f"Error fallback: {str(e)}",
                memory_updates=[]
            )

    async def _refine_search_parameters(
        self, 
        user_message: str, 
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Refine search parameters using AI and user context"""
        try:
            # Get current session context for search refinement
            search_context = user_context.get("session_context", {})

            response = await self.llm.ainvoke(
                self.search_refiner_prompt.format_messages(
                    user_context=json.dumps(user_context, indent=2),
                    search_context=json.dumps(search_context, indent=2),
                    user_message=user_message
                )
            )

            return json.loads(response.content)

        except Exception as e:
            logger.error(f"Error refining search parameters: {e}")
            # Fallback search parameters
            return {
                "city": user_context.get("current_city", ""),
                "cuisine": None,
                "query": user_message,
                "filters": {},
                "confidence": 0.3,
                "needs_clarification": True,
                "clarification_question": "Could you specify which city you're interested in?"
            }

    async def _update_session_state(
        self, 
        user_id: int, 
        thread_id: str, 
        decision: ChatDecision,
        user_context: Dict[str, Any]
    ):
        """Update session state based on AI decision"""
        try:
            # Determine new conversation state
            if decision.action == ActionType.TRIGGER_SEARCH:
                new_state = ConversationState.SEARCHING
                context = {
                    "search_started_at": datetime.now().isoformat(),
                    "search_params": decision.search_params
                }
            elif decision.action == ActionType.FOLLOW_UP_SEARCH:
                new_state = ConversationState.SEARCHING
                # Preserve existing context and update
                context = user_context.get("session_context", {})
                context.update({
                    "follow_up_search": True,
                    "search_params": decision.search_params
                })
            elif decision.action in [ActionType.CHAT_RESPONSE, ActionType.CLARIFY_REQUEST]:
                new_state = ConversationState.CASUAL_CHAT
                context = user_context.get("session_context", {})
            else:
                # Keep current state
                current_state, context = await self.memory_system.get_session_state(user_id, thread_id)
                new_state = current_state

            # Update city if mentioned in search params
            if decision.search_params and decision.search_params.get("city"):
                await self.memory_system.set_current_city(
                    user_id, thread_id, decision.search_params["city"]
                )

            # Update session state
            await self.memory_system.set_session_state(user_id, thread_id, new_state, context)

        except Exception as e:
            logger.error(f"Error updating session state: {e}")

    # =====================================================================
    # RESPONSE GENERATION
    # =====================================================================

    async def generate_conversational_response(
        self, 
        user_id: int,
        thread_id: str,
        user_message: str,
        context_message: str = None
    ) -> str:
        """Generate a natural conversational response"""
        try:
            user_context = await self.memory_system.get_user_context(user_id, thread_id)

            # Format recent conversation history if available
            recent_messages = []  # This would come from conversation history

            response = await self.llm.ainvoke(
                self.conversation_prompt.format_messages(
                    user_context=json.dumps(user_context, indent=2),
                    recent_messages=json.dumps(recent_messages, indent=2),
                    user_message=user_message
                )
            )

            return response.content

        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            return "I'm here to help you find amazing restaurants! What are you in the mood for?"

    # =====================================================================
    # RESTAURANT RESULT PROCESSING
    # =====================================================================

    async def process_search_results(
        self, 
        user_id: int,
        thread_id: str,
        search_results: Dict[str, Any]
    ) -> str:
        """
        Process restaurant search results and generate intelligent response
        This replaces the automated search messages with contextual AI responses
        """
        try:
            user_context = await self.memory_system.get_user_context(user_id, thread_id)

            # Store restaurants in memory
            if search_results.get("success") and search_results.get("final_restaurants"):
                restaurants = search_results["final_restaurants"]
                current_city = await self.memory_system.get_current_city(user_id, thread_id)

                for restaurant in restaurants[:5]:  # Store top 5
                    restaurant_memory = RestaurantMemory(
                        restaurant_name=restaurant.get("name", "Unknown"),
                        city=current_city or "Unknown",
                        cuisine=restaurant.get("cuisine", "Unknown"),
                        recommended_date=datetime.now().isoformat(),
                        user_feedback=None,
                        rating_given=None,
                        notes=None,
                        source=search_results.get("source", "search")
                    )
                    await self.memory_system.add_restaurant_memory(user_id, restaurant_memory)

            # Update session state to presenting results
            await self.memory_system.set_session_state(
                user_id, thread_id, ConversationState.PRESENTING_RESULTS,
                {"search_results": search_results, "results_shown_at": datetime.now().isoformat()}
            )

            # Generate contextual intro message
            if search_results.get("success"):
                restaurant_count = len(search_results.get("final_restaurants", []))
                city = await self.memory_system.get_current_city(user_id, thread_id)

                intro_messages = [
                    f"Perfect! I found {restaurant_count} amazing spots in {city} for you! 🍽️",
                    f"Great news! I've discovered {restaurant_count} fantastic restaurants in {city}! ✨",
                    f"Excellent! Here are {restaurant_count} top-rated places I think you'll love in {city}! 🌟"
                ]

                # Choose based on user's conversation patterns
                patterns = await self.memory_system.get_conversation_patterns(user_id)
                if patterns.preferred_response_length == "short":
                    return f"Found {restaurant_count} great spots in {city}! 🍽️"
                else:
                    return intro_messages[0]  # Default to first option
            else:
                return "I had some trouble finding restaurants for that search. Could you try a different city or cuisine? 😔"

        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            return "I found some restaurants for you! 🍽️"

    async def generate_follow_up_suggestions(
        self, 
        user_id: int,
        thread_id: str
    ) -> Optional[str]:
        """Generate intelligent follow-up suggestions after showing results"""
        try:
            user_context = await self.memory_system.get_user_context(user_id, thread_id)
            preferences = await self.memory_system.get_user_preferences(user_id)
            current_city = await self.memory_system.get_current_city(user_id, thread_id)

            suggestions = []

            # Suggest based on preferences
            if len(preferences.preferred_cuisines) > 1:
                other_cuisines = [c for c in preferences.preferred_cuisines if c != "recent_cuisine"]
                if other_cuisines:
                    suggestions.append(f"Want to try {other_cuisines[0]} places instead?")

            # Suggest different budget range
            if preferences.budget_range == "mid-range":
                suggestions.append("Looking for budget-friendly options?")

            # Suggest nearby cities if they have preferences for multiple cities
            if len(preferences.preferred_cities) > 1:
                other_cities = [c for c in preferences.preferred_cities if c != current_city]
                if other_cities:
                    suggestions.append(f"Want recommendations for {other_cities[0]}?")

            if suggestions:
                return f"\n\n💡 {suggestions[0]}"

            return None

        except Exception as e:
            logger.error(f"Error generating follow-up suggestions: {e}")
            return None

    # =====================================================================
    # HELPER FUNCTIONS
    # =====================================================================

    async def should_trigger_search(self, user_message: str, user_context: Dict[str, Any]) -> bool:
        """Quick check if message likely needs restaurant search"""
        search_keywords = [
            "restaurant", "food", "eat", "dining", "cuisine", "meal",
            "lunch", "dinner", "breakfast", "coffee", "bar", "cafe"
        ]

        return any(keyword in user_message.lower() for keyword in search_keywords)

    async def extract_city_from_message(self, user_message: str) -> Optional[str]:
        """Extract city name from user message using simple patterns"""
        # This is a simplified version - in production you'd use NER or LLM
        city_indicators = ["in ", "at ", "near ", "around "]

        for indicator in city_indicators:
            if indicator in user_message.lower():
                parts = user_message.lower().split(indicator)
                if len(parts) > 1:
                    potential_city = parts[1].split()[0].title()
                    return potential_city

        return None


# =====================================================================
# FACTORY FUNCTION
# =====================================================================

def create_ai_chat_layer(config) -> AIChatLayer:
    """Factory function to create AI chat layer"""
    return AIChatLayer(config)