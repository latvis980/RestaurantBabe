# agents/langgraph_restaurant_agent.py
"""
FIXED LangGraph Restaurant Recommendation Agent

Main orchestrator using LangGraph's ReAct agent pattern with state management
for restaurant search and recommendation workflows.

Fixes:
1. Tool parameter passing (JSON serialization)
2. Query analysis fallbacks
3. Web search query generation
4. Error handling and graceful degradation
"""

import logging
import json
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from agents.langgraph_tools import create_restaurant_tools

logger = logging.getLogger(__name__)


class RestaurantAgentState(TypedDict):
    """State schema for the restaurant recommendation agent"""
    messages: List[Any]
    query: str
    destination: Optional[str]
    query_analysis: Optional[Dict[str, Any]]
    database_results: Optional[Dict[str, Any]]
    evaluation_results: Optional[Dict[str, Any]]
    web_results: Optional[Dict[str, Any]]
    final_recommendations: Optional[Dict[str, Any]]
    user_id: Optional[int]
    conversation_context: Optional[Dict[str, Any]]


class LangGraphRestaurantAgent:
    """
    FIXED LangGraph-based restaurant recommendation agent.

    Uses ReAct pattern with tools for multi-step restaurant search workflow:
    1. Analyze query (with fallbacks)
    2. Search database
    3. Evaluate results and route (DB-only or hybrid with web search)
    4. Format recommendations (with error handling)
    """

    def __init__(self, config):
        """Initialize the LangGraph agent with tools and checkpointer"""
        self.config = config

        logger.info("🚀 Initializing FIXED LangGraph Restaurant Agent")

        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,
            api_key=config.OPENAI_API_KEY
        )

        self.tools = create_restaurant_tools(config)

        self.checkpointer = MemorySaver()

        self.system_prompt = self._build_system_prompt()

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer
        )

        logger.info("✅ FIXED LangGraph Restaurant Agent initialized with tools and state management")

    def _build_system_prompt(self) -> str:
        """Build the enhanced system prompt for the agent"""
        return """You are an expert restaurant recommendation assistant called Restaurant Babe.

Your job is to help users find great restaurants, cafes, bars, and dining experiences based on their requests.

WORKFLOW (MANDATORY - follow this exact sequence):
1. ALWAYS start by using 'analyze_restaurant_query' to understand what the user wants
2. Then use 'search_restaurant_database' to find restaurants in our database
3. Use 'evaluate_and_route_content' to determine if we have enough results or need web search
4. If web search is needed, use 'search_web_for_restaurants' to find more options
5. Finally, use 'format_restaurant_recommendations' to create the final output

IMPORTANT GUIDELINES:
- ALWAYS analyze the query first to extract destination and preferences
- Database search is fast and preferred when we have good results
- Web search provides additional options when database results are insufficient
- Be conversational and helpful in your responses
- Focus on providing high-quality, relevant recommendations

TOOL USAGE RULES:
- Call tools in the correct sequence (analyze → search → evaluate → [web search if needed] → format)
- When passing data between tools, use JSON format for complex data structures
- Handle tool errors gracefully and inform the user
- NEVER skip the analyze step - it's critical for proper routing

PARAMETER PASSING:
- For analyze_restaurant_query: pass the raw user query as a string
- For search_restaurant_database: pass the full analysis result as JSON string
- For evaluate_and_route_content: pass combined analysis and database results as JSON string
- For search_web_for_restaurants: pass analysis data including search_queries as JSON string
- For format_restaurant_recommendations: pass all collected data as JSON string

ERROR HANDLING:
- If any tool returns an error, acknowledge it and try to proceed with available data
- If no restaurants are found at all, explain the situation and suggest alternatives
- Always provide a helpful response even if the search doesn't work perfectly

Remember: Your goal is to provide excellent restaurant recommendations efficiently!"""

    def process_query(
        self,
        query: str,
        user_id: Optional[int] = None,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a restaurant search query through the FIXED LangGraph agent.

        Args:
            query: User's restaurant search request
            user_id: Optional user identifier for conversation tracking
            conversation_context: Optional context from previous interactions

        Returns:
            Dictionary with agent response and recommendations
        """
        try:
            logger.info(f"🎯 Processing query: '{query}' for user {user_id}")

            config = {"configurable": {"thread_id": str(user_id or "default")}}

            # Add context if available
            context_msg = ""
            if conversation_context:
                prev_destination = conversation_context.get('destination')
                if prev_destination:
                    context_msg = f"\n\nContext: User's previous search was in {prev_destination}."

            # Create messages for the agent
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"{query}{context_msg}")
            ]

            logger.info(f"📨 Sending messages to agent: HumanMessage='{query}...'")
            logger.info(f"🔄 Invoking LangGraph agent...")

            # Invoke the agent
            result = self.agent.invoke(
                {"messages": messages},
                config=config
            )

            logger.info(f"📥 Agent returned {len(result.get('messages', []))} messages")

            # Extract the final response
            messages = result.get("messages", [])

            # Log message types for debugging
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    logger.info(f"  Message {i}: {msg_type} with {len(msg.tool_calls)} tool call(s)")
                    for tool_call in msg.tool_calls:
                        logger.info(f"    🔧 Tool: {tool_call.get('name', 'unknown')}")
                elif hasattr(msg, 'content'):
                    content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    logger.info(f"  Message {i}: {msg_type} content='{content_preview}'")
                else:
                    logger.info(f"  Message {i}: {msg_type}")

            # Get the final AI message with the response
            final_response = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, 'tool_calls', None):
                    final_response = msg.content
                    break

            if not final_response:
                logger.warning("❌ No final response found in messages")
                final_response = "I couldn't find any restaurants matching your criteria. Please try a different search."

            logger.info(f"✅ Query processing complete. Response length: {len(final_response)} chars")

            return {
                "success": True,
                "langchain_formatted_results": final_response,
                "agent_state": {
                    "query": query,
                    "user_id": user_id,
                    "conversation_context": conversation_context
                },
                "messages": messages
            }

        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            error_response = (
                f"I encountered an error while searching for restaurants: {str(e)}. "
                "Please try rephrasing your request or search for a different location."
            )

            return {
                "success": False,
                "langchain_formatted_results": error_response,
                "error": str(e),
                "agent_state": {
                    "query": query,
                    "user_id": user_id,
                    "error": str(e)
                }
            }

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent and its tools"""
        return {
            "agent_type": "LangGraph ReAct Agent",
            "model": self.config.OPENAI_MODEL,
            "tools_count": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "has_checkpointer": bool(self.checkpointer),
            "version": "fixed_v1.0"
        }