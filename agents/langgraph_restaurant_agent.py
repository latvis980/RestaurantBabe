"""
LangGraph Restaurant Recommendation Agent

Main orchestrator using LangGraph's ReAct agent pattern with state management
for restaurant search and recommendation workflows.
"""

import logging
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
    LangGraph-based restaurant recommendation agent.
    
    Uses ReAct pattern with tools for multi-step restaurant search workflow:
    1. Analyze query
    2. Search database
    3. Evaluate results and route (DB-only or hybrid with web search)
    4. Format recommendations
    """

    def __init__(self, config):
        """Initialize the LangGraph agent with tools and checkpointer"""
        self.config = config
        
        logger.info("🚀 Initializing LangGraph Restaurant Agent")
        
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,
            api_key=config.OPENAI_API_KEY
        )
        
        self.tools = create_restaurant_tools(config)
        
        self.checkpointer = MemorySaver()
        
        system_prompt = self._build_system_prompt()
        
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            state_modifier=system_prompt,
            checkpointer=self.checkpointer
        )
        
        logger.info("✅ LangGraph Restaurant Agent initialized with tools and state management")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent"""
        return """You are an expert restaurant recommendation assistant called Restaurant Babe.

Your job is to help users find great restaurants, cafes, bars, and dining experiences based on their requests.

WORKFLOW:
1. First, use 'analyze_restaurant_query' to understand what the user wants
2. Then, use 'search_restaurant_database' to find restaurants in our database
3. Use 'evaluate_and_route_content' to determine if we have enough results or need web search
4. If web search is needed, use 'search_web_for_restaurants' to find more options
5. Finally, use 'format_restaurant_recommendations' to create the final output

IMPORTANT GUIDELINES:
- Always analyze the query first to extract destination and preferences
- Database search is fast and preferred when we have good results
- Web search provides additional options when database results are insufficient
- Be conversational and helpful in your responses
- Focus on providing high-quality, relevant recommendations

TOOL USAGE:
- Call tools in the correct sequence (analyze → search → evaluate → format)
- Pass the output of one tool as input to the next when needed
- If any tool returns an error, handle it gracefully and inform the user

Remember: Your goal is to provide excellent restaurant recommendations efficiently!"""

    def process_query(
        self,
        query: str,
        user_id: Optional[int] = None,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a restaurant search query through the LangGraph agent.
        
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
            
            context_msg = ""
            if conversation_context:
                prev_destination = conversation_context.get('destination')
                if prev_destination:
                    context_msg = f"\n\nContext: User's previous search was in {prev_destination}."
            
            user_message = f"{query}{context_msg}"
            
            messages = [HumanMessage(content=user_message)]
            
            logger.info("🔄 Invoking LangGraph agent...")
            result = self.agent.invoke(
                {"messages": messages},
                config=config
            )
            
            response = self._extract_response(result)
            
            logger.info("✅ Query processing complete")
            return {
                "success": True,
                "response": response,
                "raw_result": result
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error processing your request. Please try again."
            }

    def _extract_response(self, result: Dict[str, Any]) -> str:
        """
        Extract the final response from agent result.
        
        Args:
            result: Raw result from agent invocation
        
        Returns:
            Formatted response string
        """
        try:
            messages = result.get("messages", [])
            
            if not messages:
                return "I couldn't process your request. Please try again."
            
            last_message = messages[-1]
            
            if hasattr(last_message, 'content'):
                return last_message.content
            elif isinstance(last_message, dict):
                return last_message.get('content', 'No response generated')
            else:
                return str(last_message)
                
        except Exception as e:
            logger.error(f"Error extracting response: {e}")
            return "I encountered an error generating a response."

    def get_conversation_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation state for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            Conversation state or None
        """
        try:
            config = {"configurable": {"thread_id": str(user_id)}}
            state = self.checkpointer.get(config)
            return state
        except Exception as e:
            logger.error(f"Error retrieving conversation state: {e}")
            return None

    def clear_conversation_state(self, user_id: int) -> bool:
        """
        Clear conversation state for a user (e.g., when destination changes).
        
        Args:
            user_id: User identifier
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🧹 Clearing conversation state for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation state: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "tools_available": len(self.tools),
            "model": self.config.OPENAI_MODEL,
            "checkpointer_enabled": self.checkpointer is not None
        }


def create_langgraph_agent(config):
    """
    Factory function to create a LangGraph restaurant agent.
    
    Args:
        config: Application configuration
    
    Returns:
        LangGraphRestaurantAgent instance
    """
    return LangGraphRestaurantAgent(config)
