# Overview

RestaurantBabe is an AI-powered restaurant recommendation system that provides personalized dining suggestions through a Telegram bot interface. The system combines database search, web scraping, and AI analysis to deliver curated restaurant recommendations based on user queries and location preferences.

The application uses a sophisticated multi-agent architecture to process user requests, search multiple data sources, analyze content quality, and format recommendations in a conversational manner. It supports both text and voice input, handles location-based searches, and provides comprehensive restaurant information including descriptions, ratings, and sources.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Core Framework
- **LangGraph Agent System**: Phase 1 implementation of LangGraph-based ReAct agent with tool orchestration and state management
- **LangChain Orchestration**: Central orchestrator manages the multi-agent workflow with LangSmith tracing for monitoring and debugging
- **Async Processing**: Built on asyncio with Gunicorn async workers for handling concurrent requests efficiently
- **Railway Deployment**: Optimized for Railway platform with proper resource management and Docker containerization
- **Dual Architecture**: Supports both traditional orchestrator and new LangGraph agent (configurable via USE_LANGGRAPH_AGENT flag)

## Multi-Agent Architecture

### LangGraph Agent Framework (Phase 1)
The system now includes a LangGraph-based agent that orchestrates tools using the ReAct pattern:

**Core Tools:**
1. **analyze_restaurant_query**: Analyzes user queries to extract destination and preferences
2. **search_restaurant_database**: Searches local restaurant database with AI filtering
3. **evaluate_and_route_content**: Evaluates results and determines search strategy
4. **search_web_for_restaurants**: Performs web searches using Brave API
5. **format_restaurant_recommendations**: Formats final output for users

**Features:**
- Stateful conversation with memory checkpointing
- Autonomous tool selection and orchestration
- Error handling and graceful degradation
- Configurable via `USE_LANGGRAPH_AGENT` flag in config.py

### Traditional Agent Pipeline
The system also maintains the original pipeline of specialized AI agents:

1. **Query Analyzer**: Parses user requests to extract destination, cuisine preferences, and search intent
2. **Database Search Agent**: Searches local restaurant database using proximity and content matching
3. **Content Evaluation Agent**: Evaluates database results and determines if web search is needed
4. **Brave Search Agent**: Performs web searches using Brave Search API with AI-powered result filtering
5. **Browserless Scraper**: Extracts restaurant content from web pages using Playwright automation
6. **Text Cleaner Agent**: Processes and structures scraped content into clean restaurant data
7. **Editor Agent**: Formats final recommendations with engaging descriptions
8. **Follow-up Search Agent**: Validates restaurant details using Google Maps API

## Location Services
- **GPS Coordinates**: Handles location pins from Telegram users for proximity-based searches
- **Google Maps Integration**: Uses Google Maps Places API for venue verification and additional details
- **Media Verification**: Cross-references restaurants with professional food media coverage
- **Dual Search Modes**: Supports both city-wide searches and location-based proximity searches

## AI Model Strategy
- **Primary Model**: OpenAI GPT-4o-mini for most operations (cost-optimized)
- **Enhanced Analysis**: Claude Sonnet for complex reasoning tasks when needed
- **Fast Processing**: DeepSeek for content sectioning and rapid analysis tasks
- **Voice Recognition**: OpenAI Whisper for voice message transcription

## Data Processing Pipeline
1. **Content Extraction**: Playwright-based scraping with structure-preserving text extraction
2. **Intelligent Cleaning**: AI-powered content cleaning that maintains restaurant context
3. **Deduplication**: Smart restaurant matching to combine multiple sources for the same venue
4. **Quality Scoring**: AI evaluation of content quality and relevance
5. **Source Attribution**: Proper citation of information sources

## Conversation Management
- **State Tracking**: Maintains conversation context across multiple interactions
- **Smart Routing**: AI-powered detection of query types (restaurant requests vs general questions)
- **Destination Changes**: Intelligent detection when users switch to different cities
- **Follow-up Handling**: Manages conversation flow for clarifications and additional requests

# External Dependencies

## AI Services
- **OpenAI**: GPT-4o-mini for query analysis, content evaluation, and text generation
- **Anthropic Claude**: Sonnet model for complex reasoning and enhanced analysis
- **DeepSeek**: Fast content processing and sectioning tasks
- **LangSmith**: Tracing and monitoring of AI operations

## Search and Data Sources
- **Brave Search API**: Primary web search engine for restaurant content discovery
- **Tavily API**: Secondary search for professional food media coverage
- **Google Maps API**: Place verification, geocoding, and additional venue details
- **Google Places API**: Enhanced venue information and ratings

## Database and Storage
- **Supabase**: Primary database for restaurant data storage with PostgreSQL backend
- **Supabase Storage**: File storage for scraped content and processed data
- **Redis**: Session management and caching (implicit through Supabase)

## Communication Platform
- **Telegram Bot API**: Primary user interface for receiving queries and sending recommendations
- **Voice Processing**: OpenAI Whisper for transcribing voice messages

## Infrastructure Services
- **Railway**: Cloud deployment platform with automatic scaling
- **Browserless**: Web scraping infrastructure for content extraction
- **Playwright**: Browser automation for dynamic content scraping

## Utility Services
- **Geopy**: Geocoding and location utilities
- **BeautifulSoup**: HTML parsing and content extraction
- **Schedule**: Automated file cleanup and maintenance tasks