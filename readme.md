# Restaurant Recommendation App

A Python application that provides restaurant recommendations through a Telegram bot interface. The app uses Tavily for searching restaurant information from specific sources, OpenAI for formatting the results in a friendly tone, and LangChain for orchestrating these components. LangSmith is used for tracing and debugging.

## Features

- **Telegram Bot Interface**: Users can send text or voice messages to get restaurant recommendations
- **Voice Recognition**: Voice messages are transcribed using OpenAI's Whisper model
- **Intelligent Search**: Tavily search targets specific restaurant sources (Michelin Guide, Conde Nast, etc.)
- **Friendly Recommendations**: OpenAI formats results in a friendly, engaging tone
- **LangSmith Tracing**: All operations are traced for monitoring and debugging

## Components

1. **Tavily Search Agent** (`tavily_agent.py`): Searches for restaurant information from specified sources
2. **OpenAI Formatting Agent** (`openai_agent.py`): Formats raw search results with a friendly tone
3. **LangChain Orchestrator** (`langchain_orchestrator.py`): Connects the agents and implements tracing
4. **Telegram Bot** (`telegram_bot.py`): Provides the user interface and handles voice messages
5. **Configuration** (`config.py`): API keys and default parameters
6. **Main Application** (`main.py`): Entry point and component initialization

## Installation

### Prerequisites

- Python 3.10 or later
- API keys for:
  - OpenAI
  - Tavily
  - Telegram Bot
  - LangSmith (optional, for tracing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/restaurant-recommendation-app.git
cd restaurant-recommendation-app
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=restaurant-recommender
```

## Usage

### Running in Polling Mode (Development)

```bash
python main.py
```

### Running in Webhook Mode (Production)

```bash
python main.py --webhook --webhook-url https://your-webhook-url.com
```

### Command Line Options

- `--webhook`: Run in webhook mode instead of polling
- `--webhook-url`: Webhook URL (required for webhook mode)
- `--webhook-path`: Webhook path (default