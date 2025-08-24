# AI Agent Hello World

A simple AI agent built with LangChain and OpenAI GPT-4.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

Run the agent:
```bash
python ai_agent.py
```

The agent has access to:
- Calculator tool for mathematical operations
- Weather tool (mock) for weather information

## Example Interactions

- "What is 25 * 4?"
- "What's the weather in New York?"
- "Calculate the square root of 144"