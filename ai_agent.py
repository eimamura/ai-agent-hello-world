import os
import requests
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()


def calculator_tool(expression: str) -> str:
    """Simple calculator tool that evaluates mathematical expressions."""
    import ast
    import operator
    
    # Supported operations
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def safe_eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return ops[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)
            return ops[type(node.op)](operand)
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def weather_tool(location: str) -> str:
    """Mock weather tool that returns fake weather data."""
    return f"The weather in {location} is sunny with 75°F temperature."


def search_tool(query: str) -> str:
    """Web search tool using DuckDuckGo API."""
    try:
        # Use DuckDuckGo's instant answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Try to get instant answer
        if data.get('AbstractText'):
            return f"Search result for '{query}': {data['AbstractText']}"
        elif data.get('Answer'):
            return f"Search result for '{query}': {data['Answer']}"
        elif data.get('Definition'):
            return f"Search result for '{query}': {data['Definition']}"
        else:
            # Fallback to related topics if available
            topics = data.get('RelatedTopics', [])
            if topics:
                first_topic = topics[0]
                if isinstance(first_topic, dict) and first_topic.get('Text'):
                    return f"Search result for '{query}': {first_topic['Text']}"
            
            return f"No detailed results found for '{query}'. Try rephrasing your search query."
            
    except requests.exceptions.RequestException as e:
        return f"Search error: Unable to perform search due to network issue - {str(e)}"
    except Exception as e:
        return f"Search error: {str(e)}"


def create_ai_agent():
    """Create an AI agent using LangChain and OpenAI GPT-4."""
    
    # Initialize GPT-4 model
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Define tools available to the agent
    tools = [
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="Useful for mathematical calculations. Input should be a valid mathematical expression."
        ),
        Tool(
            name="Weather",
            func=weather_tool,
            description="Get weather information for a specific location. Input should be a location name."
        ),
        Tool(
            name="Search",
            func=search_tool,
            description="Search the web for information. Input should be a search query or question."
        )
    ]
    
    # Create system prompt
    system_message = """You are a helpful AI assistant with access to various tools.
    Use the tools when needed to provide accurate and helpful responses.
    Always be polite and professional in your interactions."""
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor


def main():
    """Main function to run the AI agent."""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable.")
        return
    
    # Create the agent
    agent = create_ai_agent()
    
    print("AI Agent is ready! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = agent.invoke({"input": user_input})
            print(f"\nAI Agent: {response['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()