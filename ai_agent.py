import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


def calculator_tool(expression: str) -> str:
    """Simple calculator tool that evaluates mathematical expressions."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def weather_tool(location: str) -> str:
    """Mock weather tool that returns fake weather data."""
    return f"The weather in {location} is sunny with 75°F temperature."


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