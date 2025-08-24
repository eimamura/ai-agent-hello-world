import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def main():
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable.")
        return
    
    # Initialize the model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create tools
    tools = [
        Tool(
            name="get_weather",
            func=get_weather,
            description="Get weather information for a city"
        )
    ]
    
    # Create agent with memory
    memory = MemorySaver()
    agent = create_react_agent(llm, tools, checkpointer=memory)
    
    # Run the agent
    config = {"configurable": {"thread_id": "test-thread"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        config=config
    )
    
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()