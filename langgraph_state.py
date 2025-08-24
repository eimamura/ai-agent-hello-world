import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()


class State(TypedDict):
    messages: Annotated[list, "The messages in the conversation"]
    name: Annotated[str, "User's name"]
    greeting_count: Annotated[int, "Number of greetings"]


def greeting_node(state: State):
    """Node that handles initial greeting."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Simple name extraction
    words = last_message.lower().split()
    name = "friend"
    
    if "my name is" in last_message.lower():
        try:
            name_idx = words.index("is") + 1
            if name_idx < len(words):
                name = words[name_idx].capitalize()
        except:
            pass
    elif "i am" in last_message.lower():
        try:
            name_idx = words.index("am") + 1
            if name_idx < len(words):
                name = words[name_idx].capitalize()
        except:
            pass
    
    greeting_msg = AIMessage(content=f"Hello {name}! Nice to meet you. How can I help you today?")
    
    return {
        "messages": messages + [greeting_msg],
        "name": name,
        "greeting_count": state.get("greeting_count", 0) + 1
    }


def chat_node(state: State):
    """Node that handles general chat."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    messages = state["messages"]
    name = state.get("name", "friend")
    
    # Add context about the user
    system_context = f"You are chatting with {name}. Be friendly and helpful."
    context_msg = HumanMessage(content=f"Context: {system_context}\n\nUser message: {messages[-1].content}")
    
    response = llm.invoke([context_msg])
    
    return {
        "messages": messages + [response],
        "name": name,
        "greeting_count": state.get("greeting_count", 0)
    }


def goodbye_node(state: State):
    """Node that handles goodbye."""
    messages = state["messages"]
    name = state.get("name", "friend")
    count = state.get("greeting_count", 0)
    
    goodbye_msg = AIMessage(content=f"Goodbye {name}! It was nice chatting with you. We exchanged {count} greetings!")
    
    return {
        "messages": messages + [goodbye_msg],
        "name": name,
        "greeting_count": count
    }


def router_node(state: State):
    """Router node that just passes through state."""
    return state


def router(state: State) -> str:
    """Route to appropriate node based on message content."""
    if not state["messages"]:
        return "greeting"
    
    last_message = state["messages"][-1].content.lower()
    
    # Check for greetings
    greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(word in last_message for word in greeting_words) and "name" in last_message:
        return "greeting"
    
    # Check for goodbye
    goodbye_words = ["bye", "goodbye", "see you", "farewell", "quit", "exit"]
    if any(word in last_message for word in goodbye_words):
        return "goodbye"
    
    # Default to chat
    return "chat"


def create_state_graph():
    """Create a LangGraph with state management."""
    
    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("goodbye", goodbye_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        router,
        {
            "greeting": "greeting",
            "chat": "chat", 
            "goodbye": "goodbye"
        }
    )
    
    # Add edges back to router after each node
    workflow.add_edge("greeting", "router")
    workflow.add_edge("chat", "router") 
    workflow.add_edge("goodbye", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def main():
    """Main function to run the state-based LangGraph agent."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable.")
        return
    
    # Create the state graph
    agent = create_state_graph()
    
    print("LangGraph State Agent is ready! Say hello and tell me your name!")
    print("Type 'bye' to exit.")
    print("-" * 50)
    
    # Initialize state
    state = {
        "messages": [],
        "name": "",
        "greeting_count": 0
    }
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        try:
            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))
            
            # Run the agent
            result = agent.invoke(state)
            
            # Update state
            state = result
            
            # Get the last AI message
            ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                last_ai_msg = ai_messages[-1]
                print(f"\nAI Agent: {last_ai_msg.content}")
                
                # Check if it's a goodbye message
                if "goodbye" in last_ai_msg.content.lower():
                    break
            
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()