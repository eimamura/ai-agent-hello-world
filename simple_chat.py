import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()


def create_simple_chat():
    """Create a simple chat model without agents or tools."""
    
    # Initialize GPT-4 model
    chat_model = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    return chat_model


def main():
    """Main function to run the simple chat model."""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable.")
        return
    
    # Create the chat model
    chat = create_simple_chat()
    
    # Set system message
    system_msg = SystemMessage(content="You are a helpful AI assistant. Be concise and friendly.")
    
    print("Simple Chat Model is ready! Type 'quit' to exit.")
    print("-" * 50)
    
    # Store conversation history
    messages = [system_msg]
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message to conversation
        messages.append(HumanMessage(content=user_input))
        
        try:
            # Get response from chat model
            response = chat.invoke(messages)
            print(f"\nAI: {response.content}")
            
            # Add AI response to conversation history
            messages.append(response)
            
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()