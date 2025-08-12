import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from pydantic import SecretStr

# Load environment variables
load_dotenv()

OPENAI_API_KEY = SecretStr(os.getenv("OPENAI_API_KEY") or "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

def simple_chat():
    """Simple LangChain chat application with conversation history"""
    
    # Initialize the chat model
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.7,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    
    # Chat history to maintain context
    from langchain.schema import BaseMessage
    messages: list[BaseMessage] = [
        SystemMessage(content="You are a helpful AI assistant. Be concise and friendly.")
    ]
    
    print("> Simple LangChain Chat App")
    print("Type 'quit' to exit, 'clear' to clear history\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye! =K")
            break
        elif user_input.lower() == 'clear':
            messages = [SystemMessage(content="You are a helpful AI assistant. Be concise and friendly.")]
            print("= Chat history cleared!")
            continue
        elif not user_input:
            continue
        
        # Add user message to history
        messages.append(HumanMessage(content=user_input))
        
        try:
            # Get AI response
            response = llm.invoke(messages)
            ai_response = response.content
            
            # Add AI response to history
            messages.append(AIMessage(content=ai_response))
            
            # Display response
            print(f"AI: {ai_response}\n")
            
        except Exception as e:
            print(f"L Error: {e}\n")

if __name__ == "__main__":
    simple_chat()