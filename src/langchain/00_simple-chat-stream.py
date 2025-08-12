import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from pydantic import SecretStr

# Load environment variables
load_dotenv()

# LangSmith configuration
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="pr-healthy-weedkiller-33"

# OpenAI configuration
OPENAI_API_KEY = SecretStr(os.getenv("OPENAI_API_KEY") or "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

def simple_chat_stream():
    """Simple LangChain chat application with streaming support"""
    
    # Initialize the chat model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
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
    
    # Display LangSmith tracing status
    if LANGSMITH_API_KEY:
        print(f"📊 LangSmith tracing enabled - Project: {LANGSMITH_PROJECT}")
    else:
        print("💡 Set LANGSMITH_API_KEY to enable tracing")
    print()
    
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
            # Stream AI response with LangSmith tracing
            print("AI: ", end="", flush=True)
            full_response = ""
            
            for chunk in llm.stream(messages):
                if chunk.content:
                    print(str(chunk.content), end="", flush=True)
                    full_response += str(chunk.content)
            
            print("\n")  # Add newline after streaming
            
            # Add AI response to history
            messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    simple_chat_stream()