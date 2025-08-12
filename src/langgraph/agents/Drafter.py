from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY or not OPENAI_BASE_URL:
    raise ValueError("Please set OPENAI_API_KEY and OPENAI_BASE_URL in your environment variables.")

class DrafterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

document_content = ""

@tool
def update(content: str) -> str:
    """Update the document content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the document content to a file."""
    global document_content
    if not document_content:
        return "Document is empty. Nothing to save."
    if not filename.endswith('.txt'):
        filename += '.txt'

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        return f"Document saved successfully to {filename}."
    except Exception as e:
        return f"An error occurred while saving the document: {str(e)}"
    
tools = [update, save]

model = ChatOpenAI(api_key=SecretStr(OPENAI_API_KEY), base_url=OPENAI_BASE_URL, model="gpt-4o-mini").bind_tools(tools=tools)

def our_agent(state: DrafterState) -> DrafterState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    IMPORTANT RULES:
    - When you create or write new content (like emails, letters, documents), ALWAYS use the 'update' tool to save it to the document.
    - When the user asks you to save content, use the 'save' tool with the filename they specify.
    - If the user wants to modify existing content, use the 'update' tool with the complete updated content.
    - Always use the update tool before showing content to the user, so they can save it later.
    
    The current document content is: {document_content if document_content else "(empty)"}
    """)
    
    all_messages = [system_prompt] + list(state['messages'])

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")

    return DrafterState(messages=list(state['messages']) + [response])

def should_continue(state: DrafterState) -> str:
    messages = state['messages']
    if not messages:
        return "user_input"
    
    last_message = messages[-1]
    
    # Check if user said bye
    if isinstance(last_message, HumanMessage):
        content = last_message.content
        if isinstance(content, str) and content.lower() in ['bye', 'quit', 'exit']:
            return "end"
    
    # If the last message has tool calls, go to tools
    if hasattr(last_message, 'tool_calls') and getattr(last_message, 'tool_calls', None):
        return "tools"
    
    # If we just processed a tool result that indicates saving, continue conversation
    if isinstance(last_message, ToolMessage):
        return "user_input"
        
    # Otherwise, wait for user input
    return "user_input"

def get_user_input(state: DrafterState) -> DrafterState:
    user_input = input("\n👤 You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        return DrafterState(messages=list(state['messages']) + [HumanMessage(content="bye")])
    
    user_message = HumanMessage(content=user_input)
    return DrafterState(messages=list(state['messages']) + [user_message])

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(DrafterState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.add_node("user_input", get_user_input)

graph.set_entry_point("user_input")  # Start with user input
graph.add_edge("user_input", "agent")  # User input goes to agent

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "user_input": "user_input", 
        "end": END
    }
)

graph.add_edge("tools", "agent")  # After tools, go back to agent

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    print("👋 Hi! I'm Drafter, your document writing assistant.")
    print("📝 I can help you create, update, and save documents.")
    print("💬 Type 'quit', 'exit', or 'bye' to end the conversation.\n")
    

    state = DrafterState(messages=[])
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()