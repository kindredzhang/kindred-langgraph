from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import HumanMessage # Message for user input
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
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

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

tools = [add, subtract, multiply, divide]

model = ChatOpenAI(api_key=SecretStr(OPENAI_API_KEY), base_url=OPENAI_BASE_URL, model="gpt-4o-mini", temperature=0.7).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_message = SystemMessage(
        content="You are a helpful assistant that can perform basic arithmetic operations. Use the tools provided to answer questions."
    )
    call_messages = [system_message] + list(state["messages"])
    response = model.invoke(call_messages)
    return AgentState(
        messages= [response]
    )

def shoudle_condition(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
        return "countinue"
    else:
        return "end"    

graph = StateGraph(AgentState)

graph.add_node("model_call", model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("model_call")

graph.add_conditional_edges(
    "model_call",
    shoudle_condition,
    {
        "countinue": "tool_node",
        "end": END
    }
)

graph.add_edge("tool_node", "model_call")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = AgentState(messages=[HumanMessage(content="What is 2 + 2? What is 2/0? and tell me a joke.")])
print_stream(app.stream(inputs, stream_mode="values"))

