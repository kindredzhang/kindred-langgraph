from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from pydantic import SecretStr
import os


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY or not OPENAI_BASE_URL:
    raise ValueError("Please set OPENAI_API_KEY and OPENAI_BASE_URL in your environment variables.")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("Please set TAVILY_API_KEY in your environment variables.")

class AgentState(TypedDict):
    """State of the agent, containing messages."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool()
def search(query: str) -> str:
    """Search for a query using TVLY."""

    tavily_client = TavilySearch(
                api_key=TAVILY_API_KEY,
                max_results=3,
                topic="general"
            )
    
    response = tavily_client.invoke(query)    
    # response's structure is:
            # {
            #     'query': 'What happened at the last wimbledon',
            #     'follow_up_questions': None,
            #     'answer': None,
            #     'images': [],
            #     'results': [{'title': "Andy Murray pulls out of the men's singles draw at his last Wimbledon",
            #                 'url': 'https://www.nbcnews.com/news/sports/andy-murray-wimbledon-tennis-singles-draw-rcna159912',
            #                 'content': "NBC News Now LONDON — Andy Murray, one of the last decade's most successful ..."
            #                 'score': 0.6755297,
            #                 'raw_content': None
            #                 }],
            #     'response_time': 1.31
            # }
    
    final_result = ""
    for result in response['results']:
        final_result += f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content']}\n\n"
    
    if not final_result:
        return "No relevant results found."
    
    return final_result


tools = [search]

model = ChatOpenAI(api_key=SecretStr(OPENAI_API_KEY), base_url=OPENAI_BASE_URL, model="gpt-4.1-mini", temperature=0.7).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    
    system_message = SystemMessage(
        content="You are a helpful assistant that MUST use the search tool to answer ALL questions. Always search for current information before responding, even for general knowledge questions."
    )
    final_messages = [system_message] + list(state["messages"])
    response = model.invoke(final_messages)
    return AgentState(
        messages= [response]
    )

def shoudle_condition(state: AgentState) -> str:
    if state["messages"] and len(state["messages"]) > 0:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
            return "countinue"
        else:
            return "end"
    else:
        return "end"
    
graph = StateGraph(AgentState)

graph.add_node("model_call", model_call)

tools_node = ToolNode(tools=tools)
graph.add_node("tools_node", tools_node)

graph.set_entry_point("model_call")
graph.add_conditional_edges(
    "model_call",
    shoudle_condition,
    {
        "countinue": "tools_node",
        "end": END
    }
)
graph.add_edge("tools_node", "model_call")

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = AgentState(messages=[HumanMessage(content="What is the latest version of LangGraph and what new features does it have?")])
print_stream(agent.stream(inputs, stream_mode="values"))

#  AgentState a= new AgentState()
# List<BaseMessages> message = new ArrarList();
# HumanMessage h = new HumanMessage();
# h.setContent("hello")
# message.add(h)
# a.setMessahes(message);