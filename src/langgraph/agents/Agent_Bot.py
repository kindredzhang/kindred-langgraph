from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import SecretStr
import os
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

open_base_url = os.getenv("OPENAI_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"Using OpenAI base URL: {open_base_url}")
print(f"Using OpenAI API key: {openai_api_key}")

if open_base_url is None or openai_api_key is None:
    raise ValueError("Please set the OPENAI_BASE_URL and OPENAI_API_KEY environment variables.")

llm = ChatOpenAI(
    base_url=open_base_url,
    api_key=SecretStr(openai_api_key),
    model="gpt-4o-mini",
    temperature=0.7
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("You: ")

while user_input.lower() != "exit":
    state = agent.invoke(AgentState(messages=[HumanMessage(content=user_input)]))
    user_input = input("You: ")