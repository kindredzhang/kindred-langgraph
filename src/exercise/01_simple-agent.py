from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name : str

def greeting_node(state: AgentState) -> AgentState:

    state['name'] = state['name'] + ", you're doing an amazing job learning LangGraph!"

    return state

graph = StateGraph(AgentState)
graph.add_node("greet", greeting_node)

graph.set_entry_point("greet")
graph.set_finish_point("greet")

app = graph.compile()

result = app.invoke(AgentState(name="Kindred"))

print(result["name"])