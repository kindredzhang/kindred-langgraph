from typing import List, TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    values: List[int]
    name: str
    result: str

def sum_values_node(state: AgentState) -> AgentState:
    
    state['result'] = f"Hi there {state['name']}, the sum of your values is {sum(state['values'])}."
    return state

graph = StateGraph(AgentState)

graph.add_node("sum_values", sum_values_node)

graph.set_entry_point("sum_values")
graph.set_finish_point("sum_values")

app = graph.compile()

result = app.invoke(AgentState(values=[1, 2, 3, 4], name="Kindred", result=""))

print(result["result"])