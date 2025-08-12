from typing import List, TypedDict
from langgraph.graph import StateGraph
from math import prod

class AgentState(TypedDict):
    names: List[str]
    values: List[int]
    operation: str
    result: str

def sum_values_node(state: AgentState) -> AgentState:
    if state["operation"] == '+':
        state['result'] = f"Hi there {', '.join(state['names'])}, current operation is ' {state['operation']} ', the sum of your values is {sum(state['values'])}."
    elif state["operation"] == '*':
        state['result'] = f"Hi there {', '.join(state['names'])}, current operation is ' {state['operation']} ', the product of your values is {prod(state['values'])}."
    raise ValueError(f"Unsupported operation: {state['operation']}. Supported operations are '+' and '*'.")

graph = StateGraph(AgentState)


graph.add_node("sum_values_node", sum_values_node)

graph.set_entry_point("sum_values_node")
graph.set_finish_point("sum_values_node")

app = graph.compile()

result = app.invoke(AgentState(names=["Alice", "Bob"], values=[1, 2, 3, 4], operation="/", result=""))

print(result["result"])
