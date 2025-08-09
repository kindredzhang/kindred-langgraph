from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    names: str
    age: int
    skills: List[str]
    result: str

def first_node(state: AgentState) -> AgentState:
    state['result'] = f"hi, {state['names'] }, welcome to the Langgraph system, "
    return state

def second_node(state: AgentState) -> AgentState:
    state['result'] += f"you are {state['age']} years old "
    return state

def third_node(state: AgentState) -> AgentState: 
    state['result'] += f"and your skills are {', '.join(state['skills'])}."
    return state

graph = StateGraph(AgentState)

graph.add_node("first_node", first_node)
graph.add_node("second_node", second_node)
graph.add_node("third_node", third_node)

graph.set_entry_point("first_node")
graph.set_finish_point("third_node")

graph.add_edge("first_node", "second_node")
graph.add_edge("second_node", "third_node")

app = graph.compile()

result = app.invoke(AgentState(names="Alice", age=30, skills=["Python", "Langgraph"], result=""))

print(result["result"])
# Output: hi, Alice, welcome to the Langgraph system, you are 30 years old and your skills are Python, Langgraph.