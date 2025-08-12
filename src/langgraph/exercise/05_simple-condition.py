from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    number1: int
    operation1: str
    number2: int
    finalNumber1: int
    number3: int
    operation2: str
    number4: int
    finalNumber2: int

def add_node1(state: AgentState) -> AgentState:
    state['finalNumber1'] = state['number1'] + state['number2']
    return state

def subtract_node1(state: AgentState) -> AgentState:
    state['finalNumber1'] = state['number1'] - state['number2']
    return state

def decide_node1 (state: AgentState) -> AgentState:
    if state['operation1'] != 'add' and state['operation1'] != 'subtract':
        raise ValueError("Invalid operation1. Must be 'add' or 'subtract'.")
    if state['operation1'] == 'add':
        return "add_node1"
    elif state['operation1'] == 'subtract':
        return "subtract_node1"
    
def add_node2(state: AgentState) -> AgentState:
    state['finalNumber2'] = state['number3'] + state['number4']
    return state

def subtract_node2(state: AgentState) -> AgentState:
    state['finalNumber2'] = state['number3'] - state['number4']
    return state

def decide_node2 (state: AgentState) -> AgentState:
    if state['operation2'] != 'add' and state['operation2'] != 'subtract':
        raise ValueError("Invalid operation1. Must be 'add' or 'subtract'.")
    if state['operation2'] == 'add':
        return "add_node2"
    elif state['operation2'] == 'subtract':
        return "subtract_node2"
    
graph = StateGraph(AgentState)

graph.add_node("decide_node1", lambda state: state)
graph.add_node("add_node1", add_node1)
graph.add_node("subtract_node1", subtract_node1)
graph.add_node("decide_node2", lambda state: state)
graph.add_node("add_node2", add_node2)
graph.add_node("subtract_node2", subtract_node2)

graph.add_edge(START, "decide_node1")
graph.add_conditional_edges(
    "decide_node1",
    decide_node1,
    {
        "add_node1": "add_node1",
        "subtract_node1": "subtract_node1"
    }
)
graph.add_edge("add_node1", "decide_node2")
graph.add_edge("subtract_node1", "decide_node2")

graph.add_conditional_edges(
    "decide_node2",
    decide_node2,
    {
        "add_node2": "add_node2",
        "subtract_node2": "subtract_node2"
    }
)
graph.add_edge("add_node2", END)
graph.add_edge("subtract_node2", END)

app = graph.compile()

answer = app.invoke(AgentState(number1=10, operation1='add', number2=5, finalNumber1=0, number3=20, operation2='subtract', number4=15, finalNumber2=0))

print(f"Final Number 1: {answer['finalNumber1']}")
print(f"Final Number 2: {answer['finalNumber2']}")
# Output: Final Number 1: 15
# Output: Final Number 2: 5

