from typing import List, Dict, Any, TypedDict
import random
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    player_name: str
    target_number: int
    guesses: List[int]
    attempts: int
    hint: str
    lower_bound: int
    upper_bound: int

def initialize_game(state: AgentState) -> AgentState:
    state['player_name'] = f"Hi, {state['player_name']}"
    state['target_number'] = random.randint(state['lower_bound'], state['upper_bound'])
    state['guesses'] = []
    state['attempts'] = 0
    state["hint"] = "Game started! Try to guess the number."
    state['lower_bound'] = 1
    state['upper_bound'] = 20
    return state

def guess_node(state: AgentState) -> AgentState:
    possible_guesses = [i for i in range(state["lower_bound"], state["upper_bound"] + 1) if i not in state["guesses"]]
    if possible_guesses:
        guess = random.choice(possible_guesses)
    else:
        guess = random.randint(state["lower_bound"], state["upper_bound"])
    
    state["guesses"].append(guess)
    state["attempts"] += 1
    print(f"Attempt {state['attempts']}: Guessing {guess} (Current range: {state['lower_bound']}-{state['upper_bound']})")
    return state

def hint_node(state: AgentState) -> AgentState:
    if state['guesses'][-1] < state['target_number']:
        state['hint'] = "Too low! Try a higher number."
        print(state['hint'])
    elif state['guesses'][-1] > state['target_number']:
        state['hint'] = "Too high! Try a lower number."
        print(state['hint'])
    else:
        state['hint'] = "Congratulations! You've guessed the number!"
        print(f"Success! {state['hint']}")
    return state

def should_continue(state: AgentState) -> str:
    if state['guesses'][-1] == state['target_number']:
        print(f"Game over! {state['player_name']} won!")
        return "END"
    elif state['attempts'] >= 10:
        print(f"Game over! {state['player_name']} has used all attempts.")
        state['hint'] = "Game over! You've used all attempts."
        return "END"
    else:
        print(f"{state['player_name']} has made {state['attempts']} attempts. Keep guessing!")
        return "guess_node"
    
graph = StateGraph(AgentState)

graph.add_node("initialize_game", initialize_game)
graph.add_node("guess_node", guess_node)
graph.add_node("hint_node", hint_node)

graph.set_entry_point("initialize_game")
graph.add_edge("initialize_game", "guess_node")
graph.add_edge("guess_node", "hint_node")

graph.add_conditional_edges(
    "hint_node",
    should_continue,
    {
        "guess_node": "guess_node",
        "END": END
    }
)

app = graph.compile()

answer = app.invoke(AgentState(
    player_name="Alice",
    target_number=0,
    guesses=[],
    attempts=0,
    hint="",
    lower_bound=1,
    upper_bound=20
))
