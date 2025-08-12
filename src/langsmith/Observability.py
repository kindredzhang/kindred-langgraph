from dotenv import load_dotenv
from pydantic import SecretStr
import os
from langsmith.wrappers import wrap_openai

load_dotenv()


LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_API_KEY = SecretStr(os.getenv("LANGSMITH_API_KEY") or "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")
OPENAI_API_KEY = SecretStr(os.getenv("OPENAI_API_KEY") or "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

import openai
from langsmith import wrappers

# Use OpenAI client same as you normally would.
# After each conversation, you can view the local chat records at https://smith.langchain.com/
# On the web interface, you can clearly see the details of each conversation (also called a "Trace"), including:

# Complete call chain: You can see every step from user input to final output. 
# If your application calls tools, retrieves documents, or has multiple intermediate steps, each will be displayed step by step.

# Inputs and outputs: For each step, you can see the input (e.g., the full prompt sent to the model) and the output (the model's raw response).

# Performance metrics: This includes key indicators such as token usage, response latency, and cost.

# Errors and debugging: If any step fails, it will be clearly marked, helping you quickly locate the root cause.
client = wrappers.wrap_openai(openai.OpenAI())

# Chat API:
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

messages = [
    ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant."),
    ChatCompletionUserMessageParam(role="user", content="What physics breakthroughs do you predict will happen by 2300?"),
]
completion = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages
)
print(completion.choices[0].message.content)

completion_response = client.completions.create(
    model="gpt-4o-mini",
    prompt="hello?",
    max_tokens=256,
)
print(completion_response.choices[0].text)