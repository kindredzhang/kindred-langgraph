from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr
import os

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_API_KEY = SecretStr(os.getenv("LANGSMITH_API_KEY") or "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")

OPENAI_API_KEY = SecretStr(os.getenv("OPENAI_API_KEY") or "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = Client(api_key=OPENAI_API_KEY.get_secret_value(), api_url=OPENAI_BASE_URL)

print(LANGSMITH_API_KEY)
# 0 create evaluation dataset
dataset = client.create_dataset(
    dataset_name="Sample dataset", description="A sample dataset in LangSmith."
)

# Create examples
examples = [
    {
        "inputs": {"question": "Which country is Mount Kilimanjaro located in?"},
        "outputs": {"answer": "Mount Kilimanjaro is located in Tanzania."},
    },
    {
        "inputs": {"question": "What is Earth's lowest point?"},
        "outputs": {"answer": "Earth's lowest point is The Dead Sea."},
    },
]

# Add examples to the dataset
client.create_examples(dataset_id=dataset.id, examples=examples)

# 1 create app
def create_qa_chain(inputs):
    """一个简单的问答Chain"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": inputs["question"]})

# 2 define Evaluator client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:gpt-4.1-mini",
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )
    return eval_result

# 3 run and view evaluation results
# Define the target function for evaluation
def target(inputs):
    return {"answer": create_qa_chain(inputs)}

# Run evaluation using built-in "qa" evaluator
# After running the evaluation, a link will be provided to view the results in langsmith
experiment_results = client.evaluate(
    target,
    data="Sample dataset",
    evaluators=[
        correctness_evaluator,
        # can add multiple evaluators here
    ],
    experiment_prefix="first-eval-in-langsmith",
    max_concurrency=2,
)