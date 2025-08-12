from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from langchain_chroma import Chroma
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY or not OPENAI_BASE_URL:
    raise ValueError("Please set OPENAI_API_KEY and OPENAI_BASE_URL in your environment variables.")

# 
llm = ChatOpenAI(
    api_key=SecretStr(OPENAI_API_KEY),
    base_url=OPENAI_BASE_URL,
    model="gpt-4o-mini"
)

# 向量 
# 每一个向量之间的转换都需要向量模型的参与
# 每一个llm 都有对应的向量模型
embeddings = OpenAIEmbeddings(
    api_key=SecretStr(OPENAI_API_KEY),
    base_url=OPENAI_BASE_URL,
    model="text-embedding-3-small",
)

pdf_path = "张家豪-Java开发工程师.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
except Exception as e:
    raise RuntimeError(f"Failed to load and split PDF: {str(e)}")

# 文件切割器 把pdf识别到的文件 转换为 ai可以看懂的格式
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
)

pages_split = text_splitter.split_documents(pages)

# 把上述文件转换为向量后存储的地方，换句话来说就是db文件 数据库
persist_directory = "/home/kindredzhang/data/projects/ai/kindred-langgraph/src/langgraph/agents"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# 数据库名
collection_name = "rag_agent_collection"

# chroma实例
try:
    vector_store = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
except Exception as e:
    raise RuntimeError(f"Failed to create vector store: {str(e)}")

# 检索 当我使用检索功能的时候会查询到相似的数据 search_kwargs：返回数量
# 比如 pdf文件拆成了30块，有十条信息和我的问题相匹配，默认返回5条
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """Retrieve relevant information from the resume document."""
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the resume document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools=tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> str:
    """Check if the last message contains tool calls."""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and getattr(last_message, 'tool_calls', None):
        return "tools"
    else:
        return "end"

system_prompt = """
You are an intelligent AI assistant who answers questions about 张家豪's resume and professional background based on the PDF document loaded into your knowledge base.

IMPORTANT: You MUST always use the retriever_tool to search the document before answering ANY question about 张家豪. Never refuse to answer without checking the document first.

Use the retriever tool available to answer questions about his work experience, skills, education, projects, contact information, and other professional information. You can make multiple calls if needed.

If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.

Remember: Always check the document using the retriever_tool before responding to any question about 张家豪.
"""

def call_llm(state: AgentState) -> AgentState:
    """Call the LLM with the current state and return the response."""
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    response = llm.invoke(messages)
    print(f"\n🤖 AI: {response.content}")    
    return AgentState(messages=list(state['messages']) + [response])

# Create the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", call_llm)
graph.add_node("tools", ToolNode(tools))

# Add conditional edges
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# Add edge from tools back to agent
graph.add_edge("tools", "agent")

# Set entry point and compile
graph.set_entry_point("agent")
app = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)]
        result = app.invoke(AgentState(messages=messages))
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

if __name__ == "__main__":
    running_agent()


# 提问 》》 拿着提问的问题去匹配# 》》 拿着匹配到的内容拼接第一次提问的问题去重新问问题 》》 回答