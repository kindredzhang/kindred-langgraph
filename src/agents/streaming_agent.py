from typing import Annotated, Sequence, TypedDict, Generator, Dict, Any
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr
import os

from .models import StreamMessage, MessageType, ConversationSession, ToolCall, ToolResult
from .storage import StreamingStorage, SimpleFileStorage

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your environment variables.")


class StreamingAgentState(TypedDict):
    """流式Agent状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session: ConversationSession
    current_step: str


# 定义工具
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool
def divide_numbers(a: int, b: int) -> float:
    """Divide two numbers. Returns error if division by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b


class StreamingReActAgent:
    """支持真正流式响应的ReAct Agent"""
    
    def __init__(self, storage_dir: str = "data"):
        self.storage = StreamingStorage(SimpleFileStorage(storage_dir))
        self.tools = [add_numbers, multiply_numbers, divide_numbers]
        self.model = ChatOpenAI(
            api_key=SecretStr(OPENAI_API_KEY),
            base_url=OPENAI_BASE_URL,
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=True  # 启用流式响应
        ).bind_tools(self.tools)
        
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph状态图"""
        graph = StateGraph(StreamingAgentState)
        
        # 添加节点
        graph.add_node("thinking", self._thinking_node)
        graph.add_node("reasoning", self._reasoning_node) 
        graph.add_node("tool_calling", self._tool_calling_node)
        graph.add_node("tool_execution", ToolNode(self.tools))
        graph.add_node("final_answer", self._final_answer_node)
        
        # 设置入口点
        graph.set_entry_point("thinking")
        
        # 添加边
        graph.add_edge("thinking", "reasoning")
        graph.add_conditional_edges(
            "reasoning",
            self._should_use_tools,
            {
                "use_tools": "tool_calling",
                "final_answer": "final_answer"
            }
        )
        graph.add_edge("tool_calling", "tool_execution")
        graph.add_edge("tool_execution", "reasoning")
        graph.add_edge("final_answer", END)
        
        return graph
    
    def _thinking_node(self, state: StreamingAgentState) -> StreamingAgentState:
        """思考节点 - 分析用户问题"""
        session = state["session"]
        
        thinking_msg = StreamMessage(
            type=MessageType.THINKING,
            content="🤔 让我分析一下你的问题...",
            metadata={"step": "initial_thinking"}
        )
        
        self.storage.add_stream_message(session, thinking_msg)
        
        return {
            "messages": state["messages"],
            "session": session,
            "current_step": "thinking"
        }
    
    def _reasoning_node(self, state: StreamingAgentState) -> StreamingAgentState:
        """推理节点 - 调用LLM进行推理"""
        session = state["session"]
        
        reasoning_msg = StreamMessage(
            type=MessageType.REASONING,
            content="💭 正在推理和分析...",
            metadata={"step": "reasoning"}
        )
        self.storage.add_stream_message(session, reasoning_msg)
        
        # 构建系统提示
        system_message = SystemMessage(
            content="""你是一个helpful的数学助手。你可以使用提供的工具来进行计算。

当用户提出问题时，请按以下步骤思考：
1. 理解用户的问题
2. 判断是否需要使用工具
3. 如果需要，选择合适的工具并说明原因
4. 执行计算
5. 给出最终答案

请始终用中文回复，并解释你的思考过程。"""
        )
        
        # 调用LLM
        call_messages = [system_message] + list(state["messages"])
        response = self.model.invoke(call_messages)
        
        # 更新推理内容
        content = response.content if isinstance(response.content, str) else str(response.content or "")
        reasoning_update = StreamMessage(
            type=MessageType.REASONING,
            content=f"💭 推理结果: {content}",
            metadata={
                "step": "reasoning_complete",
                "has_tool_calls": bool(getattr(response, "tool_calls", None))
            }
        )
        self.storage.add_stream_message(session, reasoning_update)
        
        return {
            "messages": [response],
            "session": session,
            "current_step": "reasoning"
        }
    
    def _tool_calling_node(self, state: StreamingAgentState) -> StreamingAgentState:
        """工具调用节点 - 准备工具调用"""
        session = state["session"]
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
            for tool_call in getattr(last_message, "tool_calls", []):
                tool_msg = StreamMessage(
                    type=MessageType.TOOL_CALL,
                    content=f"🔧 准备调用工具: {tool_call['name']}",
                    metadata={
                        "tool_name": tool_call["name"],
                        "tool_args": tool_call["args"],
                        "tool_id": tool_call.get("id", "")
                    }
                )
                self.storage.add_stream_message(session, tool_msg)
        
        return {
            "messages": state["messages"],
            "session": session,
            "current_step": "tool_calling"
        }
    
    def _final_answer_node(self, state: StreamingAgentState) -> StreamingAgentState:
        """最终答案节点"""
        session = state["session"]
        last_message = state["messages"][-1]
        
        content = last_message.content if isinstance(last_message.content, str) else str(last_message.content or "")
        
        final_msg = StreamMessage(
            type=MessageType.FINAL_ANSWER,
            content=f"✅ 最终答案: {content}",
            metadata={"step": "completed"}
        )
        self.storage.add_stream_message(session, final_msg)
        
        # 更新会话状态
        self.storage.storage.update_session_status(session.session_id, "completed")
        
        return {
            "messages": state["messages"],
            "session": session,
            "current_step": "completed"
        }
    
    def _should_use_tools(self, state: StreamingAgentState) -> str:
        """判断是否需要使用工具"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
            return "use_tools"
        return "final_answer"
    
    def stream_process(self, question: str) -> Generator[StreamMessage, None, ConversationSession]:
        """流式处理用户问题"""
        # 创建会话
        session = self.storage.create_session(question)
        
        # 构建初始状态
        initial_state = StreamingAgentState(
            messages=[HumanMessage(content=question)],
            session=session,
            current_step="start"
        )
        
        try:
            # 运行图并流式返回消息
            for state in self.app.stream(initial_state, stream_mode="values"):
                current_session = state["session"]
                
                # 获取最新消息并流式返回
                if current_session.messages:
                    latest_messages = current_session.messages[-1:]  # 只返回最新消息
                    for msg in latest_messages:
                        yield msg
                
                # 模拟实时处理延迟
                import time
                time.sleep(0.5)
            
            return session
            
        except Exception as e:
            # 错误处理
            error_msg = StreamMessage(
                type=MessageType.ERROR,
                content=f"❌ 处理过程中发生错误: {str(e)}",
                metadata={"error_type": type(e).__name__}
            )
            self.storage.add_stream_message(session, error_msg)
            self.storage.storage.update_session_status(session.session_id, "error")
            yield error_msg
            return session
    
    def get_session_messages(self, session_id: str) -> list:
        """获取会话的所有消息"""
        session = self.storage.storage.get_session(session_id)
        return session.messages if session else []