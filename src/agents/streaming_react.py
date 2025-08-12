from typing import Annotated, Sequence, TypedDict, Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from enum import Enum
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY or not OPENAI_BASE_URL:
    raise ValueError("Please set OPENAI_API_KEY and OPENAI_BASE_URL in your environment variables.")


class MessageType(Enum):
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"
    THINKING = "thinking"


@dataclass
class StreamMessage:
    """流式消息结构体"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    type: MessageType = MessageType.THINKING
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ConversationSession:
    """会话数据存储结构体"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_question: str = ""
    messages: List[StreamMessage] = field(default_factory=list)
    status: str = "active"  # active, completed, error
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, message: StreamMessage):
        """添加消息到会话"""
        message.session_id = self.session_id
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_question": self.user_question,
            "messages": [msg.to_dict() for msg in self.messages],
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class ConversationStore:
    """简单的内存存储系统（模拟数据库）"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
    
    def create_session(self, user_question: str) -> ConversationSession:
        """创建新会话"""
        session = ConversationSession(user_question=user_question)
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def update_session_status(self, session_id: str, status: str):
        """更新会话状态"""
        if session_id in self.sessions:
            self.sessions[session_id].status = status
            self.sessions[session_id].updated_at = datetime.now()
    
    def list_sessions(self) -> List[ConversationSession]:
        """列出所有会话"""
        return list(self.sessions.values())


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session: ConversationSession
    stream_messages: List[StreamMessage]


# 工具定义
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


tools = [add, subtract, multiply, divide]
model = ChatOpenAI(
    api_key=SecretStr(OPENAI_API_KEY), 
    base_url=OPENAI_BASE_URL, 
    model="gpt-4o-mini", 
    temperature=0.7
).bind_tools(tools)


class StreamingReactAgent:
    """支持流式响应的ReAct Agent"""
    
    def __init__(self):
        self.store = ConversationStore()
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(AgentState)
        
        graph.add_node("reasoning", self._reasoning_node)
        graph.add_node("tool_execution", self._tool_execution_node)
        graph.add_node("final_response", self._final_response_node)
        
        tool_node = ToolNode(tools=tools)
        graph.add_node("tool_node", tool_node)
        
        graph.set_entry_point("reasoning")
        
        graph.add_conditional_edges(
            "reasoning",
            self._should_use_tools,
            {
                "use_tools": "tool_execution",
                "final_answer": "final_response"
            }
        )
        
        graph.add_edge("tool_execution", "tool_node")
        graph.add_edge("tool_node", "reasoning")
        graph.add_edge("final_response", END)
        
        return graph
    
    def _reasoning_node(self, state: AgentState) -> AgentState:
        """推理节点"""
        session = state["session"]
        
        # 添加思考消息
        thinking_msg = StreamMessage(
            type=MessageType.THINKING,
            content="正在分析你的问题..."
        )
        session.add_message(thinking_msg)
        
        # 构建系统消息和调用LLM
        system_message = SystemMessage(
            content="You are a helpful assistant that can perform basic arithmetic operations and answer questions. Think step by step and use tools when needed."
        )
        call_messages = [system_message] + list(state["messages"])
        response = model.invoke(call_messages)
        
        # 添加推理消息
        content = response.content if isinstance(response.content, str) else str(response.content or "正在处理...")
        reasoning_msg = StreamMessage(
            type=MessageType.REASONING,
            content=content,
            metadata={"has_tool_calls": bool(getattr(response, "tool_calls", None))}
        )
        session.add_message(reasoning_msg)
        
        return {
            "messages": [response],
            "session": session,
            "stream_messages": state.get("stream_messages", []) + [thinking_msg, reasoning_msg]
        }
    
    def _tool_execution_node(self, state: AgentState) -> AgentState:
        """工具执行节点"""
        session = state["session"]
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
            for tool_call in getattr(last_message, "tool_calls", []):
                # 添加工具调用消息
                tool_call_msg = StreamMessage(
                    type=MessageType.TOOL_CALL,
                    content=f"调用工具: {tool_call['name']}",
                    metadata={
                        "tool_name": tool_call["name"],
                        "tool_args": tool_call["args"]
                    }
                )
                session.add_message(tool_call_msg)
        
        return {
            "messages": state["messages"],
            "session": session,
            "stream_messages": state.get("stream_messages", [])
        }
    
    def _final_response_node(self, state: AgentState) -> AgentState:
        """最终响应节点"""
        session = state["session"]
        last_message = state["messages"][-1]
        
        # 添加最终答案消息
        content = last_message.content if isinstance(last_message.content, str) else str(last_message.content or "任务完成")
        final_msg = StreamMessage(
            type=MessageType.FINAL_ANSWER,
            content=content,
            metadata={"completed": True}
        )
        session.add_message(final_msg)
        
        # 更新会话状态
        self.store.update_session_status(session.session_id, "completed")
        
        return {
            "messages": state["messages"],
            "session": session,
            "stream_messages": state.get("stream_messages", []) + [final_msg]
        }
    
    def _should_use_tools(self, state: AgentState) -> str:
        """判断是否需要使用工具"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
            return "use_tools"
        return "final_answer"
    
    def process_question(self, question: str) -> ConversationSession:
        """处理用户问题并返回完整会话"""
        # 创建会话
        session = self.store.create_session(question)
        
        # 构建初始状态
        initial_state = AgentState(
            messages=[HumanMessage(content=question)],
            session=session,
            stream_messages=[]
        )
        
        # 运行图
        try:
            final_state = None
            for state in self.app.stream(initial_state, stream_mode="values"):
                final_state = state
                # 这里可以实时发送流式消息到前端
                self._emit_stream_messages(state.get("stream_messages", []))
            
            return final_state["session"] if final_state else session
            
        except Exception as e:
            # 添加错误消息
            error_msg = StreamMessage(
                type=MessageType.ERROR,
                content=f"处理过程中发生错误: {str(e)}",
                metadata={"error_type": type(e).__name__}
            )
            session.add_message(error_msg)
            self.store.update_session_status(session.session_id, "error")
            return session
    
    def _emit_stream_messages(self, messages: List[StreamMessage]):
        """模拟向前端发送流式消息"""
        for msg in messages:
            print(f"[STREAM] {msg.type.value.upper()}: {msg.content}")
    
    def get_frontend_response_format(self, session: ConversationSession) -> Dict[str, Any]:
        """生成前端响应格式"""
        response = {
            "session_id": session.session_id,
            "question": session.user_question,
            "status": session.status,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "steps": []
        }
        
        # 按类型组织消息
        current_step = None
        for msg in session.messages:
            if msg.type == MessageType.THINKING:
                current_step = {
                    "type": "thinking",
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                response["steps"].append(current_step)
            
            elif msg.type == MessageType.REASONING:
                current_step = {
                    "type": "reasoning",
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "has_tool_calls": msg.metadata.get("has_tool_calls", False)
                }
                response["steps"].append(current_step)
            
            elif msg.type == MessageType.TOOL_CALL:
                tool_step = {
                    "type": "tool_execution",
                    "tool_name": msg.metadata.get("tool_name", ""),
                    "tool_args": msg.metadata.get("tool_args", {}),
                    "timestamp": msg.timestamp.isoformat()
                }
                response["steps"].append(tool_step)
            
            elif msg.type == MessageType.FINAL_ANSWER:
                response["steps"].append({
                    "type": "final_answer",
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                })
        
        return response


def main():
    """主函数：演示流式响应系统"""
    agent = StreamingReactAgent()
    
    print("=== 流式响应 ReAct Agent 演示 ===\n")
    
    # 测试问题
    test_questions = [
        "计算 100 / 4 然后乘以 3",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*50}")
        print(f"问题 {i}: {question}")
        print('='*50)
        
        # 处理问题
        session = agent.process_question(question)
        
        print(f"\n会话ID: {session.session_id}")
        print(f"状态: {session.status}")
        
        print(f"\n--- 消息流 ---")
        for msg in session.messages:
            print(f"[{msg.type.value.upper()}] {msg.content}")
            if msg.metadata:
                print(f"  元数据: {msg.metadata}")
        
        # 生成前端格式
        frontend_format = agent.get_frontend_response_format(session)
        print(f"\n--- 前端响应格式 ---")
        print(json.dumps(frontend_format, ensure_ascii=False, indent=2))
    
    # 显示所有会话
    print(f"\n{'='*50}")
    print("所有会话总结")
    print('='*50)
    
    all_sessions = agent.store.list_sessions()
    for session in all_sessions:
        print(f"会话 {session.session_id[:8]}...: {session.user_question} - {session.status}")


if __name__ == "__main__":
    main()