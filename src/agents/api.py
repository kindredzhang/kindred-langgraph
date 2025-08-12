from typing import Dict, Any, Generator
import json

from .streaming_agent import StreamingReActAgent
from .models import StreamMessage, MessageType


class StreamingAPI:
    """流式响应API接口"""
    
    def __init__(self, storage_dir: str = "data"):
        self.agent = StreamingReActAgent(storage_dir)
    
    def ask_question_stream(self, question: str) -> Generator[str, None, Dict[str, Any]]:
        """
        流式处理问题并返回JSON格式的响应
        
        Args:
            question: 用户问题
            
        Yields:
            str: JSON格式的流式消息
            
        Returns:
            Dict: 最终的会话总结
        """
        session = None
        
        try:
            for message in self.agent.stream_process(question):
                # 构建流式响应格式
                stream_response = {
                    "type": "stream",
                    "data": {
                        "message_id": message.id,
                        "message_type": message.type.value,
                        "content": message.content,
                        "metadata": message.metadata,
                        "timestamp": message.timestamp.isoformat()
                    }
                }
                yield json.dumps(stream_response, ensure_ascii=False)
            
            # 获取完整会话信息
            if hasattr(self.agent.stream_process(question), '__iter__'):
                # 重新处理以获取最终会话
                for _ in self.agent.stream_process(question):
                    pass
                
                # 获取最新会话
                sessions = self.agent.storage.storage.list_sessions()
                if sessions:
                    session = self.agent.storage.storage.get_session(sessions[-1])
        
        except Exception as e:
            error_response = {
                "type": "error",
                "data": {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            }
            yield json.dumps(error_response, ensure_ascii=False)
        
        # 返回最终总结
        if session:
            return self._build_final_response(session)
        else:
            return {"error": "Session not found"}
    
    def _build_final_response(self, session) -> Dict[str, Any]:
        """构建最终响应格式"""
        # 按类型分组消息
        messages_by_type = {
            "thinking": [],
            "reasoning": [],
            "tool_calls": [],
            "tool_results": [],
            "final_answer": [],
            "errors": []
        }
        
        for msg in session.messages:
            if msg.type == MessageType.THINKING:
                messages_by_type["thinking"].append({
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                })
            elif msg.type == MessageType.REASONING:
                messages_by_type["reasoning"].append({
                    "content": msg.content,
                    "metadata": msg.metadata,
                    "timestamp": msg.timestamp.isoformat()
                })
            elif msg.type == MessageType.TOOL_CALL:
                messages_by_type["tool_calls"].append({
                    "content": msg.content,
                    "tool_name": msg.metadata.get("tool_name", ""),
                    "tool_args": msg.metadata.get("tool_args", {}),
                    "timestamp": msg.timestamp.isoformat()
                })
            elif msg.type == MessageType.TOOL_RESULT:
                messages_by_type["tool_results"].append({
                    "content": msg.content,
                    "result": msg.metadata.get("result", ""),
                    "timestamp": msg.timestamp.isoformat()
                })
            elif msg.type == MessageType.FINAL_ANSWER:
                messages_by_type["final_answer"].append({
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                })
            elif msg.type == MessageType.ERROR:
                messages_by_type["errors"].append({
                    "content": msg.content,
                    "error_type": msg.metadata.get("error_type", ""),
                    "timestamp": msg.timestamp.isoformat()
                })
        
        return {
            "type": "final_response",
            "session": {
                "session_id": session.session_id,
                "question": session.user_question,
                "status": session.status,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            },
            "conversation_flow": {
                "thinking_steps": messages_by_type["thinking"],
                "reasoning_steps": messages_by_type["reasoning"],
                "tool_operations": messages_by_type["tool_calls"],
                "tool_results": messages_by_type["tool_results"],
                "final_answers": messages_by_type["final_answer"],
                "errors": messages_by_type["errors"]
            },
            "summary": {
                "total_messages": len(session.messages),
                "tools_used": len(messages_by_type["tool_calls"]),
                "has_errors": len(messages_by_type["errors"]) > 0,
                "completion_status": session.status
            }
        }
    
    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """获取会话历史"""
        session = self.agent.storage.storage.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return self._build_final_response(session)
    
    def list_all_sessions(self) -> Dict[str, Any]:
        """列出所有会话"""
        session_ids = self.agent.storage.storage.list_sessions()
        sessions_info = []
        
        for session_id in session_ids:
            session = self.agent.storage.storage.get_session(session_id)
            if session:
                sessions_info.append({
                    "session_id": session.session_id,
                    "question": session.user_question,
                    "status": session.status,
                    "created_at": session.created_at.isoformat(),
                    "message_count": len(session.messages)
                })
        
        return {
            "total_sessions": len(sessions_info),
            "sessions": sessions_info
        }


def simulate_frontend_interaction():
    """模拟前端交互的示例函数"""
    api = StreamingAPI()
    
    print("=== 模拟前端流式交互 ===\n")
    
    questions = [
        "计算 25 + 37 是多少？",
        "100 除以 4 然后乘以 8 等于多少？",
        "帮我算一下 15 * 6 + 10"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"问题 {i}: {question}")
        print('='*60)
        
        print("\n🔄 流式响应:")
        print("-" * 40)
        
        # 模拟流式响应
        final_response = None
        for stream_data in api.ask_question_stream(question):
            try:
                data = json.loads(stream_data)
                if data["type"] == "stream":
                    msg_data = data["data"]
                    print(f"[{msg_data['message_type'].upper()}] {msg_data['content']}")
                    
                    # 如果有元数据，显示相关信息
                    if msg_data.get('metadata'):
                        metadata = msg_data['metadata']
                        if 'tool_name' in metadata:
                            print(f"  → 工具: {metadata['tool_name']}, 参数: {metadata['tool_args']}")
                        elif 'has_tool_calls' in metadata:
                            print(f"  → 需要工具调用: {metadata['has_tool_calls']}")
                
                elif data["type"] == "error":
                    print(f"❌ 错误: {data['data']['error']}")
                    
            except json.JSONDecodeError:
                print(f"❌ 无法解析响应: {stream_data}")
        
        print("\n📊 会话总结:")
        print("-" * 40)
        sessions = api.list_all_sessions()
        if sessions["sessions"]:
            latest_session = sessions["sessions"][-1]
            print(f"会话ID: {latest_session['session_id'][:8]}...")
            print(f"状态: {latest_session['status']}")
            print(f"消息数量: {latest_session['message_count']}")


if __name__ == "__main__":
    simulate_frontend_interaction()