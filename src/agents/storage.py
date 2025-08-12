import os
import json
from typing import Optional, List
from datetime import datetime

from .models import ConversationSession, StreamMessage


class SimpleFileStorage:
    """简单的文件存储系统，用于模拟数据库"""
    
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = storage_dir
        self.sessions_file = os.path.join(storage_dir, "sessions.txt")
        self.messages_file = os.path.join(storage_dir, "messages.txt")
        self._ensure_storage_dir()
        self._init_files()
    
    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
    
    def _init_files(self):
        """初始化存储文件"""
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                f.write("# Sessions Table Structure\n")
                f.write("# session_id|user_question|status|created_at|updated_at\n")
                f.write("# Format: UUID|TEXT|VARCHAR(20)|TIMESTAMP|TIMESTAMP\n\n")
        
        if not os.path.exists(self.messages_file):
            with open(self.messages_file, 'w', encoding='utf-8') as f:
                f.write("# Messages Table Structure\n")
                f.write("# message_id|session_id|type|content|metadata|timestamp\n")
                f.write("# Format: UUID|UUID|VARCHAR(20)|TEXT|JSON|TIMESTAMP\n\n")
    
    def save_session(self, session: ConversationSession) -> None:
        """保存会话到文件"""
        session_line = f"{session.session_id}|{session.user_question}|{session.status}|{session.created_at.isoformat()}|{session.updated_at.isoformat()}\n"
        
        # 读取现有会话，检查是否已存在
        existing_sessions = []
        if os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                existing_sessions = f.readlines()
        
        # 更新或添加会话
        updated = False
        for i, line in enumerate(existing_sessions):
            if line.startswith(session.session_id):
                existing_sessions[i] = session_line
                updated = True
                break
        
        if not updated:
            existing_sessions.append(session_line)
        
        # 写回文件
        with open(self.sessions_file, 'w', encoding='utf-8') as f:
            f.writelines(existing_sessions)
    
    def save_message(self, message: StreamMessage, session_id: str) -> None:
        """保存消息到文件"""
        metadata_json = json.dumps(message.metadata, ensure_ascii=False)
        message_line = f"{message.id}|{session_id}|{message.type.value}|{message.content}|{metadata_json}|{message.timestamp.isoformat()}\n"
        
        with open(self.messages_file, 'a', encoding='utf-8') as f:
            f.write(message_line)
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """从文件读取会话"""
        if not os.path.exists(self.sessions_file):
            return None
        
        with open(self.sessions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split('|')
                if len(parts) >= 5 and parts[0] == session_id:
                    session = ConversationSession(
                        session_id=parts[0],
                        user_question=parts[1],
                        status=parts[2],
                        created_at=datetime.fromisoformat(parts[3]),
                        updated_at=datetime.fromisoformat(parts[4])
                    )
                    
                    # 加载消息
                    session.messages = self.get_messages_by_session(session_id)
                    return session
        
        return None
    
    def get_messages_by_session(self, session_id: str) -> List[StreamMessage]:
        """获取会话的所有消息"""
        messages = []
        
        if not os.path.exists(self.messages_file):
            return messages
        
        with open(self.messages_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split('|', 5)  # 限制分割次数，因为content可能包含|
                if len(parts) >= 6 and parts[1] == session_id:
                    try:
                        metadata = json.loads(parts[4]) if parts[4] else {}
                    except json.JSONDecodeError:
                        metadata = {}
                    
                    from .models import MessageType
                    message = StreamMessage(
                        id=parts[0],
                        type=MessageType(parts[2]),
                        content=parts[3],
                        metadata=metadata,
                        timestamp=datetime.fromisoformat(parts[5])
                    )
                    messages.append(message)
        
        return sorted(messages, key=lambda x: x.timestamp)
    
    def list_sessions(self) -> List[str]:
        """列出所有会话ID"""
        session_ids = []
        
        if not os.path.exists(self.sessions_file):
            return session_ids
        
        with open(self.sessions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split('|')
                if len(parts) >= 1:
                    session_ids.append(parts[0])
        
        return session_ids
    
    def update_session_status(self, session_id: str, status: str) -> None:
        """更新会话状态"""
        session = self.get_session(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.now()
            self.save_session(session)


class StreamingStorage:
    """专门用于流式数据的存储管理器"""
    
    def __init__(self, storage: SimpleFileStorage):
        self.storage = storage
    
    async def stream_save_message(self, message: StreamMessage, session_id: str):
        """异步保存消息（这里是同步实现的模拟）"""
        self.storage.save_message(message, session_id)
        return message
    
    def create_session(self, user_question: str) -> ConversationSession:
        """创建新会话"""
        session = ConversationSession(user_question=user_question)
        self.storage.save_session(session)
        return session
    
    def add_stream_message(self, session: ConversationSession, message: StreamMessage) -> StreamMessage:
        """添加流式消息"""
        session.add_message(message)
        self.storage.save_message(message, session.session_id)
        self.storage.save_session(session)  # 更新会话时间戳
        return message