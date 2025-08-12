"""
Streaming ReAct Agent Package

一个支持流式响应的ReAct Agent系统，包含：
- 实时流式响应
- 工具调用支持  
- 多段信息返回结构
- 简单文件存储系统
- 前端响应格式模拟

使用示例:
    from agents.api import StreamingAPI
    
    api = StreamingAPI()
    for response in api.ask_question_stream("计算25+37"):
        print(response)
"""

from .models import StreamMessage, MessageType, ConversationSession
from .storage import SimpleFileStorage, StreamingStorage  
from .streaming_agent import StreamingReActAgent
from .api import StreamingAPI

__version__ = "1.0.0"
__all__ = [
    "StreamMessage",
    "MessageType", 
    "ConversationSession",
    "SimpleFileStorage",
    "StreamingStorage",
    "StreamingReActAgent",
    "StreamingAPI"
]