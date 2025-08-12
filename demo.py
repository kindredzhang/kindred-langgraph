#!/usr/bin/env python3
"""
流式响应ReAct Agent演示
展示真正的流式响应、工具调用和多段信息返回结构
"""

import time
import json
from typing import Generator

from src.agents.api import StreamingAPI


def print_stream_header(title: str):
    """打印流式响应标题"""
    print(f"\n{'=' * 60}")
    print(f"🚀 {title}")
    print('=' * 60)


def print_stream_separator():
    """打印分隔符"""
    print("-" * 60)


def demonstrate_streaming_response():
    """演示流式响应功能"""
    api = StreamingAPI("demo_data")
    
    print("🎯 流式响应ReAct Agent演示系统")
    print("📚 展示：thinking → reasoning → tool_calling → final_answer")
    
    # 测试问题列表
    test_questions = [
        {
            "question": "请计算 25 + 37 的结果",
            "description": "简单加法运算"
        },
        {
            "question": "帮我计算 (12 * 8) / 4 的值",
            "description": "多步骤计算"
        },
        {
            "question": "100除以5，然后加上20，最后乘以2等于多少？",
            "description": "复杂的多步骤计算"
        }
    ]
    
    for i, test_case in enumerate(test_questions, 1):
        print_stream_header(f"测试案例 {i}: {test_case['description']}")
        print(f"❓ 问题: {test_case['question']}")
        print_stream_separator()
        
        # 记录开始时间
        start_time = time.time()
        
        # 流式处理响应
        stream_messages = []
        try:
            for stream_data in api.ask_question_stream(test_case['question']):
                try:
                    data = json.loads(stream_data)
                    
                    if data["type"] == "stream":
                        msg_data = data["data"]
                        stream_messages.append(msg_data)
                        
                        # 实时显示流式消息
                        print(f"📨 [{msg_data['message_type'].upper()}] {msg_data['content']}")
                        
                        # 显示元数据
                        if msg_data.get('metadata') and msg_data['metadata']:
                            metadata = msg_data['metadata']
                            if 'tool_name' in metadata:
                                print(f"   🔧 工具: {metadata['tool_name']}")
                                print(f"   📝 参数: {json.dumps(metadata['tool_args'], ensure_ascii=False)}")
                            elif 'has_tool_calls' in metadata:
                                print(f"   🤖 需要工具: {'是' if metadata['has_tool_calls'] else '否'}")
                            elif 'step' in metadata:
                                print(f"   📍 步骤: {metadata['step']}")
                        
                        print()  # 空行分隔
                        
                        # 模拟实时效果
                        time.sleep(0.3)
                    
                    elif data["type"] == "error":
                        print(f"❌ 错误: {data['data']['error']}")
                        print(f"   错误类型: {data['data']['error_type']}")
                
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解析错误: {e}")
                    print(f"   原始数据: {stream_data}")
        
        except Exception as e:
            print(f"❌ 处理过程中发生错误: {str(e)}")
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        print_stream_separator()
        print(f"⏱️  处理耗时: {duration:.2f}秒")
        print(f"📊 流式消息数量: {len(stream_messages)}")
        
        # 显示消息类型统计
        msg_types = {}
        for msg in stream_messages:
            msg_type = msg['message_type']
            msg_types[msg_type] = msg_types.get(msg_type, 0) + 1
        
        print("📈 消息类型统计:")
        for msg_type, count in msg_types.items():
            print(f"   {msg_type}: {count} 条")
        
        print()


def demonstrate_frontend_format():
    """演示前端响应格式"""
    api = StreamingAPI("demo_data")
    
    print_stream_header("前端响应格式演示")
    
    question = "计算 15 * 4 + 25 的结果"
    print(f"❓ 测试问题: {question}")
    print_stream_separator()
    
    # 收集所有流式响应
    all_responses = []
    for stream_data in api.ask_question_stream(question):
        all_responses.append(stream_data)
        # 不实时打印，收集完后统一展示
    
    print("📱 模拟前端接收的数据流:")
    print()
    
    for i, response in enumerate(all_responses, 1):
        print(f"第 {i} 条消息:")
        print("```json")
        try:
            # 格式化JSON输出
            data = json.loads(response)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print(response)
        print("```")
        print()
    
    # 获取会话历史
    sessions = api.list_all_sessions()
    if sessions["sessions"]:
        latest_session_id = sessions["sessions"][-1]["session_id"]
        history = api.get_session_history(latest_session_id)
        
        print("📚 完整会话历史结构:")
        print("```json")
        print(json.dumps(history, ensure_ascii=False, indent=2))
        print("```")


def demonstrate_session_management():
    """演示会话管理功能"""
    api = StreamingAPI("demo_data")
    
    print_stream_header("会话管理演示")
    
    # 列出所有会话
    sessions = api.list_all_sessions()
    print(f"📊 总会话数: {sessions['total_sessions']}")
    print()
    
    if sessions["sessions"]:
        print("📝 会话列表:")
        for session in sessions["sessions"][-3:]:  # 只显示最近3个
            print(f"  🆔 ID: {session['session_id'][:8]}...")
            print(f"  ❓ 问题: {session['question']}")
            print(f"  📅 创建时间: {session['created_at']}")
            print(f"  📊 消息数量: {session['message_count']}")
            print(f"  ✅ 状态: {session['status']}")
            print()


def interactive_demo():
    """交互式演示"""
    api = StreamingAPI("demo_data")
    
    print_stream_header("交互式演示")
    print("💡 输入数学问题，观察流式响应过程")
    print("💡 输入 'quit' 退出，'history' 查看历史会话")
    print_stream_separator()
    
    while True:
        try:
            question = input("\n🤔 请输入你的问题: ").strip()
            
            if question.lower() == 'quit':
                print("👋 再见！")
                break
            elif question.lower() == 'history':
                demonstrate_session_management()
                continue
            elif not question:
                print("❌ 请输入有效问题")
                continue
            
            print("\n🔄 流式响应中...")
            print("-" * 40)
            
            # 处理流式响应
            for stream_data in api.ask_question_stream(question):
                try:
                    data = json.loads(stream_data)
                    if data["type"] == "stream":
                        msg_data = data["data"]
                        print(f"[{msg_data['message_type'].upper()}] {msg_data['content']}")
                except json.JSONDecodeError:
                    print(f"⚠️  解析错误: {stream_data}")
            
            print("-" * 40)
            print("✅ 响应完成")
        
        except KeyboardInterrupt:
            print("\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")


def main():
    """主函数"""
    print("🎉 流式响应ReAct Agent演示系统启动")
    print("=" * 60)
    
    while True:
        print("\n📋 演示选项:")
        print("1. 🚀 流式响应演示")
        print("2. 📱 前端格式演示") 
        print("3. 📚 会话管理演示")
        print("4. 💬 交互式体验")
        print("5. 🚪 退出")
        
        try:
            choice = input("\n👆 请选择演示类型 (1-5): ").strip()
            
            if choice == '1':
                demonstrate_streaming_response()
            elif choice == '2':
                demonstrate_frontend_format()
            elif choice == '3':
                demonstrate_session_management()
            elif choice == '4':
                interactive_demo()
            elif choice == '5':
                print("👋 谢谢使用，再见！")
                break
            else:
                print("❌ 无效选择，请输入 1-5")
        
        except KeyboardInterrupt:
            print("\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")


if __name__ == "__main__":
    main()