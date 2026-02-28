"""流式响应服务"""
import asyncio
import json
import queue
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from agents.intent_agent import process_intent_stream
from graph import app as workflow_app
from models.schemas import AgentState
from utils.constants import (
    CONFIRM_KEYWORDS,
    MAX_CONVERSATION_TURNS,
    QUEUE_CHECK_INTERVAL,
    REJECT_KEYWORDS,
    STREAM_CHUNK_SIZE,
    STREAM_DELAY,
)
from utils.decorators import update_conversation_history
from utils.logger import logger


class StreamService:
    """流式响应服务"""
    
    def __init__(self):
        self.workflow_app = workflow_app
    
    async def send_sse_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> str:
        """
        发送SSE事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            
        Returns:
            格式化的SSE消息
        """
        data['type'] = event_type
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    async def stream_quick_response(
        self,
        response_text: str
    ) -> AsyncGenerator[str, None]:
        """
        流式输出快速响应(缓存的问候语)
        
        Args:
            response_text: 响应文本
            
        Yields:
            SSE消息
        """
        for i in range(0, len(response_text), STREAM_CHUNK_SIZE):
            chunk = response_text[i:i+STREAM_CHUNK_SIZE]
            yield await self.send_sse_event('content', {'content': chunk})
            await asyncio.sleep(STREAM_DELAY)
    
    def get_workflow_config(self, thread_id: str) -> Dict[str, Dict[str, str]]:
        """
        获取工作流配置
        
        Args:
            thread_id: 会话ID
            
        Returns:
            配置字典
        """
        return {"configurable": {"thread_id": thread_id}}
    
    async def get_conversation_history(
        self,
        config: Dict[str, Dict[str, str]]
    ) -> list:
        """
        获取对话历史
        
        Args:
            config: 工作流配置
            
        Returns:
            对话历史列表
        """
        try:
            current_state = self.workflow_app.get_state(config)
            if current_state and current_state.values:
                history = current_state.values.get("conversation_history", [])
                logger.debug(f"加载对话历史: {len(history)} 条")
                return history
        except Exception as e:
            logger.warning(f"获取对话历史失败: {e}")
        return []
    
    async def update_conversation_history_async(
        self,
        config: Dict[str, Dict[str, str]],
        user_input: str,
        assistant_response: str = ""
    ) -> None:
        """
        异步更新对话历史
        
        Args:
            config: 工作流配置
            user_input: 用户输入
            assistant_response: 助手回复
        """
        try:
            current_state = self.workflow_app.get_state(config)
            if current_state:
                conversation_history = current_state.values.get("conversation_history", [])
                conversation_history = update_conversation_history(
                    conversation_history,
                    user_input,
                    assistant_response,
                    max_turns=MAX_CONVERSATION_TURNS
                )
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.workflow_app.update_state(
                        config,
                        {"conversation_history": conversation_history}
                    )
                )
                logger.debug(f"对话历史已更新: {len(conversation_history)} 条")
        except Exception as e:
            logger.warning(f"更新对话历史失败: {e}")
    
    async def check_report_confirmation(
        self,
        config: Dict[str, Dict[str, str]],
        user_input: str
    ) -> Optional[str]:
        """
        检查是否在等待报告确认
        
        Args:
            config: 工作流配置
            user_input: 用户输入
            
        Returns:
            'confirm', 'reject', 'unclear' 或 None
        """
        try:
            current_state = self.workflow_app.get_state(config)
            if not (current_state and current_state.values.get("waiting_for_report_confirmation")):
                return None
            
            logger.info("检测到报告确认请求")
            user_input_lower = user_input.lower()
            
            is_confirm = any(keyword in user_input_lower for keyword in CONFIRM_KEYWORDS)
            is_reject = any(keyword in user_input_lower for keyword in REJECT_KEYWORDS)
            
            logger.debug(f"用户输入: {user_input_lower}, 确认={is_confirm}, 拒绝={is_reject}")
            
            if is_reject:
                return 'reject'
            elif is_confirm:
                return 'confirm'
            else:
                return 'unclear'
        except Exception as e:
            logger.warning(f"检查报告确认失败: {e}")
            return None
    
    async def handle_report_rejection(
        self,
        config: Dict[str, Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        处理用户拒绝生成报告
        
        Args:
            config: 工作流配置
            
        Yields:
            SSE消息
        """
        yield await self.send_sse_event('content', {
            'content': '好的，如果以后需要生成报告，请告诉我。'
        })
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.workflow_app.update_state(
                config,
                {"waiting_for_report_confirmation": False}
            )
        )
    
    async def handle_report_generation(
        self,
        config: Dict[str, Dict[str, str]],
        initial_input: Dict[str, str]
    ) -> AsyncGenerator[str, None]:
        """
        处理报告生成
        
        Args:
            config: 工作流配置
            initial_input: 初始输入
            
        Yields:
            SSE消息
        """
        yield await self.send_sse_event('content', {
            'content': '正在生成报告...'
        })
        
        loop = asyncio.get_event_loop()
        final_result = await loop.run_in_executor(
            None,
            lambda: self.workflow_app.invoke(initial_input, config)
        )
        
        if final_result.get("report_path"):
            import os
            filename = os.path.basename(final_result["report_path"])
            report_url = f"/api/download/{filename}"
            success_msg = f"\n\n✓ 报告生成成功！\n\n下载链接：{report_url}"
            yield await self.send_sse_event('content', {'content': success_msg})
            yield await self.send_sse_event('report', {
                'filename': filename,
                'url': report_url
            })
        else:
            error_msg = final_result.get("error", "报告生成失败")
            yield await self.send_sse_event('content', {
                'content': f'\n\n{error_msg}'
            })
    
    async def process_intent_with_stream(
        self,
        user_input: str,
        conversation_history: list
    ) -> AsyncGenerator[tuple[Optional[AgentState], str], None]:
        """
        处理意图理解并流式输出
        
        Args:
            user_input: 用户输入
            conversation_history: 对话历史
            
        Yields:
            (结果状态或None, SSE消息)
        """
        content_queue = queue.Queue()
        
        def sync_callback(content: str):
            content_queue.put(('content', content))
        
        temp_state: AgentState = {
            "user_input": user_input,
            "is_complete": False,
            "conversation_history": conversation_history,
        }
        
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            None,
            lambda: process_intent_stream(temp_state, sync_callback)
        )
        
        result_state = None
        while True:
            try:
                msg_type, content = content_queue.get_nowait()
                if msg_type == 'content':
                    yield (None, await self.send_sse_event('content', {'content': content}))
            except queue.Empty:
                if future.done():
                    result_state = future.result()
                    break
                await asyncio.sleep(QUEUE_CHECK_INTERVAL)
        
        # 输出队列中剩余的内容
        while not content_queue.empty():
            msg_type, content = content_queue.get_nowait()
            if msg_type == 'content':
                yield (None, await self.send_sse_event('content', {'content': content}))
        
        # 最后返回结果状态
        yield (result_state, "")
    
    async def execute_query_workflow(
        self,
        config: Dict[str, Dict[str, str]],
        initial_input: Dict[str, str],
        user_input: str
    ) -> AsyncGenerator[str, None]:
        """
        执行查询工作流
        
        Args:
            config: 工作流配置
            initial_input: 初始输入
            user_input: 用户输入
            
        Yields:
            SSE消息
        """
        yield await self.send_sse_event('content', {
            'content': '正在查询数据...'
        })
        
        loop = asyncio.get_event_loop()
        final_result = await loop.run_in_executor(
            None,
            lambda: self.workflow_app.invoke(initial_input, config)
        )
        
        # 输出性能统计
        perf_data = final_result.get("performance_metrics", {})
        yield await self.send_sse_event('performance', {'metrics': perf_data})
        logger.info(f"✓ 性能统计: {json.dumps(perf_data, ensure_ascii=False)}")
        
        # 输出查询结果
        if final_result.get("error"):
            result_content = final_result.get("error", "")
            yield await self.send_sse_event('content', {
                'content': '\n\n' + result_content
            })
            
            # 更新对话历史
            await self.update_conversation_history_async(
                config,
                user_input,
                "查询完成"
            )
        else:
            yield await self.send_sse_event('content', {
                'content': '\n\n查询完成，但未返回结果。'
            })


# 创建全局服务实例
stream_service = StreamService()
