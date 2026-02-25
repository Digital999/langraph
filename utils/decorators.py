"""公共装饰器和工具函数"""
import time
import traceback
from functools import wraps
from typing import Callable, Any
from utils.logger import logger


def handle_agent_error(error_message: str = "处理失败"):
    """
    Agent 错误处理装饰器
    
    Args:
        error_message: 默认错误消息
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            try:
                return func(state, *args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} 错误:\n{traceback.format_exc()}")
                error_msg = str(e)
                
                # 检查是否是 API 错误
                if "500" in error_msg or "InternalServerError" in error_msg:
                    state["error"] = "抱歉，服务暂时不可用，请稍后重试。"
                elif "timeout" in error_msg.lower():
                    state["error"] = "请求超时，请稍后重试。"
                else:
                    state["error"] = f"{error_message}: {error_msg}"
                
                state["next_step"] = "end"
                return state
        return wrapper
    return decorator


class PerformanceTimer:
    """性能计时器上下文管理器"""
    
    def __init__(self, name: str, metrics_dict: dict = None):
        """
        初始化计时器
        
        Args:
            name: 计时器名称
            metrics_dict: 用于存储性能指标的字典
        """
        self.name = name
        self.metrics_dict = metrics_dict
        self.start_time = None
        self.elapsed_ms = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        logger.debug(f"{self.name} 耗时: {self.elapsed_ms:.0f}ms")
        
        if self.metrics_dict is not None:
            self.metrics_dict[self.name] = self.elapsed_ms
        
        return False  # 不抑制异常


def update_conversation_history(history: list, user_input: str, assistant_response: str = "", max_turns: int = 10) -> list:
    """
    更新对话历史
    
    Args:
        history: 当前对话历史列表
        user_input: 用户输入
        assistant_response: 助手回复
        max_turns: 最大保留轮数
        
    Returns:
        更新后的对话历史
    """
    history.append(f"用户: {user_input}")
    if assistant_response:
        history.append(f"助手: {assistant_response}")
    
    # 只保留最近的对话
    max_messages = max_turns * 2
    if len(history) > max_messages:
        history = history[-max_messages:]
    
    return history


def build_history_string(history: list, max_turns: int = 3) -> str:
    """
    构建对话历史字符串
    
    Args:
        history: 对话历史列表
        max_turns: 最大显示轮数
        
    Returns:
        格式化的对话历史字符串
    """
    if not history:
        return "（无历史对话）"
    
    # 只取最近的对话
    recent_history = history[-(max_turns * 2):]
    return "\n".join(recent_history)
