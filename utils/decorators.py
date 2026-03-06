"""公共装饰器和工具函数"""
import time
from typing import Dict, List

from utils.logger import logger


class PerformanceTimer:
    """性能计时器上下文管理器"""
    
    def __init__(self, name: str, metrics_dict: Dict[str, float] | None = None):
        """
        初始化计时器
        
        Args:
            name: 计时器名称
            metrics_dict: 用于存储性能指标的字典
        """
        self.name = name
        self.metrics_dict = metrics_dict
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        logger.debug(f"{self.name} 耗时: {self.elapsed_ms:.0f}ms")
        
        if self.metrics_dict is not None:
            self.metrics_dict[self.name] = self.elapsed_ms
        
        return False


def build_history_string(history: List[str], max_turns: int = 3) -> str:
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
