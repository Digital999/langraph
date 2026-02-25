"""日志工具"""
import sys
from datetime import datetime
from typing import Any

class Logger:
    """简单的日志工具"""
    
    @staticmethod
    def _format_time():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    @staticmethod
    def info(message: str, **kwargs):
        """信息日志"""
        timestamp = Logger._format_time()
        extra = f" | {kwargs}" if kwargs else ""
        print(f"[{timestamp}] [INFO] {message}{extra}", flush=True)
    
    @staticmethod
    def debug(message: str, **kwargs):
        """调试日志"""
        timestamp = Logger._format_time()
        extra = f" | {kwargs}" if kwargs else ""
        print(f"[{timestamp}] [DEBUG] {message}{extra}", flush=True)
    
    @staticmethod
    def error(message: str, **kwargs):
        """错误日志"""
        timestamp = Logger._format_time()
        extra = f" | {kwargs}" if kwargs else ""
        print(f"[{timestamp}] [ERROR] {message}{extra}", file=sys.stderr, flush=True)
    
    @staticmethod
    def warning(message: str, **kwargs):
        """警告日志"""
        timestamp = Logger._format_time()
        extra = f" | {kwargs}" if kwargs else ""
        print(f"[{timestamp}] [WARNING] {message}{extra}", flush=True)
    
    @staticmethod
    def llm_input(prompt: str, max_length: int = 200):
        """LLM 输入日志"""
        timestamp = Logger._format_time()
        truncated = prompt[:max_length] + "..." if len(prompt) > max_length else prompt
        print(f"[{timestamp}] [LLM-INPUT] {truncated}", flush=True)
    
    @staticmethod
    def llm_output(content: str, max_length: int = 500):
        """LLM 输出日志"""
        timestamp = Logger._format_time()
        truncated = content[:max_length] + "..." if len(content) > max_length else content
        print(f"[{timestamp}] [LLM-OUTPUT] {truncated}", flush=True)
    
    @staticmethod
    def llm_stream(content: str):
        """LLM 流式输出日志"""
        print(content, end="", flush=True)
    
    @staticmethod
    def separator(char: str = "=", length: int = 80):
        """分隔线"""
        print(char * length, flush=True)

# 全局日志实例
logger = Logger()
