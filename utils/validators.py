"""输入验证工具"""
import html
import re


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    清理用户输入,防止注入攻击
    
    Args:
        text: 输入文本
        max_length: 最大长度
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除首尾空白
    text = text.strip()
    
    # HTML转义
    text = html.escape(text)
    
    # 移除控制字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # 限制长度
    if len(text) > max_length:
        text = text[:max_length]
    
    return text
