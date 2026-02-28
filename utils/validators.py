"""输入验证工具"""
import html
import re
from typing import Tuple


def validate_phone_number(phone: str) -> Tuple[bool, str]:
    """
    验证手机号码格式
    
    Args:
        phone: 手机号码字符串
        
    Returns:
        (是否有效, 错误消息)
    """
    if not phone:
        return False, "手机号码不能为空"
    
    # 移除空格和特殊字符
    phone = re.sub(r'[\s\-()]', '', phone)
    
    # 验证是否为11位数字
    if not re.match(r'^1[3-9]\d{9}$', phone):
        return False, "请提供有效的11位手机号码"
    
    return True, ""


def validate_id_card(id_card: str) -> Tuple[bool, str]:
    """
    验证身份证号码格式
    
    Args:
        id_card: 身份证号码字符串
        
    Returns:
        (是否有效, 错误消息)
    """
    if not id_card:
        return False, "身份证号码不能为空"
    
    # 移除空格
    id_card = id_card.strip()
    
    # 验证18位身份证
    if not re.match(r'^\d{17}[\dXx]$', id_card):
        return False, "请提供有效的18位身份证号码"
    
    return True, ""


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
