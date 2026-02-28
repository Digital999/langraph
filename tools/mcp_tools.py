"""MCP 工具集 - 数据库查询和第三方接口调用"""
import time
import traceback
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Annotated, Dict

import httpx
from langchain_core.tools import tool
from sqlalchemy import create_engine, text

from config import settings
from utils.logger import logger

# 数据库连接 - 配置连接池
DATABASE_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
engine = create_engine(
    DATABASE_URL,
    pool_size=10,              # 连接池大小
    max_overflow=20,           # 最大溢出连接数
    pool_timeout=30,           # 获取连接超时(秒)
    pool_recycle=3600,         # 连接回收时间(1小时)
    pool_pre_ping=True,        # 连接前检查是否有效
    echo=False                 # 生产环境关闭SQL日志
)

def serialize_value(value: Any) -> Any:
    """
    将数据库值转换为JSON兼容的类型
    
    Args:
        value: 数据库返回的值
        
    Returns:
        JSON兼容的值
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None:
        return None
    return value


def serialize_row(row_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    将数据库行转换为JSON兼容的字典
    
    Args:
        row_mapping: 数据库行映射
        
    Returns:
        JSON兼容的字典
    """
    return {key: serialize_value(value) for key, value in row_mapping.items()}

@tool
def query_user_by_phone(phone: Annotated[str, "11位手机号码"]) -> Dict[str, Any]:
    """
    根据电话号码查询用户基本信息。这是第一步必须调用的工具，用于获取用户的user_id等基本信息。
    
    Args:
        phone: 11位手机号码
        
    Returns:
        用户基本信息字典，包含id, phone, username字段，其中id即为user_id， 如果success为False则表示没有找到该用户
    """
    start_time = time.time()
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT id, phone, username FROM users WHERE phone = :phone"),
                {"phone": phone}
            )
            row = result.fetchone()
            elapsed_ms = (time.time() - start_time) * 1000
            
            if row:
                return {
                    "success": True, 
                    "data": serialize_row(dict(row._mapping)),
                    "_perf": {"query_user_by_phone": elapsed_ms}
                }
            return {
                "success": False, 
                "error": "未找到该手机号对应的用户信息",
                "_perf": {"query_user_by_phone": elapsed_ms}
            }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"query_user_by_phone 错误:\n{traceback.format_exc()}")
        return {
            "success": False, 
            "error": f"查询失败: {str(e)}",
            "_perf": {"query_user_by_phone": elapsed_ms}
        }

@tool
def query_user_package(user_id: Annotated[int, "用户ID，从query_user_by_phone获取"]) -> Dict[str, Any]:
    """
    查询用户套餐信息。必须先调用query_user_by_phone获取user_id后才能调用此工具。
    
    Args:
        user_id: 用户ID，从query_user_by_phone的返回结果中获取
        
    Returns:
        用户套餐信息字典，包含套餐名称、类型、有效期等
        比如：meal_status标识套餐状态。2为使用中，6为已退租
        start_date和end_date为套餐开始和结束时间
    """
    start_time = time.time()
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM user_meal WHERE user_id = :user_id order by id desc"),
                {"user_id": user_id}
            )
            row = result.fetchone()
            elapsed_ms = (time.time() - start_time) * 1000
            
            if row:
                return {
                    "success": True, 
                    "data": serialize_row(dict(row._mapping)),
                    "_perf": {"query_user_package": elapsed_ms}
                }
            return {
                "success": False, 
                "error": "未找到用户套餐信息",
                "_perf": {"query_user_package": elapsed_ms}
            }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"query_user_package 错误:\n{traceback.format_exc()}")
        return {
            "success": False, 
            "error": f"查询失败: {str(e)}",
            "_perf": {"query_user_package": elapsed_ms}
        }

@tool
def query_identity_info(
    name: Annotated[str, "姓名，从query_user_by_phone获取"], 
    id_card: Annotated[str, "身份证号，从query_user_by_phone获取"], 
    phone: Annotated[str, "手机号"]
) -> Dict[str, Any]:
    """
    验证身份三要素（姓名、身份证号、手机号）。调用第三方接口进行实名验证。
    必须先调用query_user_by_phone获取name和id_card后才能调用此工具。
    
    Args:
        name: 姓名
        id_card: 身份证号
        phone: 手机号
        
    Returns:
        身份验证结果，包含验证是否通过等信息
    """
    start_time = time.time()
    try:
        with httpx.Client() as client:
            response = client.get(
                settings.IDENTITY_API_URL,
                params={"name": name, "idcard": id_card, "mobile": phone},
                headers={"Authorization": f"APPCODE {settings.IDENTITY_API_KEY}"},
                timeout=5
            )
            response.raise_for_status()
            elapsed_ms = (time.time() - start_time) * 1000
            return {
                "success": True, 
                "data": response.json(),
                "_perf": {"query_identity_info": elapsed_ms}
            }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"query_identity_info 错误:\n{traceback.format_exc()}")
        return {
            "success": False, 
            "error": f"身份验证失败: {str(e)}",
            "_perf": {"query_identity_info": elapsed_ms}
        }

@tool
def query_realname_info(user_id: Annotated[int, "用户ID，从query_user_by_phone获取"]) -> Dict[str, Any]:
    """
    查询用户实名信息。必须先调用query_user_by_phone获取user_id后才能调用此工具。
    
    Args:
        user_id: 用户ID，从query_user_by_phone的返回结果中获取
        
    Returns:
        用户实名信息字典，包含真实姓名、身份证号等
    """
    start_time = time.time()
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT is_authentication FROM users WHERE id = :user_id"),
                {"user_id": user_id}
            )
            row = result.fetchone()
            elapsed_ms = (time.time() - start_time) * 1000
            
            if row:
                return {
                    "success": True, 
                    "data": serialize_row(dict(row._mapping)),
                    "_perf": {"query_realname_info": elapsed_ms}
                }
            return {
                "success": False, 
                "error": "未找到用户实名信息",
                "_perf": {"query_realname_info": elapsed_ms}
            }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"query_realname_info 错误:\n{traceback.format_exc()}")
        return {
            "success": False, 
            "error": f"查询失败: {str(e)}",
            "_perf": {"query_realname_info": elapsed_ms}
        }

# 导出所有工具
ALL_TOOLS = [
    query_user_by_phone,
    query_user_package,
    query_identity_info,
    query_realname_info
]
