"""配置文件"""
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4")
    
    # 数据库配置
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_USER: str = os.getenv("DB_USER", "root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_NAME: str = os.getenv("DB_NAME", "user_db")
    
    # 第三方接口配置
    IDENTITY_API_URL: str = os.getenv("IDENTITY_API_URL", "")
    IDENTITY_API_KEY: str = os.getenv("IDENTITY_API_KEY", "")
    
    class Config:
        env_file = ".env"

settings = Settings()
