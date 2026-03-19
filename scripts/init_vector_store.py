"""初始化向量数据库脚本"""
import asyncio
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag import init_knowledge_base
from utils.logger import logger


async def main():
    logger.info("=" * 60)
    logger.info("开始初始化向量数据库")
    logger.info("=" * 60)
    
    try:
        init_knowledge_base(knowledge_dir="./knowledge_base")
        
        logger.info("=" * 60)
        logger.info("向量数据库初始化完成！")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
