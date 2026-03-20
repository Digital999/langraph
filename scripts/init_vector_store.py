"""初始化向量数据库脚本

用法：
  # 从 knowledge_base 目录加载 md/txt 文档
  python scripts/init_vector_store.py

  # 从 Excel 文件加载问答数据
  python scripts/init_vector_store.py --excel path/to/qa.xlsx

  # 清除旧数据后从 Excel 重新初始化
  python scripts/init_vector_store.py --excel path/to/qa.xlsx --clear
"""
import argparse
import asyncio
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag import init_knowledge_base, init_milvus_from_excel
from utils.logger import logger


async def main():
    parser = argparse.ArgumentParser(description="初始化向量数据库")
    parser.add_argument(
        "--excel", type=str, default=None,
        help="Excel 文件路径（第1列=分组，第2列=问题，第3列=相似问题，第4列=答案）",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="清除已有向量集合后重新初始化",
    )
    parser.add_argument(
        "--dir", type=str, default="./knowledge_base",
        help="知识库目录路径（默认 ./knowledge_base）",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("开始初始化向量数据库")
    logger.info("=" * 60)

    try:
        if args.excel:
            init_milvus_from_excel(
                excel_path=args.excel,
                clear_existing=args.clear,
            )
        else:
            if args.clear:
                from rag import get_vector_store
                get_vector_store().delete_collection()
                logger.info("已清除旧集合")
            init_knowledge_base(knowledge_dir=args.dir)

        logger.info("=" * 60)
        logger.info("向量数据库初始化完成！")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"初始化失败: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
