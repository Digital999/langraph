"""Milvus 向量数据库管理模块（替代 ChromaDB）"""
import os
from typing import List, Optional
from pathlib import Path

from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import settings
from utils.logger import logger


class MilvusVectorStoreManager:
    """Milvus 向量数据库管理器"""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        connection_args: Optional[dict] = None
    ):
        """初始化 Milvus 向量数据库
        
        Args:
            collection_name: 集合名称
            connection_args: Milvus 连接参数
        """
        # 从配置读取集合名称
        if collection_name is None:
            collection_name = settings.MILVUS_COLLECTION
        
        self.collection_name = collection_name
        
        # 从配置读取连接参数
        if connection_args is None:
            # 构建 Milvus 连接参数
            host = settings.MILVUS_HOST
            port = settings.MILVUS_PORT
            
            connection_args = {
                "host": host,
                "port": port,
            }
            
            # 如果配置了用户名和密码，添加认证信息
            if settings.MILVUS_USER:
                connection_args["user"] = settings.MILVUS_USER
            if settings.MILVUS_PASSWORD:
                connection_args["password"] = settings.MILVUS_PASSWORD
        
        self.connection_args = connection_args
        
        # 初始化 Embedding 模型
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        
        # 初始化向量数据库（延迟初始化）
        self._vector_store = None
        
        logger.info(f"Milvus 向量数据库已配置: {collection_name}")
    
    @property
    def vector_store(self):
        """延迟初始化向量数据库"""
        if self._vector_store is None:
            self._vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                auto_id=True,
            )
        return self._vector_store
    
    def load_documents(self, directory: str) -> List[Document]:
        """加载目录下的所有文档
        
        Args:
            directory: 文档目录路径
            
        Returns:
            文档列表
        """
        documents = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.warning(f"目录不存在: {directory}")
            return documents
        
        # 加载 txt 文件
        try:
            txt_loader = DirectoryLoader(
                directory,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents.extend(txt_loader.load())
        except Exception as e:
            logger.warning(f"加载 txt 文件失败: {e}")
        
        # 加载 md 文件
        try:
            md_loader = DirectoryLoader(
                directory,
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader
            )
            documents.extend(md_loader.load())
        except Exception as e:
            logger.warning(f"加载 md 文件失败: {e}")
        
        logger.info(f"从 {directory} 加载了 {len(documents)} 个文档")
        return documents
    
    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """切分文档
        
        Args:
            documents: 文档列表
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            切分后的文档块列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"文档切分完成: {len(documents)} 个文档 → {len(chunks)} 个块")
        return chunks
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量数据库
        
        Args:
            documents: 文档列表
        """
        if not documents:
            logger.warning("没有文档需要添加")
            return
        
        self.vector_store.add_documents(documents)
        logger.info(f"已添加 {len(documents)} 个文档到 Milvus")
    
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 过滤条件
            
        Returns:
            相关文档列表
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            expr=filter  # Milvus 使用 expr 而不是 filter
        )
        logger.debug(f"检索到 {len(results)} 个相关文档")
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        filter: Optional[dict] = None
    ) -> List[tuple[Document, float]]:
        """带相似度分数的搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 过滤条件
            
        Returns:
            (文档, 相似度分数) 列表
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            expr=filter
        )
        logger.debug(f"检索到 {len(results)} 个相关文档（带分数）")
        return results
    
    def delete_collection(self) -> None:
        """删除集合"""
        # Milvus 的删除方法
        try:
            self.vector_store.col.drop()
            logger.info(f"已删除集合: {self.collection_name}")
        except Exception as e:
            logger.warning(f"删除集合失败: {e}")
    
    def get_retriever(self, k: int = 3):
        """获取检索器
        
        Args:
            k: 返回结果数量
            
        Returns:
            检索器对象
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )


# 全局向量数据库实例
_milvus_store_manager: Optional[MilvusVectorStoreManager] = None


def get_milvus_store() -> MilvusVectorStoreManager:
    """获取全局 Milvus 实例"""
    global _milvus_store_manager
    if _milvus_store_manager is None:
        _milvus_store_manager = MilvusVectorStoreManager()
    return _milvus_store_manager


def init_milvus_knowledge_base(knowledge_dir: str = "./knowledge_base") -> None:
    """初始化 Milvus 知识库
    
    Args:
        knowledge_dir: 知识库目录
    """
    logger.info("开始初始化 Milvus 知识库...")
    
    vector_store = get_milvus_store()
    
    # 加载所有文档
    all_documents = []
    knowledge_path = Path(knowledge_dir)
    
    if not knowledge_path.exists():
        logger.warning(f"知识库目录不存在: {knowledge_dir}")
        return
    
    # 遍历子目录
    for subdir in knowledge_path.iterdir():
        if subdir.is_dir():
            logger.info(f"加载目录: {subdir.name}")
            docs = vector_store.load_documents(str(subdir))
            
            # 添加元数据（标记文档类型）
            for doc in docs:
                doc.metadata["category"] = subdir.name
            
            all_documents.extend(docs)
    
    if not all_documents:
        logger.warning("没有找到任何文档")
        return
    
    # 切分文档
    chunks = vector_store.split_documents(all_documents)
    
    # 添加到向量数据库
    vector_store.add_documents(chunks)
    
    logger.info(f"Milvus 知识库初始化完成！共处理 {len(chunks)} 个文档块")
