"""Milvus 向量数据库管理模块"""
from typing import List, Optional
from pathlib import Path

from langchain_milvus import Milvus
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import settings
from rag.volcengine_embeddings import VolcEngineMultimodalEmbeddings
from utils.logger import logger


class MilvusVectorStoreManager:
    """Milvus 向量数据库管理器"""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        connection_args: Optional[dict] = None
    ):
        if collection_name is None:
            collection_name = settings.MILVUS_COLLECTION
        
        self.collection_name = collection_name
        
        if connection_args is None:
            connection_args = {
                "host": settings.MILVUS_HOST,
                "port": settings.MILVUS_PORT,
            }
            if settings.MILVUS_USER:
                connection_args["user"] = settings.MILVUS_USER
            if settings.MILVUS_PASSWORD:
                connection_args["password"] = settings.MILVUS_PASSWORD
        
        self.connection_args = connection_args
        
        self.embeddings = VolcEngineMultimodalEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.effective_embedding_api_key,
        )
        
        self._vector_store: Optional[Milvus] = None
        self._available = True
        
        logger.info(
            f"Milvus 向量数据库已配置: collection={collection_name}, "
            f"embedding_model={settings.EMBEDDING_MODEL}"
        )
    
    @property
    def vector_store(self) -> Milvus:
        """延迟初始化向量数据库"""
        if self._vector_store is None:
            try:
                self._vector_store = Milvus(
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                    connection_args=self.connection_args,
                    auto_id=True,
                )
            except Exception as e:
                self._available = False
                logger.error(f"Milvus 连接失败: {e}")
                raise
        return self._vector_store
    
    @property
    def is_available(self) -> bool:
        """检查 Milvus 是否可用"""
        if not self._available:
            return False
        try:
            _ = self.vector_store
            return True
        except Exception:
            return False
    
    def load_documents(self, directory: str) -> List[Document]:
        """加载目录下的所有文档"""
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
        
        # 加载 md 文件（优先用 TextLoader 以避免 unstructured 依赖问题）
        try:
            md_loader = DirectoryLoader(
                directory,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
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
        """切分文档"""
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
        """添加文档到向量数据库"""
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
        """相似度搜索"""
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            expr=filter
        )
        logger.debug(f"检索到 {len(results)} 个相关文档")
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        score_threshold: Optional[float] = None,
        filter: Optional[dict] = None
    ) -> List[tuple[Document, float]]:
        """带相似度分数的搜索，支持分数阈值过滤
        
        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 分数阈值（L2 距离，越小越相关；设为 None 不过滤）
            filter: 过滤条件
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            expr=filter
        )
        
        if score_threshold is not None:
            before_count = len(results)
            results = [(doc, score) for doc, score in results if score <= score_threshold]
            if len(results) < before_count:
                logger.debug(
                    f"分数过滤: {before_count} → {len(results)} "
                    f"(阈值={score_threshold})"
                )
        
        logger.debug(f"检索到 {len(results)} 个相关文档（带分数）")
        return results
    
    def delete_collection(self) -> None:
        """删除集合"""
        try:
            self.vector_store.col.drop()
            self._vector_store = None
            logger.info(f"已删除集合: {self.collection_name}")
        except Exception as e:
            logger.warning(f"删除集合失败: {e}")
    
    def get_retriever(self, k: int = 3):
        """获取检索器"""
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )


_milvus_store_manager: Optional[MilvusVectorStoreManager] = None


def get_milvus_store() -> MilvusVectorStoreManager:
    """获取全局 Milvus 实例"""
    global _milvus_store_manager
    if _milvus_store_manager is None:
        _milvus_store_manager = MilvusVectorStoreManager()
    return _milvus_store_manager


def init_milvus_knowledge_base(knowledge_dir: str = "./knowledge_base") -> None:
    """初始化 Milvus 知识库"""
    logger.info("开始初始化 Milvus 知识库...")
    
    store = get_milvus_store()
    
    all_documents = []
    knowledge_path = Path(knowledge_dir)
    
    if not knowledge_path.exists():
        logger.warning(f"知识库目录不存在: {knowledge_dir}")
        return
    
    for subdir in knowledge_path.iterdir():
        if subdir.is_dir():
            logger.info(f"加载目录: {subdir.name}")
            docs = store.load_documents(str(subdir))
            for doc in docs:
                doc.metadata["category"] = subdir.name
            all_documents.extend(docs)
    
    if not all_documents:
        logger.warning("没有找到任何文档")
        return
    
    chunks = store.split_documents(all_documents)
    store.add_documents(chunks)
    
    logger.info(f"Milvus 知识库初始化完成！共处理 {len(chunks)} 个文档块")
