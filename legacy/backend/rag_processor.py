"""历史 RAG 实现，仅供迁移对照；当前实现位于 src/deep_sea_explorer/。"""

import os
import logging
import fitz
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """初始化RAG处理器"""
        try:
            self.embedding_model = SentenceTransformer('../local_models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf')
            self.documents = []
            self.embeddings = None
            self.index = None
            self.chunk_size = 500
            self.overlap = 50
            logger.info("RAG处理器初始化成功")
        except Exception as e:
            logger.error(f"RAG初始化失败: {e}")
            self.embedding_model = None
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF提取文本"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PDF文本提取失败: {e}")
            return ""
    
    def chunk_text(self, text: str) -> list:
        """将文本分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += self.chunk_size - self.overlap
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]
    
    def add_document(self, content: str, doc_id: str = None):
        """添加文档到知识库"""
        chunks = self.chunk_text(content)
        for i, chunk in enumerate(chunks):
            self.documents.append({
                'content': chunk,
                'doc_id': doc_id or f"doc_{len(self.documents)}",
                'chunk_id': i
            })
    
    def add_pdf(self, pdf_path: str, doc_id: str = None):
        """添加PDF文档"""
        text = self.extract_text_from_pdf(pdf_path)
        if text:
            self.add_document(text, doc_id or os.path.basename(pdf_path))
            return True
        return False
    
    def build_index(self):
        """构建向量索引"""
        if not self.documents or not self.embedding_model:
            logger.warning("没有文档或模型未加载，无法建立索引")
            return
        
        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.embedding_model.encode(texts)
        
        # 构建FAISS索引
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # 归一化向量
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"索引构建完成，包含 {len(self.documents)} 个文档块")
    
    def search(self, query: str, top_k: int = 3) -> list:
        """搜索相关文档"""
        if not self.index or not self.embedding_model:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
    def get_context(self, query: str, max_length: int = 1000) -> str:
        """获取查询相关的上下文"""
        results = self.search(query, top_k=5)
        
        context_parts = []
        total_length = 0
        
        for result in results:
            content = result['content']
            if total_length + len(content) <= max_length:
                context_parts.append(content)
                total_length += len(content)
            else:
                remaining = max_length - total_length
                if remaining > 100:
                    context_parts.append(content[:remaining])
                break
        
        return "\n\n".join(context_parts)
