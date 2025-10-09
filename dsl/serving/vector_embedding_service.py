"""
Task 6.3.55: Vector embeddings inference (pgvector/FAISS)
Vector embeddings service for RAG and semantic search
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_id: str
    embedding_dimension: int
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product
    index_type: str = "flat"  # flat, ivf, hnsw

@dataclass
class VectorDocument:
    """Document with vector embedding"""
    doc_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: datetime

class VectorEmbeddingService:
    """
    Vector embeddings service for RAG/semantic operations
    Task 6.3.55: RAG/semantic with index lifecycle
    """
    
    def __init__(self):
        self.embedding_configs: Dict[str, EmbeddingConfig] = {}
        self.vector_indexes: Dict[str, List[VectorDocument]] = {}  # Simple in-memory storage
        self.embedding_cache: Dict[str, List[float]] = {}
    
    def register_embedding_model(self, config: EmbeddingConfig) -> bool:
        """Register embedding model configuration"""
        try:
            self.embedding_configs[config.model_id] = config
            self.vector_indexes[config.model_id] = []
            
            logger.info(f"Registered embedding model: {config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register embedding model: {e}")
            return False
    
    async def generate_embedding(self, model_id: str, text: str) -> List[float]:
        """Generate embedding for text"""
        if model_id not in self.embedding_configs:
            raise ValueError(f"Unknown embedding model: {model_id}")
        
        # Check cache first
        cache_key = f"{model_id}:{hash(text)}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        config = self.embedding_configs[model_id]
        
        # Simulate embedding generation
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Generate random embedding (in production, would use actual model)
        embedding = np.random.normal(0, 1, config.embedding_dimension).tolist()
        
        # Normalize for cosine similarity
        if config.similarity_metric == "cosine":
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
        
        # Cache the embedding
        self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    async def add_document(self, model_id: str, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to vector index"""
        try:
            if model_id not in self.embedding_configs:
                raise ValueError(f"Unknown embedding model: {model_id}")
            
            # Generate embedding
            embedding = await self.generate_embedding(model_id, content)
            
            # Create document
            document = VectorDocument(
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            
            # Add to index
            self.vector_indexes[model_id].append(document)
            
            logger.info(f"Added document {doc_id} to index {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    async def semantic_search(self, model_id: str, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        if model_id not in self.embedding_configs:
            raise ValueError(f"Unknown embedding model: {model_id}")
        
        # Generate query embedding
        query_embedding = await self.generate_embedding(model_id, query)
        
        # Search in index
        results = []
        documents = self.vector_indexes[model_id]
        
        for doc in documents:
            similarity = self._calculate_similarity(
                query_embedding,
                doc.embedding,
                self.embedding_configs[model_id].similarity_metric
            )
            
            if similarity >= threshold:
                results.append({
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "similarity": similarity,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat()
                })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    async def similarity_search(self, model_id: str, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search by embedding vector"""
        if model_id not in self.embedding_configs:
            raise ValueError(f"Unknown embedding model: {model_id}")
        
        results = []
        documents = self.vector_indexes[model_id]
        
        for doc in documents:
            similarity = self._calculate_similarity(
                embedding,
                doc.embedding,
                self.embedding_configs[model_id].similarity_metric
            )
            
            results.append({
                "doc_id": doc.doc_id,
                "content": doc.content,
                "similarity": similarity,
                "metadata": doc.metadata
            })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float], metric: str) -> float:
        """Calculate similarity between two vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_product == 0:
                return 0.0
            return dot_product / norm_product
        
        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(v1 - v2)
            return 1.0 / (1.0 + distance)
        
        elif metric == "dot_product":
            # Dot product similarity
            return np.dot(v1, v2)
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    async def batch_add_documents(self, model_id: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add multiple documents in batch"""
        if model_id not in self.embedding_configs:
            raise ValueError(f"Unknown embedding model: {model_id}")
        
        results = {"added": 0, "failed": 0, "errors": []}
        
        for doc_data in documents:
            try:
                success = await self.add_document(
                    model_id,
                    doc_data["doc_id"],
                    doc_data["content"],
                    doc_data.get("metadata", {})
                )
                
                if success:
                    results["added"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Document {doc_data.get('doc_id', 'unknown')}: {str(e)}")
        
        return results
    
    def get_index_stats(self, model_id: str) -> Dict[str, Any]:
        """Get statistics for vector index"""
        if model_id not in self.embedding_configs:
            return {}
        
        documents = self.vector_indexes[model_id]
        config = self.embedding_configs[model_id]
        
        return {
            "model_id": model_id,
            "total_documents": len(documents),
            "embedding_dimension": config.embedding_dimension,
            "similarity_metric": config.similarity_metric,
            "index_type": config.index_type,
            "cache_size": len([k for k in self.embedding_cache.keys() if k.startswith(f"{model_id}:")]),
            "last_updated": max([d.created_at for d in documents]).isoformat() if documents else None
        }
    
    async def delete_document(self, model_id: str, doc_id: str) -> bool:
        """Delete document from index"""
        if model_id not in self.vector_indexes:
            return False
        
        documents = self.vector_indexes[model_id]
        original_length = len(documents)
        
        self.vector_indexes[model_id] = [d for d in documents if d.doc_id != doc_id]
        
        deleted = len(self.vector_indexes[model_id]) < original_length
        
        if deleted:
            logger.info(f"Deleted document {doc_id} from index {model_id}")
        
        return deleted
    
    async def rebuild_index(self, model_id: str) -> bool:
        """Rebuild vector index (for optimization)"""
        if model_id not in self.embedding_configs:
            return False
        
        # In a real implementation, this would rebuild the index structure
        # For now, just log the operation
        document_count = len(self.vector_indexes[model_id])
        
        logger.info(f"Rebuilding index for model {model_id} with {document_count} documents")
        
        # Simulate rebuild time
        await asyncio.sleep(0.1 * document_count / 100)  # Scale with document count
        
        return True

# Global vector embedding service
vector_embedding_service = VectorEmbeddingService()
