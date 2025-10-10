from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import os
import yaml

logger = logging.getLogger(__name__)

@dataclass
class AIPHAConfig:
    """Configuración central del sistema de conocimiento, cargada del config.yaml global"""
    
    PROJECT_ROOT: Path
    KNOWLEDGE_DB_PATH: Path
    LOGS_PATH: Path
    CHROMA_PERSIST_DIR: Path
    COLLECTION_NAME: str
    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSION: int
    LLM_PROVIDER: str
    LLM_MODEL: str
    AUTO_CAPTURE: bool
    CAPTURE_TYPES: List[str]
    API_KEY: Optional[str] = None  # Se lee de env var
    
    def __init__(self, global_config: Dict[str, Any]):
        km_config = global_config.get('knowledge_manager', {})
        
        self.PROJECT_ROOT = Path(km_config.get('project_root', "./"))
        self.KNOWLEDGE_DB_PATH = Path(km_config.get('knowledge_db_path', "./aipha_project/knowledge_base"))
        self.LOGS_PATH = Path(km_config.get('logs_path', "./aipha_project/logs"))
        self.CHROMA_PERSIST_DIR = Path(km_config.get('chroma_persist_dir', "./aipha_project/chroma_db"))
        self.COLLECTION_NAME = km_config.get('collection_name', "aipha_development")
        self.EMBEDDING_MODEL = km_config.get('embedding_model', "all-MiniLM-L6-v2")
        self.EMBEDDING_DIMENSION = km_config.get('embedding_dimension', 384)
        self.LLM_PROVIDER = km_config.get('llm_provider', "openai")
        self.LLM_MODEL = km_config.get('llm_model', "gpt-3.5-turbo")
        
        # API_KEY siempre se lee de la variable de entorno para seguridad
        self.API_KEY = os.getenv(km_config.get('api_key_env_var', "OPENAI_API_KEY"))
        
        self.AUTO_CAPTURE = km_config.get('auto_capture', True)
        self.CAPTURE_TYPES = km_config.get('capture_types', [
            "decision", "architecture", "implementation",
            "test", "bug_fix", "optimization", "documentation", "principle"
        ])
        
        # Crear directorios si no existen
        for path_attr in [self.PROJECT_ROOT, self.KNOWLEDGE_DB_PATH, 
                          self.LOGS_PATH, self.CHROMA_PERSIST_DIR]:
            path_attr.mkdir(parents=True, exist_ok=True)
        logger.info(f"AIPHAConfig cargada y directorios verificados.")

from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_query(self, input) -> List[List[float]]:
        if isinstance(input, str):
            embedding = self.model.encode([input])[0]
        elif isinstance(input, list):
            embedding = self.model.encode(input)[0]
        else:
            raise ValueError(f"Unsupported input type for embed_query: {type(input)}")
        emb_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        return [emb_list]

    def name(self) -> str:
        return f"sentence-transformers-{self.model_name}"

class VectorDBManager:
    def __init__(self, config: AIPHAConfig):
        self.config = config
        self.embedding_function = SentenceTransformerEmbeddingFunction(self.config.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(self.config.CHROMA_PERSIST_DIR))
        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Para similitud coseno
        )
        logger.info(f"VectorDBManager inicializado en {self.config.CHROMA_PERSIST_DIR}.")

    def add_documents(self, documents: List[str], ids: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def search(self, query: str, n_results: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        where = {"type": filter_type} if filter_type else None
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return [{"id": id, "content": doc, "metadata": meta} for id, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0])]

if __name__ == "__main__":
    # Test VectorDBManager
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    aipha_config = AIPHAConfig(config)
    db_manager = VectorDBManager(aipha_config)
    db_manager.add_documents(["Test document 1"], ["test1"], [{"type": "test"}])
    results = db_manager.search("Test query")
    print(results)
