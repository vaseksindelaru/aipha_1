from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import os
import yaml

logger = logging.getLogger(__name__)

@dataclass
class AIPHAConfig:
    """Configuration class for the Knowledge Manager system, loaded from global config.yaml."""
    
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
        """
        Initializes AIPHAConfig from global config dict.

        Args:
            global_config (Dict[str, Any]): Global configuration dictionary.

        Side effects:
            - Sets paths and creates directories if they don't exist.

        Example:
            >>> config = AIPHAConfig({'knowledge_manager': {'project_root': './'}})
        """
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
import openai

class SentenceTransformerEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using SentenceTransformer.

    Provides embeddings for documents and queries.
    """
    def __init__(self, model_name: str):
        """
        Initializes the embedding function with a SentenceTransformer model.

        Args:
            model_name (str): Name of the SentenceTransformer model.

        Side effects:
            - Loads the model into memory.

        Example:
            >>> emb_func = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents.

        Args:
            input (List[str]): List of text documents.

        Returns:
            List[List[float]]: List of embeddings.

        Side effects:
            - None.

        Example:
            >>> embeddings = emb_func(["Hello world"])
        """
        embeddings = self.model.encode(input)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Alias for __call__.

        Args:
            texts (List[str]): List of text documents.

        Returns:
            List[List[float]]: List of embeddings.

        Side effects:
            - None.

        Example:
            >>> embeddings = emb_func.embed_documents(["Hello"])
        """
        return self(texts)

    def embed_query(self, input) -> List[List[float]]:
        """
        Embeds a single query.

        Args:
            input: Query text (str or list).

        Returns:
            List[List[float]]: Embedding wrapped in list.

        Side effects:
            - None.

        Example:
            >>> embedding = emb_func.embed_query("Hello")
        """
        if isinstance(input, str):
            embedding = self.model.encode([input])[0]
        elif isinstance(input, list):
            embedding = self.model.encode(input)[0]
        else:
            raise ValueError(f"Unsupported input type for embed_query: {type(input)}")
        emb_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        return [emb_list]

    def name(self) -> str:
        """
        Returns the name of the embedding function.

        Returns:
            str: Name string.

        Side effects:
            - None.

        Example:
            >>> name = emb_func.name()
        """
        return f"sentence-transformers-{self.model_name}"

class VectorDBManager:
    """
    Manages the vector database using ChromaDB for knowledge storage and retrieval.
    """
    def __init__(self, config: AIPHAConfig):
        """
        Initializes the Vector DB Manager with ChromaDB.

        Args:
            config (AIPHAConfig): Configuration object.

        Side effects:
            - Creates ChromaDB client and collection if not exists.

        Example:
            >>> db_manager = VectorDBManager(config)
        """
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
        """
        Adds documents to the ChromaDB collection.

        Args:
            documents (List[str]): List of document contents.
            ids (List[str]): Unique IDs for documents.
            metadatas (Optional[List[Dict[str, Any]]]): Metadata for each document.

        Side effects:
            - Persists to ChromaDB.

        Example:
            >>> db_manager.add_documents(["Test"], ["id1"])
        """
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def search(self, query: str, n_results: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Searches the vector DB for relevant documents.

        Args:
            query (str): Search query.
            n_results (int): Number of results to return.
            filter_type (Optional[str]): Filter by metadata type.

        Returns:
            List[Dict[str, Any]]: List of matching documents with id, content, metadata.

        Side effects:
            - None.

        Example:
            >>> results = db_manager.search("test", 5)
        """
        where = {"type": filter_type} if filter_type else None
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return [{"id": id, "content": doc, "metadata": meta} for id, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0])]

from dataclasses import asdict
from datetime import datetime
import uuid

@dataclass
class DevelopmentStep:
    id: str
    timestamp: str
    type: str
    title: str
    content: str
    metadata: Dict[str, Any]

class CaptureSystem:
    """
    Handles capturing and storing development steps in the vector DB.
    """
    def __init__(self, config: AIPHAConfig, db_manager: VectorDBManager):
        """
        Initializes the Capture System.

        Args:
            config (AIPHAConfig): Configuration object.
            db_manager (VectorDBManager): Vector DB manager instance.

        Side effects:
            - None.

        Example:
            >>> capture_system = CaptureSystem(config, db_manager)
        """
        self.config = config
        self.db_manager = db_manager
        logger.info("CaptureSystem inicializado.")

    def capture_manual(self, step: DevelopmentStep):
        """
        Captures a manual development step in the vector DB.

        Args:
            step (DevelopmentStep): The development step to capture.

        Side effects:
            - Adds document to vector DB.

        Example:
            >>> step = DevelopmentStep(...)
            >>> capture_system.capture_manual(step)
        """
        id = step.id
        content = f"Type: {step.type}\nTitle: {step.title}\nContent: {step.content}\nMetadata: {step.metadata}"
        metadata = {"type": step.type, **step.metadata}
        self.db_manager.add_documents([content], [id], [metadata])

    def capture_auto(self, code_snippet: str, context: str):
        """
        Captures an automatic development step based on code and context.

        Args:
            code_snippet (str): The code snippet.
            context (str): Context information.

        Side effects:
            - Adds document to vector DB if AUTO_CAPTURE is enabled.

        Example:
            >>> capture_system.capture_auto("print('Hello')", "Test context")
        """
        if self.config.AUTO_CAPTURE:
            # Lógica para inferir type, title, etc. usando LLM si es necesario
            step = DevelopmentStep(
                id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                type="auto_capture",
                title="Auto-captured code snippet",
                content=code_snippet,
                metadata={"context": context}
            )
            self.capture_manual(step)

class LLMQuerySystem:
    """
    Handles querying the LLM with context retrieved from the vector DB (RAG).
    """
    def __init__(self, config: AIPHAConfig, db_manager: VectorDBManager):
        """
        Initializes the LLM Query System.

        Args:
            config (AIPHAConfig): Configuration object.
            db_manager (VectorDBManager): Vector DB manager instance.

        Side effects:
            - Initializes OpenAI client.

        Example:
            >>> llm_system = LLMQuerySystem(config, db_manager)
        """
        self.config = config
        self.db_manager = db_manager
        if self.config.API_KEY:
            self.client = openai.OpenAI(api_key=self.config.API_KEY)
        else:
            self.client = None
        logger.info("LLMQuerySystem inicializado.")

    def query(self, user_query: str) -> str:
        """
        Queries the LLM with context retrieved from the vector DB.

        Args:
            user_query (str): The user's query.

        Returns:
            str: The LLM's response.

        Side effects:
            - Calls OpenAI API if API_KEY is set.

        Example:
            >>> response = llm_system.query("What is the test content?")
        """
        if not self.client:
            return "LLM not configured (no API_KEY)."
        # Recuperar contexto relevante
        relevant_docs = self.db_manager.search(user_query, n_results=3)
        context = "\n".join([doc['content'] for doc in relevant_docs])

        # Construir prompt
        prompt = f"Contexto:\n{context}\n\nPregunta: {user_query}\n\nRespuesta:"

        # Llamar a OpenAI
        response = self.client.chat.completions.create(
            model=self.config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
           

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    aipha_config = AIPHAConfig(config)
    db_manager = VectorDBManager(aipha_config)
    capture_system = CaptureSystem(aipha_config, db_manager)
    llm_query_system = LLMQuerySystem(aipha_config, db_manager)

    # Add entry
    step = DevelopmentStep(id=str(uuid.uuid4()), timestamp=datetime.now().isoformat(), type="test", title="Test", content="Test content", metadata={})
    capture_system.capture_manual(step)

    # Search
    results = db_manager.search("Test")
    print(results)

    # Query LLM
    query_result = llm_query_system.query("What is the test?")
    print(query_result)
