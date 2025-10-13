from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import os
import uuid
from datetime import datetime

from sentence_transformers import SentenceTransformer
import chromadb

from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class AIPHAConfig:
    """Configuración central del sistema de conocimiento, cargada del config.yaml global.

    Esta clase carga y valida la configuración para el Knowledge Manager, creando directorios necesarios.

    Args:
        global_config (Dict[str, Any]): Diccionario de configuración global cargado de config.yaml.

    Side effects:
        - Crea directorios si no existen (PROJECT_ROOT, KNOWLEDGE_DB_PATH, etc.).
        - Logs la inicialización.

    Example:
        >>> with open('config.yaml', 'r') as f:
        >>>     config = yaml.safe_load(f)
        >>> aipha_config = AIPHAConfig(config)
        >>> print(aipha_config.PROJECT_ROOT)
        Path('./')
    """

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

@dataclass
class DevelopmentStep:
    """Representa un paso de desarrollo o entrada de conocimiento para capturar en la DB.

    Atributos:
        id (str): ID único del paso.
        timestamp (str): Timestamp ISO del paso.
        type (str): Tipo/categoría (e.g., "decision").
        title (str): Título descriptivo.
        content (str): Contenido principal.
        metadata (Dict[str, Any]): Metadatos adicionales.
    """
    id: str
    timestamp: str
    type: str
    title: str
    content: str
    metadata: Dict[str, Any]

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
    """Gestiona la base de datos vectorial con ChromaDB para embeddings y búsquedas semánticas.

    Args:
        config (AIPHAConfig): Configuración del Knowledge Manager.

    Side effects:
        - Inicializa cliente ChromaDB persistente.
        - Crea/carga colección con embedding function.

    Example:
        >>> db_manager = VectorDBManager(config)
        >>> db_manager.add_documents(["Test doc"], ["id1"])
        >>> results = db_manager.search("Test")
        >>> len(results) > 0
    """
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
        """Añade documentos a la colección ChromaDB.

        Args:
            documents (List[str]): Contenidos de documentos.
            ids (List[str]): IDs únicos.
            metadatas (Optional[List[Dict[str, Any]]]): Metadatos opcionales.

        Side effects:
            - Persiste en ChromaDB; genera embeddings automáticamente.

        Example:
            >>> db_manager.add_documents(["Doc1"], ["id1"], [{"type": "test"}])
        """
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def search(self, query: str, n_results: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Busca documentos semánticamente en la colección.

        Args:
            query (str): Consulta de búsqueda.
            n_results (int): Número máximo de resultados (default: 5).
            filter_type (Optional[str]): Filtrar por metadata 'type' (default: None).

        Returns:
            List[Dict[str, Any]]: Resultados con id, content, metadata.

        Side effects:
            - None.

        Example:
            >>> results = db_manager.search("test query", filter_type="test")
            >>> results[0]['content']  # "Matching doc"
        """
        where = {"type": filter_type} if filter_type else None
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return [{"id": id, "content": doc, "metadata": meta} for id, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0]) if results['ids']]  # Evita error si vacío

class CaptureSystem:
    """Sistema para capturar entradas de conocimiento manual/auto en la DB vectorial.

    Args:
        config (AIPHAConfig): Configuración.
        db_manager (VectorDBManager): Manager de DB.

    Side effects:
        - Logs capturas.

    Example:
        >>> step = DevelopmentStep(id="1", timestamp="now", type="test", title="Test", content="Content", metadata={})
        >>> capture_system.capture_manual(step)
    """
    def __init__(self, config: AIPHAConfig, db_manager: VectorDBManager):
        self.config = config
        self.db_manager = db_manager
        logger.info("CaptureSystem inicializado.")

    def capture_manual(self, step: DevelopmentStep):
        """Captura manual de un paso de desarrollo en la DB vectorial.

        Args:
            step (DevelopmentStep): Paso a capturar.

        Side effects:
            - Añade a ChromaDB.

        Example:
            >>> capture_system.capture_manual(step)
        """
        id = step.id
        content = f"Type: {step.type}\nTitle: {step.title}\nContent: {step.content}\nMetadata: {step.metadata}"
        self.db_manager.add_documents([content], [id], [step.metadata])
        logger.info(f"Capturado manual: ID {id}, Type {step.type}")

    def capture_auto(self, code_snippet: str, context: str):
        """Captura automática de un paso basado en código y contexto.

        Args:
            code_snippet (str): Snippet de código.
            context (str): Contexto de captura.

        Side effects:
            - Añade a ChromaDB si AUTO_CAPTURE=True.

        Example:
            >>> capture_system.capture_auto("print('Hello')", "Test context")
        """
        if self.config.AUTO_CAPTURE:
            step = DevelopmentStep(
                id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                type="auto_capture",
                title="Auto-captured code snippet",
                content=code_snippet,
                metadata={"context": context}
            )
            self.capture_manual(step)
            logger.info(f"Capturado auto: ID {step.id}")

class LLMQuerySystem:
    """Sistema para consultas RAG al LLM con contexto de la DB vectorial.

    Args:
        config (AIPHAConfig): Configuración.
        db_manager (VectorDBManager): Manager de DB.

    Side effects:
        - Logs queries/responses.

    Example:
        >>> result = llm_query_system.query("Test query", "test")
        >>> print(result)  # LLM response
    """
    def __init__(self, config: AIPHAConfig, db_manager: VectorDBManager):
        self.config = config
        self.db_manager = db_manager
        self.client = OpenAI(api_key=self.config.API_KEY)  # Usa nueva interfaz openai>=1.0
        logger.info("LLMQuerySystem inicializado.")

    def query(self, user_query: str, filter_type: Optional[str] = None, n_results: int = 5) -> str:
        """Consulta el LLM con contexto recuperado de la DB (RAG).

        Args:
            user_query (str): Consulta del usuario.
            filter_type (Optional[str]): Filtrar por type (default: None).
            n_results (int): Número de docs a recuperar (default: 5).

        Returns:
            str: Respuesta del LLM.

        Side effects:
            - Llama a API OpenAI.

        Example:
            >>> llm_query_system.query("What is test?")
            'Mock response'
        """
        retrieved_docs = self.db_manager.search(user_query, n_results, filter_type)
        context = "\n".join([doc['content'] for doc in retrieved_docs])

        prompt = f"Basado en el siguiente contexto: {context}\nResponde a la pregunta: {user_query}"

        response = self.client.chat.completions.create(
            model=self.config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "Eres un asistente útil."},
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0].message.content
        logger.info(f"LLM Query: {user_query} -> Response length: {len(result)}")
        return result
           

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
