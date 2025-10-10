from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

from aipha.core.knowledge_manager.manager import AIPHAConfig, VectorDBManager, CaptureSystem, LLMQuerySystem, DevelopmentStep

logger = logging.getLogger(__name__)

# Dataclass para entradas de conocimiento (adaptada para vector DB)
@dataclass
class KnowledgeEntry:
    category: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    version: str = "1.1.0"

class ContextSentinel:
    def __init__(self, global_config: Dict[str, Any]):
        self.global_config = global_config  # Config global para rutas como storage_root
        self.storage_root = Path(self.global_config.get('system', {}).get('storage_root', './aipha_memory_storage'))
        
        # Mantener rutas para global_state y action_history (JSON/SQLite)
        self.global_state_file = self.storage_root / 'global_state.json'
        self.action_history_file = self.storage_root / 'action_history.json'
        
        # NUEVO: Inicializar Knowledge Manager
        self.aipha_km_config = AIPHAConfig(self.global_config)
        self.db_manager = VectorDBManager(self.aipha_km_config)
        self.capture_system = CaptureSystem(self.aipha_km_config, self.db_manager)
        self.llm_query_system = None
        if self.aipha_km_config.API_KEY:
            self.llm_query_system = LLMQuerySystem(self.aipha_km_config, self.db_manager)
        
        # Inicializar estado global y historial de acciones (mantener como JSON)
        self._initialize_global_state()
        self._initialize_action_history()
        
        logger.info(f"ContextSentinel inicializado con Knowledge Manager. Ruta raíz: {self.storage_root}.")

    def _initialize_global_state(self):
        if not self.global_state_file.exists():
            initial_state = {"system_initialized": False}
            with open(self.global_state_file, 'w') as f:
                json.dump(initial_state, f, indent=4)
        logger.info("Global state inicializado.")

    def _initialize_action_history(self):
        if not self.action_history_file.exists():
            with open(self.action_history_file, 'w') as f:
                json.dump([], f, indent=4)
        logger.info("Action history inicializado.")

    def add_knowledge_entry(self, category: str, title: str, content: str, metadata: Optional[Dict[str, Any]] = None, version: str = "1.1.0") -> str:
        """
        Adds a knowledge entry to the vector DB via Knowledge Manager.

        Args:
            category (str): Category of the entry (e.g., "architecture").
            title (str): Title of the entry.
            content (str): Main content of the entry.
            metadata (Optional[Dict[str, Any]]): Additional metadata. Defaults to None.
            version (str): Version of the entry. Defaults to "1.1.0".

        Returns:
            str: ID of the added entry.

        Side effects:
            - Adds document to ChromaDB collection.
            - Logs the addition.

        Example:
            >>> entry_id = sentinel.add_knowledge_entry("test", "Test Title", "Test Content")
            >>> print(entry_id)  # e.g., "uuid-string"
        """
        step = DevelopmentStep(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            type=category,
            title=title,
            content=content,
            metadata=metadata or {"version": version}
        )
        self.capture_system.capture_manual(step)
        return step.id

    def get_knowledge_entries(self, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves knowledge entries from the vector DB.

        Args:
            category (Optional[str]): Filter by category/type. Defaults to None (all).
            limit (int): Max results to return. Defaults to 100.

        Returns:
            List[Dict[str, Any]]: List of entries with id, content, metadata.

        Side effects:
            - None.

        Example:
            >>> entries = sentinel.get_knowledge_entries("test", 5)
            >>> len(entries) <= 5
        """
        results = self.db_manager.search(query=category or "", n_results=limit, filter_type=category)
        return results

    def add_code_example(self, title: str, code: str, explanation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a code example as a knowledge entry.

        Args:
            title (str): Title of the example.
            code (str): Code snippet.
            explanation (str): Explanation of the code.
            metadata (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            str: ID of the added entry.

        Side effects:
            - Adds to ChromaDB under "code_example" category.

        Example:
            >>> sentinel.add_code_example("Test Code", "print('Hello')", "Simple print.")
        """
        content = f"Code: {code}\nExplanation: {explanation}"
        return self.add_knowledge_entry(category="code_example", title=title, content=content, metadata=metadata)

    def search_code_examples(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for code examples using semantic search.

        Args:
            query (str): Search query.
            n_results (int): Max results. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: Matching code examples.

        Side effects:
            - None.

        Example:
            >>> results = sentinel.search_code_examples("print")
            >>> "print" in results[0]['content']
        """
        return self.db_manager.search(query, n_results=n_results, filter_type="code_example")

    def verify_knowledge_base_integrity(self) -> bool:
        """
        Verifies the integrity of the knowledge base (ChromaDB and JSON files).

        Returns:
            bool: True if all checks pass.

        Side effects:
            - Logs warnings on failure.

        Example:
            >>> sentinel.verify_knowledge_base_integrity()
            True
        """
        # Verificar ChromaDB
        if self.db_manager.collection.count() == 0:
            logger.warning("ChromaDB collection is empty.")
            return False

        # Verificar JSON files
        if not self.global_state_file.exists() or not self.action_history_file.exists():
            logger.warning("Global state or action history missing.")
            return False

        logger.info("Knowledge base integrity verified.")
        return True

    def _load_json(self, file_path: Path) -> Any:
        """
        Método auxiliar interno para cargar datos JSON de un archivo de forma segura.
        Maneja archivos no existentes, vacíos o errores de decodificación.
        """
        if not file_path.exists() or file_path.stat().st_size == 0:
            # Retorna una lista o un diccionario vacío por defecto según el nombre/contenido esperado.
            return [] if "history" in file_path.name or "version" in file_path.name else {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error de decodificación JSON en {file_path}: {e}. Retornando contenido por defecto.")
            return [] if "history" in file_path.name or "version" in file_path.name else {}

    def _save_json(self, file_path: Path, data: Any):
        """Método auxiliar interno para guardar datos JSON en un archivo de forma legible."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def update_global_state(self, state_update: Dict[str, Any]) -> bool:
        """
        Actualiza el estado global del sistema (global_state.json).
        Es el "tablón de anuncios" donde los agentes dejan y recogen información.
        Retorna True si la actualización es exitosa, False en caso contrario.
        """
        try:
            current_state = self._load_json(self.global_state_file) # Carga el estado actual.
            current_state.update(state_update) # Combina el estado actual con las nuevas actualizaciones.
            current_state["timestamp"] = datetime.now().isoformat() + 'Z' # Actualiza la marca de tiempo.

            self._save_json(self.global_state_file, current_state) # Guarda el estado global actualizado.

            logger.info(f"Estado global actualizado. Claves modificadas: {list(state_update.keys())}")
            return True

        except Exception as e:
            logger.error(f"Fallo al actualizar el estado global: {e}")
            return False

    def get_global_state(self) -> Dict[str, Any]:
        """
        Devuelve el estado global actual del sistema.
        Es cómo los agentes leen el "tablón de anuncios".
        """
        return self._load_json(self.global_state_file)

    def record_action(self, action_description: str, agent: str = "ContextSentinel",
                      component: str = "context_sentinel", status: str = "success",
                      details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Registra una acción en el historial de acciones del sistema (action_history.json).
        Es el "diario" cronológico de todo lo que hace el sistema.
        Retorna True si el registro es exitoso, False en caso contrario.
        """
        try:
            action = {
                "timestamp": datetime.now().isoformat() + 'Z',
                "action": action_description,
                "agent": agent,
                "component": component,
                "status": status
            }

            if details:
                action["details"] = details # Añadimos detalles adicionales si se proporcionan.

            history = self._load_json(self.action_history_file) # Carga el historial existente.
            history.append(action) # Añade la nueva acción.
            self._save_json(self.action_history_file, history) # Guarda el historial actualizado.

            logger.info(f"Acción registrada: {action_description}")
            return True

        except Exception as e:
            logger.error(f"Fallo al registrar la acción: {e}")
            return False

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Devuelve el historial completo de acciones registradas por ContextSentinel."""
        return self._load_json(self.action_history_file)

if __name__ == "__main__":
    import yaml
    from aipha.core.context_sentinel import ContextSentinel

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    sentinel = ContextSentinel(config)

    # Test add and get
    entry_id = sentinel.add_knowledge_entry("test", "Test Title", "Test Content")
    entries = sentinel.get_knowledge_entries("test")
    print(entries)