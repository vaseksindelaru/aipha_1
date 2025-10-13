from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

from .knowledge_manager.manager import AIPHAConfig, VectorDBManager, CaptureSystem, LLMQuerySystem, DevelopmentStep

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
        self.llm_query_system = None
        if self.aipha_km_config.API_KEY:
            self.llm_query_system = LLMQuerySystem(self.aipha_km_config, self.db_manager)
        self.capture_system = CaptureSystem(self.aipha_km_config, self.db_manager)
        
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
        Captures a manual knowledge entry in the vector database.

        This method creates a structured knowledge entry and stores it in the vector database
        for later retrieval and semantic search. It's designed for manual knowledge capture
        by agents or users.

        Args:
            category (str): Category of the knowledge entry (e.g., "architecture", "agent_roles").
            title (str): Descriptive title of the entry.
            content (str): Main content or body of the knowledge entry.
            metadata (Optional[Dict[str, Any]]): Additional metadata dictionary. Defaults to None.
            version (str): Version string for the entry. Defaults to "1.1.0".

        Returns:
            str: Unique identifier (UUID) of the captured knowledge entry.

        Side effects:
            - Persists the knowledge entry to ChromaDB vector database.
            - Logs the successful capture with entry ID.

        Example:
            >>> entry_id = sentinel.add_knowledge_entry(
            ...     category="architecture",
            ...     title="Layer Architecture Overview",
            ...     content="Aipha uses a 5-layer architecture..."
            ... )
            >>> print(entry_id)  # e.g., "550e8400-e29b-41d4-a716-446655440000"
        """
        entry_id = str(uuid.uuid4())
        step = DevelopmentStep(
            id=entry_id,
            timestamp=datetime.now().isoformat(),
            type=category,
            title=title,
            content=content,
            metadata=metadata or {"version": version}
        )
        self.capture_system.capture_manual(step)
        logger.info(f"Knowledge entry captured: ID={entry_id}, Category={category}, Title='{title}'")
        return entry_id

    def get_knowledge_entries(self, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves knowledge entries from the vector database with optional filtering.

        Performs semantic search across stored knowledge entries. If category is specified,
        filters results to that category. Returns entries sorted by relevance.

        Args:
            category (Optional[str]): Category filter (e.g., "architecture", "agent_roles").
                If None, returns entries from all categories. Defaults to None.
            limit (int): Maximum number of entries to return. Defaults to 100.

        Returns:
            List[Dict[str, Any]]: List of knowledge entries, each containing:
                - 'id': Unique identifier of the entry
                - 'content': Full content of the entry
                - 'metadata': Associated metadata dictionary

        Side effects:
            - None (read-only operation).

        Example:
            >>> entries = sentinel.get_knowledge_entries(category="architecture", limit=5)
            >>> len(entries) <= 5
            >>> all('architecture' in entry['metadata'].get('type', '') for entry in entries)
        """
        results = self.db_manager.search(query=category or "", n_results=limit, filter_type=category)
        return results

    def add_code_example(self, title: str, code: str, explanation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Captures a code example with explanation as a specialized knowledge entry.

        This method is specifically designed for storing code snippets with their explanations,
        making them searchable for future reference by agents or developers.

        Args:
            title (str): Descriptive title for the code example.
            code (str): The actual code snippet as a string.
            explanation (str): Detailed explanation of what the code does and why.
            metadata (Optional[Dict[str, Any]]): Additional metadata like language, framework, etc.

        Returns:
            str: Unique identifier of the captured code example entry.

        Side effects:
            - Stores the code example in vector database under "code_example" category.
            - Logs the successful capture with entry details.

        Example:
            >>> example_id = sentinel.add_code_example(
            ...     title="Simple ATR Calculation",
            ...     code="atr = ta.atr(high, low, close, timeperiod=14)",
            ...     explanation="Calculates Average True Range using TA-Lib library"
            ... )
            >>> print(example_id)  # e.g., "550e8400-e29b-41d4-a716-446655440000"
        """
        content = f"Code: {code}\nExplanation: {explanation}"
        example_id = self.add_knowledge_entry(category="code_example", title=title, content=content, metadata=metadata)
        logger.info(f"Code example captured: ID={example_id}, Title='{title}'")
        return example_id

    def search_code_examples(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic search for code examples in the knowledge base.

        Searches through stored code examples using vector similarity to find relevant
        code snippets based on natural language queries.

        Args:
            query (str): Natural language search query describing desired code functionality.
            n_results (int): Maximum number of code examples to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of matching code examples, each containing:
                - 'id': Unique identifier
                - 'content': Code and explanation text
                - 'metadata': Associated metadata

        Side effects:
            - None (read-only operation).

        Example:
            >>> examples = sentinel.search_code_examples("ATR calculation", n_results=3)
            >>> len(examples) <= 3
            >>> any("atr" in example['content'].lower() for example in examples)
        """
        return self.db_manager.search(query, n_results=n_results, filter_type="code_example")

    def verify_knowledge_base_integrity(self) -> bool:
        """
        Performs comprehensive integrity checks on all knowledge base components.

        Validates vector database connectivity, collection integrity, embedding functionality,
        and JSON file persistence. Includes detailed error reporting for troubleshooting.

        Returns:
            bool: True if all integrity checks pass, False if any component fails.

        Side effects:
            - Logs detailed warnings for each failed check.
            - Logs success message when all checks pass.
            - May perform test operations on vector database.

        Example:
            >>> is_integrity_ok = sentinel.verify_knowledge_base_integrity()
            >>> assert is_integrity_ok, "Knowledge base integrity check failed"
        """
        integrity_checks = []

        try:
            # Check ChromaDB collection exists and is accessible
            collection_count = self.db_manager.collection.count()
            if collection_count < 0:  # count() returns -1 on error
                logger.error("ChromaDB collection inaccessible")
                integrity_checks.append(False)
            else:
                logger.info(f"ChromaDB collection accessible with {collection_count} documents")
                integrity_checks.append(True)

            # Test embedding functionality with a simple query
            if collection_count > 0:
                test_results = self.db_manager.search("test query", n_results=1)
                if test_results is not None:
                    logger.info("Vector search functionality verified")
                    integrity_checks.append(True)
                else:
                    logger.error("Vector search functionality failed")
                    integrity_checks.append(False)
            else:
                logger.info("Skipping search test - empty collection")
                integrity_checks.append(True)

        except Exception as e:
            logger.error(f"ChromaDB integrity check failed: {e}")
            integrity_checks.append(False)

        # Check JSON files exist and are readable
        try:
            if not self.global_state_file.exists():
                logger.error("Global state file missing")
                integrity_checks.append(False)
            else:
                state_data = self._load_json(self.global_state_file)
                if isinstance(state_data, dict):
                    logger.info("Global state file integrity verified")
                    integrity_checks.append(True)
                else:
                    logger.error("Global state file corrupted")
                    integrity_checks.append(False)

            if not self.action_history_file.exists():
                logger.error("Action history file missing")
                integrity_checks.append(False)
            else:
                history_data = self._load_json(self.action_history_file)
                if isinstance(history_data, list):
                    logger.info("Action history file integrity verified")
                    integrity_checks.append(True)
                else:
                    logger.error("Action history file corrupted")
                    integrity_checks.append(False)

        except Exception as e:
            logger.error(f"JSON file integrity check failed: {e}")
            integrity_checks.append(False)

        # Overall result
        all_passed = all(integrity_checks)
        if all_passed:
            logger.info("Knowledge base integrity verification completed successfully")
        else:
            logger.warning(f"Knowledge base integrity check failed: {len(integrity_checks) - sum(integrity_checks)}/{len(integrity_checks)} checks failed")

        return all_passed

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
        Updates the global system state with new information.

        The global state serves as a "bulletin board" where agents can post and retrieve
        system-wide information. This method safely merges new state data with existing state.

        Args:
            state_update (Dict[str, Any]): Dictionary of state updates to apply.

        Returns:
            bool: True if the update was successful, False otherwise.

        Side effects:
            - Modifies global_state.json file with updated state.
            - Adds timestamp to the state.
            - Logs the update operation with modified keys.

        Example:
            >>> success = sentinel.update_global_state({"current_task": "ATR_implementation"})
            >>> assert success, "Failed to update global state"
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
        Retrieves the current global system state.

        Returns the complete global state dictionary that agents use to read
        system-wide information from the "bulletin board".

        Returns:
            Dict[str, Any]: Complete global state dictionary.

        Side effects:
            - None (read-only operation).

        Example:
            >>> state = sentinel.get_global_state()
            >>> current_task = state.get("current_task")
        """
        return self._load_json(self.global_state_file)

    def record_action(self, action_description: str, agent: str = "ContextSentinel",
                      component: str = "context_sentinel", status: str = "success",
                      details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Records an action in the system's chronological action history.

        The action history serves as a "system diary" tracking all operations performed
        by agents and components. Each entry includes timestamp, agent, component, and status.

        Args:
            action_description (str): Human-readable description of the action performed.
            agent (str): Name of the agent performing the action. Defaults to "ContextSentinel".
            component (str): System component involved. Defaults to "context_sentinel".
            status (str): Action outcome status. Defaults to "success".
            details (Optional[Dict[str, Any]]): Additional structured details about the action.

        Returns:
            bool: True if the action was successfully recorded, False otherwise.

        Side effects:
            - Appends new action entry to action_history.json file.
            - Logs the action recording.

        Example:
            >>> success = sentinel.record_action(
            ...     "Generated ATR proposal",
            ...     agent="ChangeProposer",
            ...     component="trading_flow",
            ...     status="success",
            ...     details={"proposal_id": "pce-atr-001"}
            ... )
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
        """
        Retrieves the complete chronological action history.

        Returns the full list of all actions recorded by the system, useful for
        auditing, debugging, and understanding system behavior over time.

        Returns:
            List[Dict[str, Any]]: Complete list of action entries, each containing:
                - 'timestamp': ISO format timestamp
                - 'action': Description of the action
                - 'agent': Agent that performed the action
                - 'component': System component involved
                - 'status': Outcome status
                - 'details': Optional additional details

        Side effects:
            - None (read-only operation).

        Example:
            >>> history = sentinel.get_action_history()
            >>> recent_actions = [h for h in history if "ATR" in h['action']]
        """
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