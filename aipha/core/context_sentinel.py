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
        if not metadata:
            metadata = {"source": "context_sentinel"}
        step = DevelopmentStep(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            type=category,
            title=title,
            content=content,
            metadata=metadata
        )
        self.capture_system.capture_manual(step)
        return step.id

    def get_knowledge_entries(self, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        results = self.db_manager.search(query=category or "", n_results=limit, filter_type=category)
        return results  # Retorna lista de dicts con id, content, metadata

    # Métodos similares para add_code_example y search_code_examples
    def add_code_example(self, title: str, code: str, explanation: str, metadata: Optional[Dict[str, Any]] = None):
        content = f"Code: {code}\nExplanation: {explanation}"
        return self.add_knowledge_entry(category="code_example", title=title, content=content, metadata=metadata)

    def search_code_examples(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        return self.db_manager.search(query, n_results=n_results, filter_type="code_example")

    def verify_knowledge_base_integrity(self) -> bool:
        # Verificar ChromaDB
        stats = self.db_manager.collection.count() > 0  # Ejemplo simple; expande si necesitas
        # Verificar JSON/SQLite para global_state y action_history
        if not self.global_state_file.exists() or not self.action_history_file.exists():
            return False
        return stats  # Retorna True si todo OK

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