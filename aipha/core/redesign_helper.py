# aipha/core/redesign_helper.py

"""
RedesignHelper - Orquestador principal del bucle de automejora de Aipha_1.1.
Este módulo coordina todos los agentes de la Capa 1.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from .atomic_update_system import CriticalMemoryRules, ChangeProposal, ApprovalStatus
from .context_sentinel import ContextSentinel
from .tools.change_proposer import ChangeProposer

logger = logging.getLogger(__name__)


class RedesignHelper:
    """
    Orquestador del ciclo de automejora de Aipha_1.1.
    
    Responsabilidades:
    - Coordinar entre CriticalMemoryRules y ContextSentinel
    - Gestionar el ciclo de vida de las propuestas de cambio
    - Cargar conocimiento base del sistema
    - Orquestar agentes (ChangeProposer, ProposalEvaluator, etc.)
    
    Attributes:
        config: Configuración del sistema cargada desde config.yaml
        storage_root: Ruta raíz del almacenamiento persistente
        critical_memory: Instancia de CriticalMemoryRules
        context: Instancia de ContextSentinel
        initialized: Flag de estado de inicialización
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.critical_memory_rules = CriticalMemoryRules(self.config)
        self.context_sentinel = ContextSentinel(self.config)  # Ahora usa Knowledge Manager internamente
        self.change_proposer = ChangeProposer(self.config)  # Ejemplo de agente

        self.initialize()
        logger.info("RedesignHelper inicializado con Knowledge Manager integrado.")
    
    def initialize(self):
        # Verificaciones de integridad
        if not self.critical_memory_rules.verify_system_integrity():
            raise ValueError("Integridad del sistema fallida.")
        if not self.context_sentinel.verify_knowledge_base_integrity():
            raise ValueError("Integridad de la base de conocimiento fallida.")

        # Cargar conocimiento inicial si es necesario
        self._load_base_knowledge()
    
    def _load_base_knowledge(self):
        # Ejemplo: Añadir conocimiento base usando el nuevo sistema
        self.context_sentinel.add_knowledge_entry(
            category="architecture",
            title="Aipha_1.1 Architecture Overview",
            content="Aipha_1.1 is structured in 5 layers: Core, Data System, Trading Flow, Oracles, Data Postprocessor."
        )
        # Añadir más entradas como "agent_roles", "protocol_flow", etc.
        # Nota: Ahora delega en CaptureSystem y VectorDBManager
        logger.info("Conocimiento base cargado en Knowledge Manager.")
    
    # Métodos placeholder para agentes (se implementarán en Fase 1)
    
    def propose_change(self, directive: str) -> Optional[ChangeProposal]:
        """
        Genera una propuesta de cambio basada en una directiva.
        
        Args:
            directive: Descripción de alto nivel del cambio deseado
        
        Returns:
            ChangeProposal si se generó exitosamente, None en caso contrario
        
        Raises:
            RuntimeError: Si el sistema no está inicializado
        """
        if not self.initialized:
            raise RuntimeError("RedesignHelper no está inicializado. Llama a initialize() primero.")
        
        # TODO: Implementar en Fase 1 con ChangeProposer real
        logger.warning("propose_change: No implementado aún (Fase 1)")
        return None
    
    def evaluate_proposal(self, proposal: ChangeProposal) -> Dict[str, Any]:
        """
        Evalúa una propuesta de cambio usando criterios objetivos.
        
        Args:
            proposal: La propuesta a evaluar
        
        Returns:
            Diccionario con resultado de evaluación
        
        Raises:
            RuntimeError: Si el sistema no está inicializado
        """
        if not self.initialized:
            raise RuntimeError("RedesignHelper no está inicializado. Llama a initialize() primero.")
        
        # TODO: Implementar en Fase 1 con ProposalEvaluator real
        logger.warning("evaluate_proposal: No implementado aún (Fase 1)")
        return {"approved": False, "reason": "Not implemented"}
    
    def demonstrate_atr_proposal_flow(self):
        proposal = self.change_proposer.generate_proposal("ATR Enhancement")  # Asume que generate_proposal usa LLM si aplica
        # ... (resto del flujo)
        logger.info("Demostración de flujo ATR completada.")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado completo del sistema.

        Returns:
            Diccionario con información de estado del sistema
        """
        return {
            "initialized": True,  # Since initialize is called in __init__
            "current_version": self.critical_memory_rules.get_current_version(),
            "storage_root": str(Path(self.config['system']['storage_root'])),
            "knowledge_entries": len(self.context_sentinel.get_knowledge_entries()),
            "action_history_size": len(self.context_sentinel.get_action_history())
        }