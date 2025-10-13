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
from .tools.proposal_evaluator import ProposalEvaluator, EvaluationResult
from .knowledge_manager.manager import DevelopmentStep

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
        self.proposal_evaluator = ProposalEvaluator(self.config, self.context_sentinel)  # Nuevo agente
        self.initialized = False

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
        self.initialized = True
        return True
    
    def _load_base_knowledge(self):
        # Ejemplo: Añadir conocimiento base usando el nuevo sistema
        self.context_sentinel.add_knowledge_entry(
            category="architecture",
            title="Aipha_1.1 Architecture Overview",
            content="Aipha_1.1 is structured in 5 layers: Core, Data System, Trading Flow, Oracles, Data Postprocessor."
        )

        # Add evaluation criteria for ProposalEvaluator
        self.context_sentinel.add_knowledge_entry(
            category="evaluation_criteria",
            title="Proposal Evaluation Criteria - Feasibility",
            content="Feasibility assessment considers: technical complexity, resource requirements, implementation timeline, dependencies, and team expertise. Score 0.8-1.0 for low complexity changes, 0.4-0.7 for moderate, 0.0-0.3 for high complexity requiring new technologies."
        )

        self.context_sentinel.add_knowledge_entry(
            category="evaluation_criteria",
            title="Proposal Evaluation Criteria - Impact",
            content="Impact evaluation measures: business value, user experience improvement, system performance gains, risk reduction, and scalability benefits. High impact (0.8-1.0) for changes affecting core functionality, moderate (0.4-0.7) for feature enhancements, low (0.0-0.3) for minor improvements."
        )

        self.context_sentinel.add_knowledge_entry(
            category="evaluation_criteria",
            title="Proposal Evaluation Criteria - Risk",
            content="Risk assessment evaluates: potential for bugs, system instability, security vulnerabilities, rollback difficulty, and operational impact. Low risk (0.0-0.3) for isolated changes with good testing, moderate (0.4-0.7) for changes affecting shared components, high risk (0.8-1.0) for core system modifications."
        )

        self.context_sentinel.add_knowledge_entry(
            category="evaluation_criteria",
            title="Proposal Evaluation Criteria - Priority Weighting",
            content="Priority levels determine evaluation thresholds: Critical (>0.8 overall score required), High (>0.7), Medium (>0.6), Low (>0.5). Critical proposals require additional security review. High priority proposals should demonstrate clear business justification."
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
        Evalúa una propuesta de cambio usando criterios objetivos con RAG.

        Utiliza el ProposalEvaluator para evaluar la propuesta basada en criterios
        de conocimiento almacenados y análisis LLM contextual.

        Args:
            proposal: La propuesta a evaluar

        Returns:
            Diccionario con resultado de evaluación incluyendo scores detallados

        Raises:
            RuntimeError: Si el sistema no está inicializado
        """
        if not self.initialized:
            raise RuntimeError("RedesignHelper no está inicializado. Llama a initialize() primero.")

        try:
            evaluation_result = self.proposal_evaluator.evaluate_proposal(proposal)

            result = {
                "approved": evaluation_result.is_approved(),
                "score": evaluation_result.score,
                "feasibility": evaluation_result.feasibility,
                "impact": evaluation_result.impact,
                "risk": evaluation_result.risk,
                "justification": evaluation_result.justification,
                "criteria_used": evaluation_result.criteria_used,
                "proposal_id": proposal.id
            }

            logger.info(f"Proposal {proposal.id} evaluation completed: score={evaluation_result.score:.2f}, "
                       f"approved={result['approved']}")

            return result

        except Exception as e:
            logger.error(f"Error evaluating proposal {proposal.id}: {e}")
            return {
                "approved": False,
                "score": 0.0,
                "reason": f"Evaluation failed: {str(e)}",
                "proposal_id": proposal.id
            }
    
    def demonstrate_atr_proposal_flow(self):
        """Demuestra el flujo completo de propuesta ATR: generación + evaluación."""
        proposal = self.change_proposer.generate_proposal("ATR")
        logger.info(f"Propuesta ATR generada: {proposal}")

        # Nuevo: Evaluar la propuesta usando ProposalEvaluator
        evaluation = self.evaluate_proposal(proposal)
        logger.info(f"Evaluación ATR completada: {evaluation}")

        if evaluation["approved"]:
            logger.info("Propuesta ATR evaluada positivamente. Proceder a aprobación.")
            # Aquí vendría approve_change en subpasos futuros
        else:
            logger.info(f"Propuesta ATR no aprobada. Score: {evaluation['score']:.2f}")

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