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
from .tools.codecraft_sage import CodecraftSage, ImplementationResult
from .tools.basic_rules_proposer import BasicRulesChangeProposer
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
        self.change_proposer = ChangeProposer(self.config)  # Ejemplo de agente hardcodeado

        # Inicializar agente LLM-based solo si hay API key
        try:
            self.basic_rules_proposer = BasicRulesChangeProposer(self.config)  # Nuevo agente LLM-based
        except ValueError:
            logger.warning("BasicRulesChangeProposer no inicializado: OPENAI_API_KEY no configurada")
            self.basic_rules_proposer = None

        self.proposal_evaluator = ProposalEvaluator(self.config, self.context_sentinel)  # Nuevo agente
        self.codecraft_sage = CodecraftSage(self.config)  # Nuevo agente para implementación
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

        # Add code generation templates and testing patterns
        self.context_sentinel.add_knowledge_entry(
            category="code_templates",
            title="ATR Implementation Template",
            content="""Template for implementing ATR-based dynamic barriers in trading engines.

Key components:
1. ATR calculation using rolling mean of True Range
2. Dynamic TP/SL calculation using ATR multipliers
3. Position labeling logic with timeout handling
4. Parameter validation and error handling

Required imports: pandas, numpy
Required columns: high, low, close
Optional: volume for enhanced ATR calculation""",
            metadata={"language": "python", "framework": "pandas", "domain": "trading"}
        )

        self.context_sentinel.add_knowledge_entry(
            category="testing_patterns",
            title="Trading Engine Test Patterns",
            content="""Comprehensive testing patterns for trading engines:

1. Unit Tests:
   - Parameter validation (positive values, valid ranges)
   - ATR calculation accuracy
   - Edge cases (NaN values, empty data)

2. Integration Tests:
   - Take Profit scenarios (price hits TP level)
   - Stop Loss scenarios (price hits SL level)
   - Timeout scenarios (position held to time limit)
   - Multiple position handling

3. Property-based Tests:
   - Invariant: labels should be -1, 0, or 1
   - Invariant: no positions should exceed time limit
   - Invariant: TP/SL levels should be calculated correctly

4. Performance Tests:
   - Large dataset processing
   - Memory usage validation
   - Execution time benchmarks""",
            metadata={"framework": "pytest", "domain": "trading", "test_types": "unit,integration,property"}
        )

        self.context_sentinel.add_knowledge_entry(
            category="code_templates",
            title="Pandas Trading Engine Boilerplate",
            content="""Standard boilerplate for pandas-based trading engines:

class BaseTradingEngine:
    def __init__(self, **kwargs):
        self.params = kwargs
        self._validate_parameters()

    def _validate_parameters(self):
        '''Validate all parameters have correct types and ranges'''
        pass

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        '''Calculate required technical indicators'''
        pass

    def label_events(self, data: pd.DataFrame, events: pd.Series) -> pd.Series:
        '''Main labeling logic - to be implemented by subclasses'''
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, Any]:
        '''Return current parameter configuration'''
        return self.params.copy()

Required error handling:
- Validate input data structure
- Handle NaN/inf values
- Check for required columns
- Validate date ranges and sorting""",
            metadata={"language": "python", "framework": "pandas", "pattern": "template_method"}
        )

        # Añadir más entradas como "agent_roles", "protocol_flow", etc.
        # Nota: Ahora delega en CaptureSystem y VectorDBManager
        logger.info("Conocimiento base cargado en Knowledge Manager.")
    
    # Métodos placeholder para agentes (se implementarán en Fase 1)
    
    def propose_change(self, directive: str, use_llm: bool = False) -> Optional[ChangeProposal]:
        """
        Genera una propuesta de cambio basada en una directiva.

        Args:
            directive: Descripción de alto nivel del cambio deseado
            use_llm: Si True, usa el agente LLM-based; si False, usa el hardcodeado

        Returns:
            ChangeProposal si se generó exitosamente, None en caso contrario

        Raises:
            RuntimeError: Si el sistema no está inicializado
        """
        if not self.initialized:
            raise RuntimeError("RedesignHelper no está inicializado. Llama a initialize() primero.")

        if use_llm:
            if self.basic_rules_proposer is None:
                logger.error("LLM-based proposer no disponible: OPENAI_API_KEY no configurada")
                return None
            logger.info(f"Generando propuesta usando LLM para directiva: '{directive}'")
            return self.basic_rules_proposer.generate_proposal(directive, self.critical_memory_rules)
        else:
            logger.info(f"Generando propuesta hardcodeada para directiva: '{directive}'")
            return self.change_proposer.generate_proposal(directive)
    
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
        """Demuestra el flujo completo de propuesta ATR: generación + evaluación + implementación."""
        proposal = self.change_proposer.generate_proposal("ATR")
        logger.info(f"Propuesta ATR generada: {proposal}")

        # Nuevo: Evaluar la propuesta usando ProposalEvaluator
        evaluation = self.evaluate_proposal(proposal)
        logger.info(f"Evaluación ATR completada: {evaluation}")

        if evaluation["approved"]:
            logger.info("Propuesta ATR evaluada positivamente. Proceder a implementación.")

            # Nuevo: Implementar el cambio usando CodecraftSage
            implementation = self.codecraft_sage.implement_change(proposal)
            logger.info(f"Implementación ATR completada: {implementation.message}")

            if implementation.success:
                logger.info("Implementación ATR exitosa. Código y tests generados.")
                logger.info(f"Archivos a modificar: {implementation.files_modified}")
                logger.info(f"Tests a crear: {implementation.test_files_created}")

                # Aquí vendría apply_atomic_update en Fase 2
                # self.critical_memory_rules.apply_atomic_update(proposal)
                logger.info("Cambio listo para aplicación atómica (Fase 2).")
            else:
                logger.warning(f"Implementación ATR fallida: {implementation.message}")
        else:
            logger.info(f"Propuesta ATR no aprobada. Score: {evaluation['score']:.2f}")

    def run_interactive_change_cycle(self, directive: str, use_llm: bool = False):
        """
        Ejecuta un ciclo completo de propuesta-revisión-aplicación de forma interactiva.
        """
        print("\n--- [INICIO] Ciclo de Cambio Interactivo ---")

        # 1. Generar propuesta
        proposal = self.propose_change(directive, use_llm=use_llm)

        if not proposal:
            print("\n--- [FIN] No se pudo generar una propuesta. Finalizando ciclo. ---")
            return

        # 2. Mostrar propuesta para revisión humana
        print("\n--- Propuesta de Cambio Generada ---")
        print(f"ID del Cambio: {proposal.change_id}")
        print(f"Versión Propuesta: {proposal.version}")
        print(f"Autor: {proposal.author}")
        print("\n[Descripción]")
        print(proposal.description)
        print("\n[Justificación]")
        print(proposal.justification)
        print("\n[Archivos Afectados]")
        for file in proposal.files_affected:
            print(f"- {file}")
        print("\n[Contenido del Diff]")
        print("--------------------------------------")
        print(proposal.diff_content)
        print("--------------------------------------")

        # 3. Solicitar aprobación humana
        while True:
            action = input("\n¿Qué deseas hacer? [a]probar / [r]echazar / [c]ancelar: ").lower()
            if action in ['a', 'r', 'c']:
                break
            print("Opción no válida. Por favor, elige 'a', 'r', o 'c'.")

        # 4. Procesar la decisión
        if action == 'a':
            print("Aprobando propuesta...")
            if self.critical_memory_rules.approve_change(proposal, "HumanOperator"):
                print("Aplicando actualización atómica...")
                if self.critical_memory_rules.apply_atomic_update(proposal):
                    print(f"¡Éxito! El sistema ha sido actualizado a la versión {self.critical_memory_rules.get_current_version()}.")
                else:
                    print("¡ERROR CRÍTICO! Falló la aplicación de la actualización atómica.")
            else:
                print("ERROR: No se pudo aprobar la propuesta.")

        elif action == 'r':
            reason = input("Introduce una razón para el rechazo: ")
            print("Rechazando propuesta...")
            self.critical_memory_rules.reject_change(proposal, "HumanOperator", reason)
            print("La propuesta ha sido rechazada y registrada.")

        else: # action == 'c'
            print("Ciclo de cambio cancelado por el usuario.")

        print("\n--- [FIN] Ciclo de Cambio Interactivo ---")

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