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
        """
        Inicializa el RedesignHelper.
        
        Args:
            config: Diccionario de configuración cargado desde config.yaml
        """
        self.config = config
        self.storage_root = Path(config['system']['storage_root'])
        
        # Componentes fundamentales de Capa 1
        self.critical_memory = CriticalMemoryRules(config)
        self.context = ContextSentinel(config)
        
        # Agentes especializados (placeholders para Fase 1)
        self.change_proposer = None
        self.proposal_evaluator = None
        self.codecraft_sage = None
        self.meta_improver = None
        
        self.initialized = False
        logger.info("RedesignHelper: Instanciado con storage_root=%s", self.storage_root)
    
    def initialize(self) -> bool:
        """
        Inicializa todos los componentes del sistema.
        
        Proceso:
        1. Verificar integridad de CriticalMemoryRules
        2. Verificar integridad de ContextSentinel
        3. Cargar criterios de evaluación por defecto
        4. Cargar conocimiento base del sistema
        
        Returns:
            True si la inicialización fue exitosa, False en caso contrario
        """
        try:
            logger.info("RedesignHelper: Iniciando sistema de memoria crítica...")
            
            # 1. Verificar integridad de CriticalMemoryRules
            if not self.critical_memory.verify_system_integrity():
                logger.error("RedesignHelper: Verificación de integridad de CriticalMemoryRules falló")
                return False
            
            # 2. Verificar integridad de ContextSentinel
            if not self.context.verify_knowledge_base_integrity():
                logger.error("RedesignHelper: Verificación de integridad de ContextSentinel falló")
                return False
            
            # 3. Cargar criterios de evaluación
            self.context.add_default_evaluation_criteria()
            logger.info("RedesignHelper: Criterios de evaluación cargados")
            
            # 4. Cargar conocimiento base
            self._load_base_knowledge()
            logger.info("RedesignHelper: Conocimiento base cargado")
            
            self.initialized = True
            logger.info("RedesignHelper: Sistema inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"RedesignHelper: Error en inicialización - {e}", exc_info=True)
            return False
    
    def _load_base_knowledge(self):
        """
        Carga el conocimiento fundamental de Aipha_1.1 en ContextSentinel.
        
        Este método puebla la base de conocimiento con:
        - Arquitectura del sistema
        - Roles de agentes
        - Flujos de protocolo
        - Principios de diseño
        - Planes de proyecto (ej. ATR)
        """
        logger.info("RedesignHelper: Añadiendo conocimiento base...")
        
        # 1. Arquitectura de Aipha_1.1
        self.context.add_knowledge_entry(
            category="architecture",
            title="Aipha_1.1 Layer Architecture",
            content="Aipha_1.1 se estructura en 5 capas principales: Núcleo (Capa 1: redesignHelper, ContextSentinel), Data System (Capa 2), Trading Flow (Capa 3), Oracles (Capa 4), y Data Postprocessor (Capa 5). El sistema actual se centra en el MVP de la Capa 1, con las demás capas planificadas para ser construidas por este núcleo.",
            metadata={"importance": "critical", "layer": "architecture", "version": "1.1.0"}
        )
        
        self.context.add_knowledge_entry(
            category="architecture",
            title="Aipha_1.1 Componentes del Núcleo (Capa 1)",
            content="La Capa 1 de Aipha_1.1 incluye: 1. redesignHelper (orquestador de agentes, actualmente la lógica de RedesignHelper), 2. AtomicUpdateSystem (gestiona CriticalMemoryRules), y 3. ContextSentinel (memoria persistente).",
            metadata={"importance": "critical", "layer": "core", "version": "1.1.0"}
        )
        
        # 2. Roles de agentes
        agent_roles = [
            {
                "name": "ChangeProposer",
                "description": "El ChangeProposer es el agente responsable de identificar y formular propuestas de mejora de alto nivel en un formato estructurado (YAML/JSON). Su tarea es traducir una directiva o un hallazgo del Data Postprocessor en una ChangeProposal concreta.",
                "component": "aipha/core/tools/change_proposer.py"
            },
            {
                "name": "ProposalEvaluator",
                "description": "El ProposalEvaluator evalúa y clasifica las propuestas de cambio según criterios objetivos (claridad, viabilidad, alineación arquitectónica). Decide si una propuesta es viable y, en el futuro, selecciona el LLM más adecuado para implementarla. Su decisión es crítica para la calidad de las mejoras.",
                "component": "aipha/core/tools/proposal_evaluator.py"
            },
            {
                "name": "CodecraftSage",
                "description": "El CodecraftSage es el agente implementador. Su rol es transformar una propuesta aprobada en código ejecutable. Genera o adapta código, escribe tests, y ejecuta un ciclo de 'generar -> probar -> corregir' hasta que el cambio sea funcional y cumpla los requisitos.",
                "component": "aipha/core/tools/codecraft_sage.py"
            },
            {
                "name": "MetaImprover",
                "description": "El MetaImprover gestiona la integración final de los cambios aprobados y verificados. Realiza acciones como la gestión de commits (en el futuro), la actualización del GlobalAgentState y el registro en el historial de versiones, consolidando así la evolución del sistema.",
                "component": "aipha/core/tools/meta_improver.py"
            },
            {
                "name": "Data Postprocessor",
                "description": "El Data Postprocessor (Capa 5) es el agente analítico. Su rol es analizar los resultados del trading flow, identificar regímenes de mercado y generar justificaciones automáticas para nuevas propuestas de mejora, cerrando el bucle de automejora.",
                "component": "aipha/data_postprocessor/"
            }
        ]
        
        for agent in agent_roles:
            self.context.add_knowledge_entry(
                category="agent_roles",
                title=f"Rol del {agent['name']}",
                content=agent["description"],
                metadata={"agent_name": agent["name"], "component": agent["component"], "version": "1.1.0"}
            )
        
        # 3. Flujos de protocolo
        protocol_flows = [
            {
                "title": "Flujo General del Bucle de Automejora",
                "content": "El bucle de automejora sigue fases de Propuesta (ChangeProposer), Evaluación (ProposalEvaluator), Implementación (CodecraftSage), e Integración (MetaImprover). Todo el proceso es orquestado, auditable y registrado.",
                "phase": "overview"
            },
            {
                "title": "Flujo de Propuesta de Cambio Detallado",
                "content": "Una propuesta se formaliza, se le asigna un ID único y una versión, y se registra como PENDING. Incluye descripción, justificación, archivos afectados y diff_content.",
                "phase": "proposal"
            },
            {
                "title": "Flujo de Evaluación de Propuesta Detallado",
                "content": "La propuesta PENDING se evalúa contra 'evaluation_criteria' (claridad, viabilidad, alineación). Si es aprobada, pasa a APPROVED; si es rechazada, se descarta. En el futuro, seleccionará el LLM más adecuado para la implementación.",
                "phase": "evaluation"
            },
            {
                "title": "Flujo de Aplicación Atómica de Cambio (CriticalMemoryRules)",
                "content": "CriticalMemoryRules aplica cambios aprobados mediante 5 pasos seguros: 1. Backup, 2. Aplicación real (simulada por ahora), 3. Actualización de Version_History, 4. Actualización de Current_Version, 5. Verificación de Integridad. Cualquier fallo detiene el proceso con registro de error.",
                "phase": "application"
            }
        ]
        
        for flow in protocol_flows:
            self.context.add_knowledge_entry(
                category="protocol_flow",
                title=flow["title"],
                content=flow["content"],
                metadata={"importance": "high", "phase": flow["phase"], "version": "1.1.0"}
            )
        
        # 4. Principios de diseño
        coding_principles = [
            {
                "title": "Principios de Modularidad en Aipha_1.1",
                "content": "El código debe ser modular, con responsabilidades claras por clase/función. Se prefiere la inyección de dependencias (pasar config, context_sentinel) en lugar de dependencias globales implícitas.",
                "importance": "high"
            },
            {
                "title": "Uso de Type Hints en Python",
                "content": "Todo el código de Aipha_1.1 debe utilizar Type Hints completos para mejorar la legibilidad, facilitar el análisis estático y prevenir errores en tiempo de desarrollo. (Ejemplo: def my_func(arg: str) -> bool:).",
                "importance": "medium"
            },
            {
                "title": "Principio de Seguridad 'Safety-First' en Aipha_1.1",
                "content": "Todo el diseño y las modificaciones del sistema Aipha_1.1 deben priorizar la seguridad, la auditabilidad y la capacidad de rollback. Ningún cambio puede poner en riesgo la integridad del sistema o su memoria persistente. Este es un principio fundamental y no negociable.",
                "importance": "critical"
            }
        ]
        
        for principle in coding_principles:
            self.context.add_knowledge_entry(
                category="coding_principles",
                title=principle["title"],
                content=principle["content"],
                metadata={"importance": principle["importance"], "type": "guideline", "version": "1.1.0"}
            )
        
        # 5. Planes de proyecto
        self.context.add_knowledge_entry(
            category="project_plan",
            title="ATR Dynamic Barriers Proposal Details",
            content="Propuesta para reemplazar TP/SL fijos en el PotentialCaptureEngine (Capa 3) con barreras dinámicas que se ajustan automáticamente en función de la volatilidad actual del mercado, utilizando el Average True Range (ATR). Parámetros clave: atr_period, tp_multiplier, sl_multiplier.",
            metadata={"proposal_id": "pce-atr-001", "component": "aipha/trading_flow/labelers/potential_capture_engine.py", "version": "1.1.0"}
        )
        
        logger.info("RedesignHelper: Conocimiento base completado")
    
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
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado completo del sistema.
        
        Returns:
            Diccionario con información de estado del sistema
        """
        return {
            "initialized": self.initialized,
            "current_version": self.critical_memory.get_current_version() if self.initialized else "N/A",
            "storage_root": str(self.storage_root),
            "knowledge_entries": len(self.context.get_knowledge_entries()) if self.initialized else 0,
            "action_history_size": len(self.context.get_action_history()) if self.initialized else 0
        }