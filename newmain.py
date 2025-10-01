# main.py

import sys
import os
from datetime import datetime
from pathlib import Path
import logging
import json
import yaml
import shutil
from typing import Dict, Any, List, Optional

# --- Configuración del sistema de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
logger = logging.getLogger(__name__)

# --- Importaciones de módulos locales ---
try:
    from atomic_update_system import CriticalMemoryRules, ChangeProposal, ApprovalStatus
    from context_sentinel import ContextSentinel
except ImportError as e:
    logger.error(f"Error importing system modules: {e}")
    logger.error("Asegúrese de que 'atomic_update_system.py' y 'context_sentinel.py' estén en el mismo directorio que 'main.py'.")
    sys.exit(1)

# --- Función para cargar la configuración desde un archivo YAML ---
def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo YAML.
    Si el archivo no existe, crea uno por defecto con la estructura aprobada.
    """
    if not config_path.exists():
        logger.warning(f"Archivo de configuración no encontrado: {config_path}. Creando uno por defecto.")
        
        default_config_content = """
system:
  storage_root: "./aipha_memory_storage"

atomic_update_system:
  version_history_file_name: "VERSION_HISTORY.json"
  global_state_file_name: "global_state.json"
  action_history_file_name: "action_history.json"
  dependencies_lock_file_name: "dependencies.lock.json"
  backups_dir_name: "backups"
  config_dir_name: "config"

context_sentinel:
  knowledge_base_db_name: "knowledge_base.db"
  global_state_dir_name: "global_state"
  global_state_file_name: "current_state.json"
  action_history_dir_name: "action_history"
  action_history_file_name: "current_history.json"
"""
        config_path.write_text(default_config_content, encoding='utf-8')
        return yaml.safe_load(default_config_content)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# --- Clase principal del sistema AiphaMemorySystem ---
class AiphaMemorySystem:
    """Orquestador principal para el sistema de memoria de Aipha_1.1."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa AiphaMemorySystem con la configuración del sistema.
        Parámetros:
            config (Dict[str, Any]): Diccionario de configuración cargado desde config.yaml.
        """
        self.config = config
        self.storage_root = Path(self.config['system']['storage_root'])
        
        self.critical_memory_rules = CriticalMemoryRules(self.config)
        self.context_sentinel = ContextSentinel(self.config)
        
        self.system_initialized = False
        logger.info("AiphaMemorySystem: Componentes CriticalMemoryRules y ContextSentinel instanciados.")
        
    def initialize_system(self) -> bool:
        """
        Orquesta la inicialización completa del sistema de memoria,
        verificando la integridad de los componentes y cargando el conocimiento inicial.
        Retorna True si la inicialización es exitosa, False en caso contrario.
        """
        try:
            logger.info("AiphaMemorySystem: Iniciando el sistema de memoria crítica...")
            
            if not self.critical_memory_rules.verify_system_integrity():
                logger.error("La verificación de integridad de CriticalMemoryRules falló. El sistema no puede inicializarse.")
                return False
            
            if not self.context_sentinel.verify_knowledge_base_integrity():
                logger.error("La verificación de integridad de la base de conocimiento de ContextSentinel falló. El sistema no puede inicializarse.")
                return False
            
            self.context_sentinel.add_default_evaluation_criteria()
            
            self._add_initial_knowledge()
            
            self.system_initialized = True
            logger.info("AiphaMemorySystem: Sistema de memoria inicializado exitosamente.")
            return True
            
        except Exception as e:
            logger.error(f"AiphaMemorySystem: La inicialización del sistema falló: {e}")
            return False

    def _add_initial_knowledge(self):
        """
        Añade entradas de conocimiento iniciales al ContextSentinel.
        Este es el "conocimiento base" que el sistema tendrá al arrancar.
        """
        logger.info("AiphaMemorySystem: Añadiendo entradas de conocimiento iniciales...")
        self.context_sentinel.add_knowledge_entry(
            category="architecture",
            title="Aipha_1.1 Layer Architecture",
            content="Aipha_1.1 se estructura en 5 capas principales: Núcleo (redesignHelper), Context Sentinel, Data Preprocessor, Trading Flow, y Oracles.",
            metadata={"importance": "critical", "layer": "architecture", "version": "1.1.0"}
        )
        self.context_sentinel.add_knowledge_entry(
            category="protocol",
            title="Atomic Update Protocol",
            content="El protocolo de actualización atómica asegura cambios controlados con aprobación explícita, previniendo reescrituras completas.",
            metadata={"type": "protocol", "critical": True, "version": "1.1.0"}
        )
        self.context_sentinel.add_knowledge_entry(
            category="project_plan",
            title="ATR Dynamic Barriers Proposal Details",
            content="Propuesta para reemplazar TP/SL fijos con barreras dinámicas basadas en Average True Range (ATR). Parámetros: atr_period, tp_multiplier, sl_multiplier.",
            metadata={"proposal_id": "pce-atr-001", "component": "aipha/trading_flow/labelers/potential_capture_engine.py", "version": "1.1.0"}
        )
        logger.info("AiphaMemorySystem: Entradas de conocimiento iniciales añadidas.")

    def demonstrate_atomic_update_protocol(self) -> bool:
        """
        Demuestra el flujo completo del protocolo de actualización atómica.
        Incluye la creación, aprobación y aplicación (simulada) de una propuesta.
        """
        logger.info("\n=== INICIANDO DEMOSTRACIÓN DEL PROTOCOLO DE ACTUALIZACIÓN ATÓMICA ===")

        # 1. Crear una propuesta de cambio de ejemplo (Propuesta ATR).
        # Usamos el método auxiliar que definiremos para esto.
        proposal_atr = self._create_sample_proposal()
        if not proposal_atr:
            logger.error("No se pudo crear la propuesta de ejemplo. Terminando demostración.")
            return False
        
        logger.info(f"Propuesta de ejemplo creada: ID='{proposal_atr.change_id}', Título='{proposal_atr.description}'")

        # --- Marcadores de posición para Proceso de Aprobación, Aplicación y Verificación ---
        logger.info("La propuesta de cambio ATR ha sido generada.")
        logger.info("Aquí irán los pasos de Aprobación, Aplicación Atómica y Verificación.")

        logger.info("=== DEMOSTRACIÓN DEL PROTOCOLO DE ACTUALIZACIÓN ATÓMICA FINALIZADA (Parcial). ===")
        return True # Retornamos True por ahora, ya que solo hemos implementado la creación.


    def _create_sample_proposal(self) -> ChangeProposal:
        """
        Crea una propuesta de cambio de ejemplo, la propuesta de Barreras Dinámicas con ATR.
        Retorna el objeto ChangeProposal.
        """
        logger.info("Creando propuesta de ejemplo para Barreras Dinámicas con ATR...")
        # Definimos los detalles de la propuesta ATR.
        diff_content_atr = """
        --- a/aipha/trading_flow/labelers/potential_capture_engine.py    (original)
        +++ b/aipha/trading_flow/labelers/potential_capture_engine.py    (modificado)
        @@ -1,10 +1,18 @@
         import pandas as pd
        +import pandas_ta as ta
         
         class PotentialCaptureEngine:
        -    def __init__(self, **kwargs):
        -        self.cfg = {
        -            'tp_fixed': kwargs.get('tp_fixed', 5.0),
        -            'sl_fixed': kwargs.get('sl_fixed', 3.0),
        +    def __init__(self, prices: pd.DataFrame, **kwargs):
        +        self.prices = prices # El DataFrame de precios ahora se pasa al constructor
        +        # Calculamos ATR una vez en la inicialización si se usan barreras dinámicas
        +        self.atr_period = kwargs.get('atr_period', 20)
        +        self.atr = ta.atr(self.prices['high'], self.prices['low'], self.prices['close'], length=self.atr_period)
        +        
        +        self.cfg = { # Configuraciones para barreras dinámicas/fijas
        +            'tp_multiplier': kwargs.get('tp_multiplier', 5.0), # Multiplicador para TP con ATR
        +            'sl_multiplier': kwargs.get('sl_multiplier', 3.0), # Multiplicador para SL con ATR
        +            'tp_fixed': kwargs.get('tp_fixed', None), # TP fijo (si se usa en lugar de ATR)
        +            'sl_fixed': kwargs.get('sl_fixed', None), # SL fijo (si se usa en lugar de ATR)
                         'time_limit': kwargs.get('time_limit', 20),
                 }
         
        @@ -14,13 +22,23 @@
                labels = pd.Series(0, index=valid_events, dtype=int)
               
                 for t0 in valid_events:
                    entry_price = df.loc[t0, 'close']
        -            sl = entry_price - self.cfg['sl_fixed']
        -            tp = entry_price + self.cfg['tp_fixed']
        +
        +            # Determinamos las barreras TP/SL, priorizando ATR dinámico si los multiplicadores están definidos
        +            if self.cfg['tp_multiplier'] is not None and self.cfg['sl_multiplier'] is not None:
        +                current_atr = self.atr.loc[t0]
        +                if pd.isna(current_atr): # Manejar NaN si no hay suficientes datos para ATR
        +                    labels[t0] = 0 # No se puede etiquetar si no hay ATR
        +                    continue
        +                sl = entry_price - (current_atr * self.cfg['sl_multiplier'])
        +                tp = entry_price + (current_atr * self.cfg['tp_multiplier'])
        +            elif self.cfg['tp_fixed'] is not None and self.cfg['sl_fixed'] is not None:
        +                sl = entry_price - self.cfg['sl_fixed']
        +                tp = entry_price + self.cfg['tp_fixed']
        +            else:
        +                labels[t0] = 0 # No se puede etiquetar si no hay configuración de barreras
        +                continue
         
                        end_loc = min(df.index.get_loc(t0) + self.cfg['time_limit'] + 1, len(df))
                        path = df.iloc[df.index.get_loc(t0) + 1 : end_loc]
         
        """
        
        # Usamos critical_memory_rules para crear la propuesta.
        proposal = self.critical_memory_rules.create_change_proposal(
            description="Implementación de Barreras Dinámicas con ATR",
            justification="Adaptar el motor a diferentes regímenes de volatilidad para reducir falsas señales y mejorar la captura de beneficios.",
            files_affected=["aipha/trading_flow/labelers/potential_capture_engine.py"],
            diff_content=diff_content_atr, # Incluimos el contenido del diff real (como un string).
            author="Aipha_System (ChangeProposer_Simulated)",
            compatibility_check="Verificado en fase de propuesta, no funcional todavía.",
            rollback_plan="Revertir los cambios en potential_capture_engine.py a la versión previa sin ATR dinámico."
        )
        logger.info("Propuesta de Barreras Dinámicas con ATR creada y registrada.")
        return proposal


# --- Función principal de ejecución del programa ---
def main():
    """
    Función principal que orquesta la inicialización y ejecución del sistema Aipha_1.1.
    """
    logger.info("Iniciando el sistema de reglas de memoria críticas de Build_Aipha_1.1...")
    
    config_file_path = Path("config.yaml")
    config = load_config(config_file_path) 
    storage_root = Path(config['system']['storage_root'])
    
    # --- Limpieza Opcional del Directorio de Almacenamiento ---
    if storage_root.exists():
        logger.info(f"Limpiando la ruta raíz de almacenamiento: '{storage_root}' para una demostración limpia...")
        try:
            shutil.rmtree(storage_root) 
            logger.info(f"Directorio '{storage_root}' limpiado exitosamente.")
        except OSError as e:
            logger.error(f"Error al eliminar el directorio '{storage_root}': {e}")
            logger.warning("Procediendo con el almacenamiento existente, lo que podría afectar la consistencia de la demostración.")
    
    # --- Inicialización del Sistema Aipha ---
    aipha_system = AiphaMemorySystem(config) 
    
    if not aipha_system.initialize_system(): 
        logger.error("La inicialización del sistema AiphaMemorySystem falló. Terminando programa.")
        return 1
    
    # --- Demostración del Protocolo de Actualización Atómica ---
    # Ahora que el sistema está inicializado, demostramos el flujo de una propuesta.
    if not aipha_system.demonstrate_atomic_update_protocol():
        logger.error("La demostración del protocolo de actualización atómica falló. Terminando programa.")
        return 1

    logger.info("El sistema Aipha_1.1 está listo para operar (inicialización y demostración completadas).")
    
    return 0

# --- Bloque de ejecución principal (Entry Point) ---
if __name__ == "__main__":
    sys.exit(main())