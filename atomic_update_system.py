# atomic_update_system.py

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import yaml
from dataclasses import dataclass, asdict
from enum import Enum
import shutil

# --- Configuración del sistema de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
logger = logging.getLogger(__name__)

# --- Función para cargar la configuración desde un archivo YAML (para pruebas independientes) ---
def load_config(config_path: Path) -> Dict[str, Any]:
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


# --- Enumeraciones ---
class ChangeType(Enum):
    """Tipos de cambios que pueden ser aplicados al sistema."""
    MAJOR = "major"
    MINOR = "minor" 
    PATCH = "patch"
    EMERGENCY = "emergency"


class ApprovalStatus(Enum):
    """Estados de aprobación para una propuesta de cambio."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


# --- Clases de Datos (Dataclasses) ---
@dataclass
class ChangeProposal:
    """Representa una propuesta de cambio para el sistema."""
    change_id: str
    timestamp: str
    version: str
    author: str
    description: str
    justification: str
    files_affected: List[str]
    diff_content: str
    compatibility_check: str
    rollback_plan: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approval_timestamp: Optional[str] = None


@dataclass
class VersionInfo:
    """Información de versión del sistema."""
    major: int
    minor: int
    patch: int
    build_timestamp: str
    
    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}-{self.build_timestamp.replace(':', '-').split('.')[0]}"


# --- Clase Principal: CriticalMemoryRules ---
class CriticalMemoryRules:
    """Implementa las reglas de memoria críticas para el almacenamiento permanente y las actualizaciones atómicas."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa CriticalMemoryRules cargando las rutas desde la configuración.
        Parámetros:
            config (Dict[str, Any]): Diccionario de configuración cargado desde config.yaml.
        """
        atomic_config = config.get('atomic_update_system', {})
        system_config = config.get('system', {})

        self.storage_root = Path(system_config.get('storage_root', "."))
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        self.version_history_file = self.storage_root / atomic_config.get('version_history_file_name', "VERSION_HISTORY.json")
        self.global_state_file = self.storage_root / atomic_config.get('global_state_file_name', "global_state.json")
        self.action_history_file = self.storage_root / atomic_config.get('action_history_file_name', "action_history.json")
        
        self.config_dir = self.storage_root / atomic_config.get('config_dir_name', "config")
        self.config_dir.mkdir(exist_ok=True)
        self.dependencies_file = self.config_dir / atomic_config.get('dependencies_lock_file_name', "dependencies.lock.json")
        
        self.backups_dir = self.storage_root / atomic_config.get('backups_dir_name', "backups")
        self.backups_dir.mkdir(exist_ok=True)
        
        # Cargamos la versión actual ANTES de inicializar el almacenamiento,
        # porque _initialize_storage puede necesitar la versión para la entrada inicial.
        self.current_version = self._load_current_version() 
        
        self._initialize_storage() # Ahora esto asegura que VERSION_HISTORY.json tenga una entrada inicial.
        logger.info(f"CriticalMemoryRules inicializado. Ruta raíz: {self.storage_root}")
    
    def _initialize_storage(self):
        """Inicializa los archivos de almacenamiento si no existen o están vacíos."""
        files_to_check = [self.version_history_file, self.global_state_file, self.action_history_file, self.dependencies_file]
        
        for f_path in files_to_check:
            if not f_path.exists() or f_path.stat().st_size == 0:
                if f_path == self.version_history_file:
                    # Lógica especial para VERSION_HISTORY.json: crear con una entrada inicial.
                    initial_version_entry = {
                        "version": str(self.current_version), # Usamos la versión que ya cargamos/por defecto.
                        "timestamp": datetime.utcnow().isoformat() + 'Z',
                        "status": "active", 
                        "author": "System",
                        "approved_by": "Developer",
                        "changes": ["Initial system initialization"],
                        "files_affected": ["all"],
                        "compatibility_check": "passed",
                        "rollback_available": False,
                        "change_id": "INIT_000"
                    }
                    self._save_json(f_path, [initial_version_entry])
                    logger.info(f"Archivo de almacenamiento inicializado: {f_path.name} con entrada inicial.")
                else:
                    # Para los demás archivos (global_state, action_history, dependencies), la lógica es la misma.
                    default_content = [] if f_path == self.action_history_file else {} # action_history es lista, otros diccionarios.
                    self._save_json(f_path, default_content)
                    logger.info(f"Archivo de almacenamiento inicializado: {f_path.name}")
            else:
                logger.debug(f"Archivo de almacenamiento existente: {f_path.name}")
    
    def _load_current_version(self) -> VersionInfo:
        """Carga la versión actual del sistema desde el historial de versiones."""
        # Nota: Este método se llama ANTES de _initialize_storage.
        # Si el archivo no existe, no hay historial, por lo que devolvemos una versión por defecto.
        # _initialize_storage se encargará de crear el archivo con la entrada inicial.
        if self.version_history_file.exists() and self.version_history_file.stat().st_size > 0:
            try:
                history = self._load_json(self.version_history_file)
                if history:
                    latest = history[-1]
                    version_str_parts = latest["version"].split('-')
                    version_num_parts = version_str_parts[0].split('.')
                    major, minor, patch = map(int, version_num_parts)
                    build_timestamp = latest["timestamp"]
                    return VersionInfo(major, minor, patch, build_timestamp)
            except json.JSONDecodeError as e:
                logger.error(f"Error de decodificación JSON en {self.version_history_file}: {e}")
            except Exception as e:
                logger.error(f"Error al cargar la versión actual: {e}")
        
        logger.warning("No se pudo cargar la versión actual del historial. Inicializando con versión por defecto 1.1.0.")
        return VersionInfo(1, 1, 0, datetime.utcnow().isoformat() + 'Z')
    
    def create_change_proposal(self, 
                             description: str,
                             justification: str,
                             files_affected: List[str],
                             diff_content: str,
                             author: str,
                             compatibility_check: str = "passed",
                             rollback_plan: str = "Revertir a la versión previa") -> ChangeProposal:
        """Crea y registra una nueva propuesta de cambio."""
        
        change_id = hashlib.md5(f"{description}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8]
        
        new_version = VersionInfo(
            self.current_version.major,
            self.current_version.minor,
            self.current_version.patch + 1,
            datetime.utcnow().isoformat() + 'Z'
        )
        
        proposal = ChangeProposal(
            change_id=change_id,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            version=str(new_version),
            author=author,
            description=description,
            justification=justification,
            files_affected=files_affected,
            diff_content=diff_content,
            compatibility_check=compatibility_check,
            rollback_plan=rollback_plan
        )
        
        logger.info(f"Propuesta de cambio creada: {change_id} - {description}")
        self._record_action(f"ChangeProposal: Propuesta '{change_id}' creada", agent="CriticalMemoryRules", component="atomic_update_system")
        
        return proposal
    
    def approve_change(self, proposal: ChangeProposal, approved_by: str) -> bool:
        """Aprueba una propuesta de cambio, si su estado es PENDING."""
        if proposal.status != ApprovalStatus.PENDING:
            logger.error(f"La propuesta {proposal.change_id} no está pendiente de aprobación. Estado actual: {proposal.status.value}")
            return False
        
        proposal.status = ApprovalStatus.APPROVED
        proposal.approved_by = approved_by
        proposal.approval_timestamp = datetime.utcnow().isoformat() + 'Z'
        
        logger.info(f"Propuesta de cambio aprobada: {proposal.change_id} por {approved_by}")
        self._record_action(f"Approval: Propuesta '{proposal.change_id}' aprobada por {approved_by}", agent="CriticalMemoryRules", component="atomic_update_system")
        
        return True
    
    def reject_change(self, proposal: ChangeProposal, rejected_by: str, reason: str) -> bool:
        """Rechaza una propuesta de cambio, si su estado es PENDING."""
        if proposal.status != ApprovalStatus.PENDING:
            logger.error(f"La propuesta {proposal.change_id} no está pendiente de aprobación. Estado actual: {proposal.status.value}")
            return False
        
        proposal.status = ApprovalStatus.REJECTED
        proposal.approved_by = rejected_by
        proposal.approval_timestamp = datetime.utcnow().isoformat() + 'Z'
        
        logger.info(f"Propuesta de cambio rechazada: {proposal.change_id} por {rejected_by} - Razón: {reason}")
        self._record_action(f"Rejection: Propuesta '{proposal.change_id}' rechazada por {rejected_by} - Razón: {reason}", agent="CriticalMemoryRules", component="atomic_update_system")
        
        return True
    
    def apply_atomic_update(self, proposal: ChangeProposal) -> bool:
        """
        Aplica un cambio aprobado siguiendo el protocolo de actualización atómica de 5 pasos.
        Este método garantiza la seguridad y la auditabilidad del proceso.
        """
        if proposal.status != ApprovalStatus.APPROVED:
            logger.error(f"No se puede aplicar la propuesta '{proposal.change_id}' porque no está aprobada. Estado actual: {proposal.status.value}")
            return False
        
        try:
            # 1. Crear una copia de seguridad del estado actual del sistema.
            self._create_backup()
            logger.info(f"Backup creado exitosamente antes de aplicar la propuesta '{proposal.change_id}'.")
            
            # 2. Aplicar el cambio. (¡AÚN ES UNA SIMULACIÓN!)
            logger.info(f"Aplicando actualización atómica para la propuesta: {proposal.change_id} - '{proposal.description}'")
            logger.info(f"Archivos afectados según la propuesta: {', '.join(proposal.files_affected)}")
            
            global_state = self._load_json(self.global_state_file)
            global_state[f"last_applied_proposal_{proposal.change_id}"] = asdict(proposal)
            self._save_json(self.global_state_file, global_state)
            logger.info(f"Simulación: La propuesta '{proposal.change_id}' se ha 'aplicado' al estado global.")

            # 3. Actualizar el historial de versiones con el cambio aplicado.
            self._update_version_history(proposal)
            logger.info(f"Historial de versiones actualizado con la propuesta '{proposal.change_id}'.")
            
            # 4. Actualizar la versión actual del sistema.
            version_num_parts = proposal.version.split('-')[0].split('.')
            self.current_version = VersionInfo(
                int(version_num_parts[0]),
                int(version_num_parts[1]),
                int(version_num_parts[2]),
                datetime.utcnow().isoformat() + 'Z'
            )
            logger.info(f"Versión actual del sistema actualizada a: {self.current_version}")
            
            # 5. Realizar una verificación de integridad post-actualización.
            if not self.verify_system_integrity():
                raise RuntimeError("La verificación de integridad falló DESPUÉS de aplicar la actualización. ¡Posible corrupción!")

            self._record_action(f"AtomicUpdate: Propuesta '{proposal.change_id}' aplicada exitosamente", agent="CriticalMemoryRules", component="atomic_update_system")
            logger.info(f"Actualización atómica para la propuesta '{proposal.change_id}' completada exitosamente.")
            return True

        except Exception as e:
            logger.error(f"Fallo crítico durante la aplicación atómica de la propuesta '{proposal.change_id}': {e}")
            self._record_action(f"Error: Fallo al aplicar la propuesta '{proposal.change_id}' - {e}", agent="CriticalMemoryRules", component="atomic_update_system", status="failure")
            return False
    
    def _create_backup(self):
        """
        Crea una copia de seguridad de todos los archivos de estado críticos del sistema.
        Utiliza hard links para eficiencia si es posible, de lo contrario, copia el contenido.
        """
        backup_timestamp = datetime.utcnow().isoformat().replace(':', '-').replace('.', '_') + 'Z'
        backup_dir = self.backups_dir / backup_timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        files_to_backup = [
            self.version_history_file,
            self.global_state_file,
            self.action_history_file,
            self.dependencies_file 
        ]
        
        for file_path in files_to_backup:
            if file_path.exists():
                backup_path = backup_dir / file_path.name
                try:
                    os.link(file_path, backup_path)
                    logger.debug(f"Backup (hard link): {file_path.name} a {backup_path}")
                except OSError: 
                    shutil.copy2(file_path, backup_path)
                    logger.debug(f"Backup (copia): {file_path.name} a {backup_path}")
            else:
                logger.warning(f"Archivo no encontrado para copia de seguridad: {file_path}")
        
        logger.info(f"Copia de seguridad creada en: {backup_dir}")
        self._record_action(f"Backup: Estado del sistema respaldado en {backup_dir}", agent="CriticalMemoryRules", component="atomic_update_system")
    
    def _update_version_history(self, proposal: ChangeProposal):
        """
        Añade una nueva entrada al historial de versiones con los detalles de la propuesta aplicada.
        """
        version_entry = {
            "version": proposal.version,
            "timestamp": proposal.timestamp,
            "status": "active",
            "author": proposal.author,
            "approved_by": proposal.approved_by,
            "changes": [proposal.description],
            "files_affected": proposal.files_affected,
            "compatibility_check": proposal.compatibility_check,
            "rollback_available": True,
            "change_id": proposal.change_id
        }
        
        history = self._load_json(self.version_history_file)
        history.append(version_entry)
        self._save_json(self.version_history_file, history)
        
        logger.info(f"Historial de versiones actualizado para la versión: {proposal.version}")
    
    def _record_action(self, action_description: str, agent: str = "CriticalMemoryRules", component: str = "atomic_update_system", status: str = "success", details: Optional[Dict[str, Any]] = None):
        """
        Registra una acción en el historial de acciones del sistema.
        Es clave para la trazabilidad y auditoría.
        """
        action = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "action": action_description,
            "agent": agent,
            "component": component,
            "status": status
        }
        if details:
            action['details'] = details
        
        history = self._load_json(self.action_history_file)
        history.append(action)
        self._save_json(self.action_history_file, history)
    
    def _load_json(self, file_path: Path) -> Any:
        """
        Método auxiliar interno para cargar datos JSON de un archivo de forma segura.
        Maneja archivos no existentes, vacíos o corruptos.
        """
        if not file_path.exists() or file_path.stat().st_size == 0:
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

    def get_version_history(self) -> List[Dict[str, Any]]:
        """Devuelve el historial completo de versiones del sistema."""
        return self._load_json(self.version_history_file)
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Devuelve el historial completo de acciones registradas por CriticalMemoryRules."""
        return self._load_json(self.action_history_file)
    
    def get_current_version(self) -> str:
        """Devuelve la cadena de la versión actual del sistema."""
        return str(self.current_version)
    
    def verify_system_integrity(self) -> bool:
        """
        Verifica la integridad del sistema CriticalMemoryRules.
        Comprueba la existencia de archivos clave y la coherencia del historial de versiones.
        """
        try:
            required_files = [
                self.version_history_file,
                self.global_state_file,
                self.action_history_file,
                self.dependencies_file
            ]
            
            for file_path in required_files:
                if not file_path.exists():
                    logger.error(f"Verificación de integridad fallida: Falta archivo requerido: {file_path}")
                    return False
                if file_path.stat().st_size == 0 and ("history" in file_path.name or "state" in file_path.name or "dependencies" in file_path.name):
                    logger.warning(f"Verificación de integridad: Archivo requerido está vacío: {file_path}")
            
            version_history = self.get_version_history()
            if not version_history:
                logger.error("Verificación de integridad fallida: El historial de versiones está vacío o no se pudo cargar.")
                return False
            
            latest_version = version_history[-1]
            if latest_version.get("status") != "active":
                logger.error("Verificación de integridad fallida: La última versión en el historial no está marcada como 'activa'.")
                return False
            
            logger.info("Verificación de integridad de CriticalMemoryRules completada exitosamente.")
            return True
            
        except Exception as e:
            logger.error(f"Verificación de integridad del sistema falló: {e}")
            return False


# --- Bloque de ejecución de ejemplo para pruebas independientes ---
if __name__ == "__main__":
    logger.info("--- Ejecutando demostración independiente de atomic_update_system.py ---")
    
    config_file_path = Path("config.yaml")
    config = load_config(config_file_path)
    
    storage_root = Path(config['system']['storage_root'])
    if storage_root.exists():
        logger.info(f"Limpiando directorio de almacenamiento '{storage_root}' para test independiente.")
        try:
            shutil.rmtree(storage_root)
        except OSError as e:
            logger.warning(f"No se pudo limpiar '{storage_root}': {e}. Podría afectar la consistencia.")

    system = CriticalMemoryRules(config)
    
    logger.info(f"Versión actual al inicio: {system.get_current_version()}")
    logger.info(f"Integridad del sistema al inicio: {system.verify_system_integrity()}")
    
    sample_proposal = system.create_change_proposal(
        description="Añadir documentación detallada sobre el framework de testing.",
        justification="Mejorar la fiabilidad del sistema con procedimientos de testing y guías de QA.",
        files_affected=["Build_Aipha_1.1.md", "project_structure.md"],
        diff_content="Sección de testing con tests unitarios, de integración y documentación de CI/CD.",
        author="TestSystem"
    )
    
    logger.info(f"Propuesta creada: {sample_proposal.change_id}")
    
    system.approve_change(sample_proposal, "Developer_Manual")
    logger.info(f"Propuesta aprobada por: {sample_proposal.approved_by}")
    
    success = system.apply_atomic_update(sample_proposal)
    logger.info(f"Actualización atómica exitosa: {success}")
    
    logger.info(f"Versión actual final: {system.get_current_version()}")
    logger.info(f"Integridad del sistema final: {system.verify_system_integrity()}")
    
    logger.info("\n--- Historial de Versiones ---")
    for entry in system.get_version_history():
        logger.info(json.dumps(entry, indent=2))

    logger.info("\n--- Historial de Acciones (últimas 5) ---")
    for entry in system.get_action_history()[-5:]:
        logger.info(json.dumps(entry, indent=2))
    
    logger.info("--- Demostración independiente de atomic_update_system.py completada. ---")