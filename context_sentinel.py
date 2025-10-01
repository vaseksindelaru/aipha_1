# context_sentinel.py

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import sqlite3 # Para la base de conocimiento
from dataclasses import dataclass, asdict
import hashlib # Para generar IDs únicos
import yaml # Importamos yaml porque la función load_config podría ser necesaria para pruebas independientes

# --- Configuración del sistema de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
logger = logging.getLogger(__name__)

# --- Función para cargar la configuración desde un archivo YAML (para pruebas independientes) ---
# Al igual que en atomic_update_system.py, esta función está aquí para permitir
# que context_sentinel.py se ejecute de forma autónoma para pruebas.
# En el flujo normal del sistema, main.py es quien carga la configuración y se la pasa.
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


# --- Clases de Datos (Dataclasses) ---
@dataclass
class KnowledgeEntry:
    """Representa una entrada en la base de conocimiento del sistema."""
    id: str
    category: str # Categoría para organizar el conocimiento (ej. "architecture", "protocol")
    title: str # Título descriptivo de la entrada
    content: str # Contenido principal de la entrada (texto, código, etc.)
    metadata: Dict[str, Any] # Metadatos adicionales (ej. importancia, capa, versión)
    timestamp: str # Marca de tiempo de cuándo se añadió la entrada
    version: str # Versión del sistema a la que pertenece esta entrada de conocimiento
    embedding: Optional[List[float]] = None # Placeholder para futuros embeddings vectoriales (para RAG avanzado)


@dataclass
class GlobalAgentState:
    """Representa el estado global compartido por todos los agentes en el sistema."""
    version: str # Versión actual del sistema
    timestamp: str # Última marca de tiempo de actualización del estado global
    system_status: str # Estado general del sistema (ej. "active", "initializing")
    components: Dict[str, str] # Estado de los componentes individuales (ej. "redesign_helper": "operational")
    latest_proposal: Optional[Dict[str, Any]] # Detalles de la última propuesta gestionada
    proposal_evaluation: Optional[Dict[str, Any]] # Resultado de la evaluación de la última propuesta
    implementation_details: Optional[Dict[str, Any]] # Detalles de la implementación de la propuesta
    integration_status: Optional[Dict[str, Any]] # Estado de integración de la propuesta


# --- Clase Principal: ContextSentinel ---
class ContextSentinel:
    """
    ContextSentinel proporciona gestión de la base de conocimiento y de la memoria
    para Aipha_1.1, sirviendo como la fuente única de verdad para el contexto del sistema.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa ContextSentinel cargando las rutas desde la configuración.
        Parámetros:
            config (Dict[str, Any]): Diccionario de configuración cargado desde config.yaml.
        """
        # Extraemos las configuraciones específicas para este componente y la raíz del sistema.
        context_config = config.get('context_sentinel', {})
        system_config = config.get('system', {})

        # Establecemos la ruta raíz para todos los archivos gestionados por este componente.
        self.storage_root = Path(system_config.get('storage_root', "."))
        self.storage_root.mkdir(parents=True, exist_ok=True) # Nos aseguramos de que esta carpeta exista.

        # Construimos las rutas completas a la base de datos y directorios de JSON.
        self.db_file = self.storage_root / context_config.get('knowledge_base_db_name', "knowledge_base.db")
        
        self.global_state_dir = self.storage_root / context_config.get('global_state_dir_name', "global_state")
        self.global_state_dir.mkdir(parents=True, exist_ok=True) # Aseguramos que este directorio exista.
        self.global_state_file = self.global_state_dir / context_config.get('global_state_file_name', "current_state.json")
        
        self.action_history_dir = self.storage_root / context_config.get('action_history_dir_name', "action_history")
        self.action_history_dir.mkdir(parents=True, exist_ok=True) # Aseguramos que este directorio exista.
        self.action_history_file = self.action_history_dir / context_config.get('action_history_file_name', "current_history.json")
        
        # Inicializamos la base de datos SQLite y los archivos JSON.
        self._initialize_knowledge_base()
        self._initialize_global_state()
        self._initialize_action_history()
        logger.info(f"ContextSentinel inicializado. Ruta raíz: {self.storage_root}")
    
    def _initialize_knowledge_base(self):
        """Inicializa la base de datos SQLite, creando las tablas necesarias si no existen."""
        conn = sqlite3.connect(self.db_file) # Conecta o crea la base de datos SQLite.
        cursor = conn.cursor() # Un cursor permite ejecutar comandos SQL.
        
        # Tabla para entradas de conocimiento general.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                version TEXT NOT NULL,
                embedding TEXT
            )
        ''')
        
        # Tabla para criterios de evaluación de propuestas.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_criteria (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE, -- 'UNIQUE' asegura que no haya nombres duplicados en la misma categoría.
                weight REAL NOT NULL,
                description TEXT NOT NULL,
                examples_positive TEXT NOT NULL,
                examples_negative TEXT NOT NULL,
                category TEXT NOT NULL
            )
        ''')
        
        # Tabla para ejemplos de código (para Agentic RAG simple).
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_examples (
                id TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                language TEXT NOT NULL,
                code TEXT NOT NULL,
                description TEXT NOT NULL,
                tags TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        conn.commit() # Guarda los cambios en la base de datos.
        conn.close() # Cierra la conexión a la base de datos.
        
        logger.info("Base de conocimiento SQLite inicializada (tablas creadas/verificadas).")
    
    def _initialize_global_state(self):
        """Inicializa el archivo de estado global (current_state.json) si no existe o está vacío."""
        if not self.global_state_file.exists() or self.global_state_file.stat().st_size == 0:
            # Creamos un estado global inicial por defecto.
            initial_state = GlobalAgentState(
                version="1.1.0",
                timestamp=datetime.utcnow().isoformat() + 'Z',
                system_status="active",
                components={
                    "redesign_helper": "operational",
                    "context_sentinel": "operational",
                    "change_proposer": "ready",
                    "proposal_evaluator": "ready",
                    "codecraft_sage": "ready",
                    "meta_improver": "ready"
                },
                latest_proposal=None,
                proposal_evaluation=None,
                implementation_details=None,
                integration_status=None
            )
            
            # Guardamos el estado inicial en el archivo JSON.
            self._save_json(self.global_state_file, asdict(initial_state))
            logger.info(f"Archivo de estado global inicializado: {self.global_state_file.name}")
    
    def _initialize_action_history(self):
        """Inicializa el archivo de historial de acciones (current_history.json) si no existe o está vacío."""
        if not self.action_history_file.exists() or self.action_history_file.stat().st_size == 0:
            # Creamos una entrada de acción inicial por defecto.
            initial_action = {
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "action": "ContextSentinel: Sistema inicializado",
                "agent": "System",
                "component": "context_sentinel",
                "status": "success",
                "details": {
                    "message": "Context Sentinel inicializado con base de conocimiento",
                    "version": "1.1.0"
                }
            }
            
            # Guardamos la acción inicial en el archivo JSON.
            self._save_json(self.action_history_file, [initial_action])
            logger.info(f"Archivo de historial de acciones inicializado: {self.action_history_file.name}")

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
        Actualiza el estado global del sistema (current_state.json).
        Es el "tablón de anuncios" donde los agentes dejan y recogen información.
        Retorna True si la actualización es exitosa, False en caso contrario.
        """
        try:
            current_state = self._load_json(self.global_state_file) # Carga el estado actual.
            current_state.update(state_update) # Combina el estado actual con las nuevas actualizaciones.
            current_state["timestamp"] = datetime.utcnow().isoformat() + 'Z' # Actualiza la marca de tiempo.
            
            self._save_json(self.global_state_file, current_state) # Guarda el estado global actualizado.
            
            logger.info(f"Estado global actualizado. Claves modificadas: {list(state_update.keys())}")
            self.record_action(f"GlobalState: Actualizado {list(state_update.keys())}", agent="ContextSentinel", component="context_sentinel")
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
        Registra una acción en el historial de acciones del sistema (current_history.json).
        Es el "diario" cronológico de todo lo que hace el sistema.
        Retorna True si el registro es exitoso, False en caso contrario.
        """
        try:
            action = {
                "timestamp": datetime.utcnow().isoformat() + 'Z',
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
    
    def add_knowledge_entry(self, category: str, title: str, content: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           version: str = "1.1.0") -> str:
        """
        Añade una nueva entrada de conocimiento a la base de datos SQLite.
        Aquí se guarda documentación, información de arquitectura, etc.
        Retorna el ID de la entrada creada, o una cadena vacía si falla.
        """
        try:
            # Generamos un ID único para la entrada de conocimiento.
            entry_id = hashlib.md5(f"{title}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
            
            entry = KnowledgeEntry(
                id=entry_id,
                category=category,
                title=title,
                content=content,
                metadata=metadata or {}, # Aseguramos que siempre haya un diccionario de metadatos.
                timestamp=datetime.utcnow().isoformat() + 'Z',
                version=version
            )
            
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Insertamos la nueva entrada en la tabla knowledge_entries.
            cursor.execute('''
                INSERT INTO knowledge_entries (id, category, title, content, metadata, timestamp, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.id,
                entry.category,
                entry.title,
                entry.content,
                json.dumps(entry.metadata), # Los metadatos se guardan como JSON dentro de un campo de texto.
                entry.timestamp,
                entry.version
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Entrada de conocimiento añadida: {entry_id} - '{title}'")
            self.record_action(f"KnowledgeBase: Entrada añadida - '{title}'", details={"entry_id": entry_id}, agent="ContextSentinel", component="context_sentinel")
            
            return entry_id
            
        except Exception as e:
            logger.error(f"Fallo al añadir entrada de conocimiento: {e}")
            return ""
    
    def get_knowledge_entries(self, category: Optional[str] = None, 
                            limit: int = 100) -> List[KnowledgeEntry]:
        """
        Recupera entradas de conocimiento de la base de datos,
        opcionalmente filtradas por categoría y limitando el número de resultados.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Construimos la consulta SQL.
            sql_query = '''
                SELECT id, category, title, content, metadata, timestamp, version
                FROM knowledge_entries
            '''
            params = []
            if category:
                sql_query += ' WHERE category = ?'
                params.append(category)
            
            sql_query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(sql_query, params)
            
            entries = []
            for row in cursor.fetchall():
                # Convertimos cada fila de la DB de nuevo a un objeto KnowledgeEntry.
                entry = KnowledgeEntry(
                    id=row[0],
                    category=row[1],
                    title=row[2],
                    content=row[3],
                    metadata=json.loads(row[4]), # Convertimos los metadatos de JSON a diccionario.
                    timestamp=row[5],
                    version=row[6]
                )
                entries.append(entry)
            
            conn.close()
            return entries
            
        except Exception as e:
            logger.error(f"Fallo al recuperar entradas de conocimiento: {e}")
            return []
    
    def add_evaluation_criteria(self, name: str, weight: float, description: str,
                               examples_positive: List[str], examples_negative: List[str],
                               category: str = "general") -> str:
        """
        Añade o actualiza criterios de evaluación en la base de datos SQLite.
        Utiliza una lógica de "upsert": si el criterio ya existe (por nombre y categoría), lo actualiza;
        de lo contrario, lo inserta.
        Retorna el ID del criterio, o una cadena vacía si falla.
        """
        try:
            criteria_id = hashlib.md5(f"{name}{category}".encode()).hexdigest()[:12] # Genera un ID.
            
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Comprobamos si ya existe un criterio con este nombre y categoría.
            cursor.execute("SELECT id FROM evaluation_criteria WHERE name = ? AND category = ?", (name, category))
            existing_id = cursor.fetchone()

            if existing_id:
                # Si existe, actualizamos sus valores.
                criteria_id = existing_id[0]
                cursor.execute('''
                    UPDATE evaluation_criteria
                    SET weight = ?, description = ?, examples_positive = ?, examples_negative = ?
                    WHERE id = ?
                ''', (
                    weight,
                    description,
                    json.dumps(examples_positive),
                    json.dumps(examples_negative),
                    criteria_id
                ))
                logger.info(f"Criterio de evaluación actualizado: '{name}' (peso: {weight})")
                self.record_action(f"Evaluation: Criterio actualizado - '{name}'", details={"criteria_id": criteria_id, "action": "updated"}, agent="ContextSentinel", component="context_sentinel")
            else:
                # Si no existe, lo insertamos.
                cursor.execute('''
                    INSERT INTO evaluation_criteria 
                    (id, name, weight, description, examples_positive, examples_negative, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    criteria_id,
                    name,
                    weight,
                    description,
                    json.dumps(examples_positive),
                    json.dumps(examples_negative),
                    category
                ))
                logger.info(f"Criterio de evaluación añadido: '{name}' (peso: {weight})")
                self.record_action(f"Evaluation: Criterio añadido - '{name}'", details={"criteria_id": criteria_id, "action": "added"}, agent="ContextSentinel", component="context_sentinel")
            
            conn.commit()
            conn.close()
            
            return criteria_id
            
        except Exception as e:
            logger.error(f"Fallo al añadir/actualizar criterio de evaluación: {e}")
            return ""
    
    def get_evaluation_criteria(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Devuelve los criterios de evaluación de la base de datos,
        opcionalmente filtrados por categoría y ordenados por peso.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            sql_query = '''
                SELECT id, name, weight, description, examples_positive, examples_negative, category
                FROM evaluation_criteria
            '''
            params = []
            if category:
                sql_query += ' WHERE category = ?'
                params.append(category)
            
            sql_query += ' ORDER BY weight DESC' # Los criterios con mayor peso primero.
            
            cursor.execute(sql_query, params)
            
            criteria = []
            for row in cursor.fetchall():
                # Convertimos las filas de la DB a diccionarios.
                criteria.append({
                    "id": row[0],
                    "name": row[1],
                    "weight": row[2],
                    "description": row[3],
                    "examples_positive": json.loads(row[4]),
                    "examples_negative": json.loads(row[5]),
                    "category": row[6]
                })
            
            conn.close()
            return criteria
            
        except Exception as e:
            logger.error(f"Fallo al recuperar criterios de evaluación: {e}")
            return []
    
    def add_code_example(self, component: str, language: str, code: str,
                        description: str, tags: List[str]) -> str:
        """
        Añade un ejemplo de código a la base de datos SQLite.
        Esto es la base del "Agentic RAG simple" para que el sistema pueda buscar código.
        Retorna el ID del ejemplo creado, o una cadena vacía si falla.
        """
        try:
            # Generamos un ID único para el ejemplo de código.
            example_id = hashlib.md5(f"{component}{code[:50]}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
            
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Insertamos el nuevo ejemplo de código.
            cursor.execute('''
                INSERT INTO code_examples (id, component, language, code, description, tags, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                example_id,
                component,
                language,
                code,
                description,
                json.dumps(tags), # Las tags se guardan como JSON dentro de un campo de texto.
                datetime.utcnow().isoformat() + 'Z'
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Ejemplo de código añadido: {example_id} para componente '{component}'")
            self.record_action(f"CodeExample: Ejemplo añadido para '{component}'", details={"example_id": example_id}, agent="ContextSentinel", component="context_sentinel")
            
            return example_id
            
        except Exception as e:
            logger.error(f"Fallo al añadir ejemplo de código: {e}")
            return ""
    
    def search_code_examples(self, query: str, component: Optional[str] = None,
                           language: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Busca ejemplos de código relevantes en la base de datos usando coincidencias parciales (LIKE).
        Es una forma simple de Agentic RAG.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Construimos la consulta SQL para buscar en código, descripción o tags.
            sql_query = '''
                SELECT id, component, language, code, description, tags, timestamp
                FROM code_examples
                WHERE (code LIKE ? OR description LIKE ? OR tags LIKE ?)
            '''
            params = [f'%{query}%', f'%{query}%', f'%{query}%'] # Añadimos comodines % para búsqueda parcial.
            
            # Filtros opcionales por componente y lenguaje.
            if component:
                sql_query += ' AND component = ?'
                params.append(component)
            
            if language:
                sql_query += ' AND language = ?'
                params.append(language)
            
            sql_query += ' ORDER BY timestamp DESC LIMIT ?' # Ordenamos por los más recientes y limitamos.
            params.append(limit)
            
            cursor.execute(sql_query, params)
            
            examples = []
            for row in cursor.fetchall():
                # Convertimos las filas de la DB a diccionarios.
                examples.append({
                    "id": row[0],
                    "component": row[1],
                    "language": row[2],
                    "code": row[3],
                    "description": row[4],
                    "tags": json.loads(row[5]), # Las tags se deserializan de JSON.
                    "timestamp": row[6]
                })
            
            conn.close()
            return examples
            
        except Exception as e:
            logger.error(f"Fallo al buscar ejemplos de código: {e}")
            return []
    
    def get_criteria_for_proposal_evaluation(self) -> List[Dict[str, Any]]:
        """
        Devuelve específicamente los criterios de evaluación categorizados para "proposal_evaluation".
        Este es un método de conveniencia para el futuro ProposalEvaluator.
        """
        return self.get_evaluation_criteria("proposal_evaluation")
    
    def add_default_evaluation_criteria(self):
        """
        Añade un conjunto de criterios de evaluación por defecto a la base de conocimiento.
        Utiliza la lógica de "upsert" de add_evaluation_criteria para no duplicar si ya existen.
        """
        logger.info("Añadiendo/actualizando criterios de evaluación por defecto...")
        criteria = [
            {
                "name": "Claridad y Especificidad",
                "weight": 0.20,
                "description": "¿La propuesta es fácil de entender y describe los cambios de forma concreta?",
                "examples_positive": ["Define ATR period=X, tp_multiplier=Y", "Ajustar __init__ y label_events"],
                "examples_negative": ["Mejorar barreras", "Hacer que sea dinámico"],
                "category": "proposal_evaluation"
            },
            {
                "name": "Alineación Arquitectónica",
                "weight": 0.25,
                "description": "¿Se alinea con los principios de Capas Canónicas y Hive Logic de Aipha_0.4.1? ¿Respeta la Capa 3?",
                "examples_positive": ["Propone cambios solo en Capa 3", "Usa librerías permitidas"],
                "examples_negative": ["Sugiere modificar Capa 1 directamente sin justificación"],
                "category": "proposal_evaluation"
            },
            {
                "name": "Viabilidad Técnica",
                "weight": 0.20,
                "description": "¿Es implementable con las herramientas existentes (pandas, pandas-ta)? ¿Necesita dependencias nuevas y justificadas?",
                "examples_positive": ["Usa pandas-ta para ATR", "Detalla cómo se integrarían los parámetros"],
                "examples_negative": ["Propone una librería de ML compleja para un cambio simple"],
                "category": "proposal_evaluation"
            }
            # Se pueden añadir más criterios aquí.
        ]
        
        for criterion in criteria:
            self.add_evaluation_criteria(
                name=criterion["name"],
                weight=criterion["weight"],
                description=criterion["description"],
                examples_positive=criterion["examples_positive"],
                examples_negative=criterion["examples_negative"],
                category=criterion["category"]
            )
        logger.info("Criterios de evaluación por defecto añadidos/actualizados.")
    
    def verify_knowledge_base_integrity(self) -> bool:
        """
        Verifica la integridad de la base de conocimiento SQLite.
        Comprueba que las tablas existan y que la base de datos no esté corrupta.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # 1. Comprobamos que las tablas requeridas existan.
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['knowledge_entries', 'evaluation_criteria', 'code_examples']
            for table in required_tables:
                if table not in tables:
                    logger.error(f"Verificación de integridad fallida: Falta tabla requerida: {table}")
                    return False
            
            # 2. Realizamos una verificación de integridad de la base de datos completa.
            # PRAGMA integrity_check es una verificación exhaustiva de la consistencia interna de SQLite.
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            if result and result[0] != 'ok':
                logger.error(f"Verificación de integridad de la base de datos fallida en {self.db_file.name}: {result[0]}")
                return False
            
            conn.close()
            logger.info("Verificación de integridad de la base de conocimiento completada exitosamente.")
            return True
            
        except Exception as e:
            logger.error(f"La verificación de integridad de la base de conocimiento falló: {e}")
            return False


# --- Bloque de ejecución de ejemplo para pruebas independientes ---
if __name__ == "__main__":
    logger.info("--- Ejecutando demostración independiente de context_sentinel.py ---")
    
    # Preparamos la configuración y NO limpiamos el storage_root para ver coexistencia.
    config_file_path = Path("config.yaml")
    config = load_config(config_file_path) 
    
    storage_root = Path(config['system']['storage_root'])
    if not storage_root.exists():
        storage_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio de almacenamiento '{storage_root}' creado.")

    # Inicializamos ContextSentinel con la configuración.
    sentinel = ContextSentinel(config)
    
    logger.info(f"Estado global inicial: {sentinel.get_global_state()}")
    logger.info(f"Entradas en historial de acciones: {len(sentinel.get_action_history())}")
    
    # 1. Añadimos criterios de evaluación por defecto (usando la lógica upsert).
    sentinel.add_default_evaluation_criteria()
    
    # 2. Añadimos una entrada de conocimiento.
    entry_id = sentinel.add_knowledge_entry(
        category="architecture",
        title="Capas de Aipha_1.1",
        content="Aipha_1.1 se estructura en 5 capas principales: Núcleo (redesignHelper), Context Sentinel, Data Preprocessor, Trading Flow, y Oracles.",
        metadata={"importance": "high", "layer": "architecture"}
    )
    logger.info(f"Entrada de conocimiento añadida: {entry_id}")
    
    # 3. Añadimos un ejemplo de código.
    example_id = sentinel.add_code_example(
        component="redesign_helper",
        language="python",
        code="""
        # Simulación de aplicación de cambios
        global_state = self._load_json(self.global_state_file)
        global_state[f"last_applied_proposal_{proposal.proposal_id}"] = proposal.to_dict()
        self._save_json(self.global_state_file, global_state)
        """,
        description="Ejemplo de cómo el sistema aplica cambios al estado global durante una actualización atómica.",
        tags=["atomic", "update", "protocol", "example"]
    )
    logger.info(f"Ejemplo de código añadido: {example_id}")
    
    # 4. Buscamos ejemplos de código.
    examples = sentinel.search_code_examples("atomic update", language="python")
    logger.info(f"Encontrados {len(examples)} ejemplos de código para 'atomic update'.")
    
    # 5. Obtenemos criterios de evaluación.
    criteria = sentinel.get_evaluation_criteria(category="proposal_evaluation")
    logger.info(f"Criterios de evaluación para propuestas: {len(criteria)} items.")
    
    # 6. Verificamos la integridad final.
    logger.info(f"Integridad de la base de conocimiento final: {sentinel.verify_knowledge_base_integrity()}")
    
    logger.info("--- Demostración independiente de context_sentinel.py completada. ---")