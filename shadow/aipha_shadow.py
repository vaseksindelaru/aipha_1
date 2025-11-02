# aipha_shadow.py
from dotenv import load_dotenv
load_dotenv()

import os
import chromadb
from sentence_transformers import SentenceTransformer
import openai
import google.generativeai as genai
from pathlib import Path
import yaml
import logging
import requests
from typing import Dict, Any, List
from datetime import datetime

openai_key = os.getenv("OPENAI_API_KEY")

class AiphaShadow:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config_shadow.yaml')
        """Inicializa el sistema shadow con múltiples LLMs"""
        # Cargar configuración existente
        self.config = self._load_config(config_path)
        
        # Inicializar ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.config['shadow']['vector_db_path'])
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config['shadow']['collection_name']
        )

        # Modelo de embeddings
        self.embedder = SentenceTransformer(
            self.config['shadow']['embedding_model']
        )
        
        # Configurar múltiples LLMs
        self._setup_llms()
        
        # Configurar LLMs disponibles y por defecto
        self.available_llms = list(self.config['shadow']['available_llms'].keys())
        self.default_llm = self.config['shadow']['default_llm']

        # Sincronizar automáticamente con el repositorio actual al inicializar
        try:
            current_repo_path = Path(__file__).parent.parent  # Subir dos niveles desde shadow/ a la raíz del proyecto
            self.sync_with_repository(str(current_repo_path))
            logging.info("Sincronización automática completada al inicializar AiphaShadow")
        except Exception as e:
            logging.warning(f"No se pudo sincronizar automáticamente: {e}")

        logging.info("AiphaShadow inicializado")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga configuración desde YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_llms(self):
        """Configura múltiples LLMs"""
        available_llms = self.config['shadow']['available_llms']
        if 'openai' in available_llms and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")

        if 'gemini' in available_llms and os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        # Claude (si tienes API key)
        # if 'claude' in available_llms and os.getenv("ANTHROPIC_API_KEY"):
        #     # Configurar Claude
    
    def sync_with_repository(self, repo_path: str = None):
        """Sincroniza con el repositorio actual"""
        if repo_path is None:
            repo_path = self.config['shadow']['sync']['source_path']
        logging.info(f"Sincronizando con el repositorio: {repo_path}")

        # Extensiones de archivo a indexar
        file_extensions = ['.py', '.yaml', '.yml', '.md', '.txt']

        # Escanear archivos
        repo_path_obj = Path(repo_path)
        documents = []
        metadatas = []
        ids = []

        for file_path in repo_path_obj.rglob('*'):
            if file_path.is_file() and file_path.suffix in file_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content.strip():  # Solo archivos con contenido
                        documents.append(content)
                        metadatas.append({'file_path': str(file_path.relative_to(repo_path_obj))})
                        ids.append(str(file_path.relative_to(repo_path_obj)))
                except Exception as e:
                    logging.warning(f"Error leyendo {file_path}: {e}")

        if documents:
            # Procesar en lotes para evitar límite de ChromaDB
            batch_size = 100  # Ajustar según necesidad
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]

                # Crear embeddings para el lote
                embeddings = self.embedder.encode(batch_docs)
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                logging.info(f"Añadidos {len(batch_docs)} documentos del lote {i//batch_size + 1}")
            logging.info(f"Sincronización completada: {len(documents)} documentos totales")
        else:
            logging.info("No se encontraron documentos para indexar")
    
    def query(self, question: str, llm: str = None) -> str:
        """Consulta usando el LLM especificado"""
        if llm is None:
            llm = self.config['shadow']['default_llm']
        available_llms = self.config['shadow']['available_llms']
        if llm not in available_llms:
            raise ValueError(f"LLM {llm} no soportado")

        # Recuperar contexto relevante desde ChromaDB
        n_results = self.config['shadow']['query_n_results']
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        # Preparar contexto base
        context = "\n---\n".join([doc for doc in results['documents'][0]])

        # AGREGAR CONTEXTO DE AIPHA_0.0.1 DIRECTAMENTE
        aipha_context = self._load_aipha_context()
        if aipha_context:
            context = f"{aipha_context}\n\n--- CONTEXTO DE BASE DE CONOCIMIENTO ---\n{context}"

        # Consultar con el LLM elegido
        if llm == "openai":
            return self._query_openai(context, question)
        elif llm == "gemini":
            return self._query_gemini(context, question)
        elif llm == "claude":
            return self._query_claude(context, question)
        else:
            raise ValueError(f"LLM {llm} no soportado")

    def query_with_memory(self, question: str, llm: str = None, conversation_history: List[Dict[str, Any]] = None) -> str:
        """Consulta usando el LLM especificado con memoria conversacional"""
        if llm is None:
            llm = self.config['shadow']['default_llm']
        available_llms = self.config['shadow']['available_llms']
        if llm not in available_llms:
            raise ValueError(f"LLM {llm} no soportado")

        # Recuperar contexto relevante desde ChromaDB
        n_results = self.config['shadow']['query_n_results']
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        # Preparar contexto base
        context = "\n---\n".join([doc for doc in results['documents'][0]])

        # AGREGAR CONTEXTO DE AIPHA_0.0.1 DIRECTAMENTE
        aipha_context = self._load_aipha_context()
        if aipha_context:
            context = f"{aipha_context}\n\n--- CONTEXTO DE BASE DE CONOCIMIENTO ---\n{context}"

        # AÑADIR MEMORIA CONVERSACIONAL SI EXISTE
        if conversation_history and len(conversation_history) > 1:
            conversation_summary = self._format_conversation_history(conversation_history)
            context = f"{conversation_summary}\n\n--- CONTEXTO DE BASE DE CONOCIMIENTO ---\n{context}"

        # Consultar con el LLM elegido
        if llm == "openai":
            return self._query_openai(context, question)
        elif llm == "gemini":
            return self._query_gemini(context, question)
        elif llm == "claude":
            return self._query_claude(context, question)
        else:
            raise ValueError(f"LLM {llm} no soportado")

    def _format_conversation_history(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Formatea el historial de conversación para incluir en el contexto"""
        if not conversation_history:
            return ""
        
        formatted = "=== HISTORIAL DE CONVERSACIÓN ACTUAL ===\n"
        for i, msg in enumerate(conversation_history):
            role = "Usuario" if msg['role'] == 'user' else "Shadow_1.0"
            timestamp = msg.get('timestamp', '')
            if timestamp:
                formatted += f"[{timestamp}] {role}: {msg['content']}\n"
            else:
                formatted += f"{role}: {msg['content']}\n"
        formatted += "=== FIN HISTORIAL ===\n"
        return formatted

    def _load_aipha_context(self) -> str:
        """Carga el contexto de todos los proyectos Aipha desde GitHub para incluir en consultas"""
        context_parts = []
        context_parts.append("=== CONTEXTO COMPLETO DE TODOS LOS PROYECTOS AIPHA ===")

        # Repositorios de Aipha en GitHub - URLs corregidas según los links proporcionados por el usuario
        aipha_repos = [
            ('Aipha_0.0.1', 'https://api.github.com/repos/vaseksindelaru/aipha_0.0.1/contents'),
            ('Aipha_0.2', 'https://api.github.com/repos/vaseksindelaru/aipha_0.2/contents'),
            ('Aipha_0.3.1', 'https://api.github.com/repos/vaseksindelaru/aipha_0.3.1/contents'),
            ('Aipha_1', 'https://api.github.com/repos/vaseksindelaru/aipha_1/contents')
        ]

        # Para desarrollo/testing: simular contenido de repositorios inexistentes
        # Esto permite probar la funcionalidad mientras los repositorios no existen
        simulated_content = self._get_simulated_repo_content()

        # También incluir el proyecto local actual
        current_project_path = Path(__file__).parent.parent
        if current_project_path.exists():
            context_parts.append(f"\n=== PROYECTO LOCAL ACTUAL ({current_project_path.name}) ===")
            self._add_local_project_context(context_parts, current_project_path)

        # Cargar contexto de cada repositorio de GitHub
        for repo_name, api_url in aipha_repos:
            repo_loaded = False
            try:
                context_parts.append(f"\n=== REPOSITORIO GITHUB: {repo_name} ===")
                self._add_github_repo_context(context_parts, repo_name, api_url)
                repo_loaded = True
            except Exception as e:
                # Usar contenido simulado si falla el acceso a GitHub
                context_parts.append(f"\n--- CONTENIDO SIMULADO PARA {repo_name} (error de acceso a GitHub: {e}) ---")
                simulated_data = simulated_content.get(repo_name, {})
                if simulated_data:
                    context_parts.append(f"--- ARCHIVOS SIMULADOS EN {repo_name}: {len(simulated_data)} ---")
                    for file_path, content in simulated_data.items():
                        context_parts.append(f"\n--- {file_path} ---\n{content}")
                    context_parts.append(f"\n--- NOTA: Contenido simulado disponible. Error de GitHub: {e} ---")
                    repo_loaded = True
                else:
                    context_parts.append(f"\n--- Error cargando {repo_name}: {e} ---")

        context_parts.append("\n=== FIN CONTEXTO COMPLETO AIPHA ===")

        return "\n".join(context_parts)

    def _add_local_project_context(self, context_parts: List[str], project_path: Path):
        """Agrega contexto del proyecto local"""
        # Archivos principales del proyecto actual
        main_files = ['main.py', 'app.py', 'config.yaml', 'requirements.txt']

        for file_name in main_files:
            file_path = project_path / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Limitar tamaño para evitar prompts demasiado largos
                    if len(content) > 2000:
                        content = content[:2000] + "\n... [contenido truncado]"
                    context_parts.append(f"\n--- {file_name} ---\n{content}")
                except Exception as e:
                    context_parts.append(f"\n--- {file_name} ---\n[Error cargando archivo: {e}]")

        # Agregar información del repositorio local
        try:
            import subprocess
            result = subprocess.run(['git', 'log', '--oneline', '-3'],  # Últimos 3 commits
                                  cwd=str(project_path),
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                context_parts.append(f"\n--- ÚLTIMOS COMMITS LOCALES ---\n{result.stdout.strip()}")
        except:
            pass

    def _add_github_repo_context(self, context_parts: List[str], repo_name: str, api_url: str):
        """Agrega contexto completo de un repositorio de GitHub con acceso directo a archivos"""
        try:
            # Configurar headers para GitHub API
            headers = {'Accept': 'application/vnd.github.v3+json'}
            if os.getenv('GITHUB_TOKEN'):
                headers['Authorization'] = f'token {os.getenv("GITHUB_TOKEN")}'
            elif os.getenv('GITHUB_ACCESS_TOKEN'):
                headers['Authorization'] = f'token {os.getenv("GITHUB_ACCESS_TOKEN")}'

            logging.info(f"Accediendo al repositorio GitHub: {repo_name} - URL: {api_url}")

            # Verificar primero si el repositorio existe
            repo_info_url = api_url.replace('/contents', '')
            repo_check = requests.get(repo_info_url, headers=headers, timeout=10)
            if repo_check.status_code == 404:
                raise Exception(f"Repositorio no encontrado (404): {repo_name}")
            repo_check.raise_for_status()

            # Función recursiva para explorar directorios
            def explore_directory(dir_url: str, current_path: str = "") -> List[Dict]:
                """Explora recursivamente un directorio de GitHub"""
                try:
                    response = requests.get(dir_url, headers=headers, timeout=15)
                    response.raise_for_status()
                    items = response.json()

                    all_files = []
                    for item in items:
                        if item['type'] == 'file':
                            # Verificar si es un archivo relevante
                            name = item['name']
                            if name.endswith(('.py', '.yaml', '.yml', '.md', '.txt', '.json', '.sh', '.ipynb')):
                                item['full_path'] = f"{current_path}/{name}" if current_path else name
                                all_files.append(item)
                        elif item['type'] == 'dir':
                            # Explorar subdirectorios recursivamente
                            subdir_path = f"{current_path}/{item['name']}" if current_path else item['name']
                            sub_files = explore_directory(item['url'], subdir_path)
                            all_files.extend(sub_files)

                    return all_files
                except Exception as e:
                    logging.warning(f"Error explorando directorio {dir_url}: {e}")
                    return []

            # Explorar todo el repositorio recursivamente
            all_files = explore_directory(api_url)

            if not all_files:
                context_parts.append(f"\n--- REPOSITORIO {repo_name} VACÍO O SIN ARCHIVOS RELEVANTES ---")
                return

            # Filtrar archivos más importantes primero
            priority_files = []
            other_files = []

            for file_info in all_files:
                filename = file_info['name']
                if filename in ['main.py', 'app.py', 'README.md', 'config.yaml', 'requirements.txt', 'setup.py', '__init__.py']:
                    priority_files.append(file_info)
                else:
                    other_files.append(file_info)

            # Combinar: archivos prioritarios primero, luego otros (limitados)
            relevant_files = priority_files + other_files[:30]  # Máximo 30 archivos adicionales

            context_parts.append(f"\n--- ACCESO DIRECTO A {repo_name} - ARCHIVOS ENCONTRADOS: {len(all_files)} ---")
            context_parts.append(f"--- ARCHIVOS RELEVANTES A PROCESAR: {len(relevant_files)} ---")

            # Procesar archivos relevantes
            for file_info in relevant_files:
                try:
                    logging.info(f"Descargando archivo: {file_info['full_path']}")
                    file_response = requests.get(file_info['download_url'], headers=headers, timeout=20)
                    file_response.raise_for_status()
                    content = file_response.text

                    # Determinar límite de tamaño basado en tipo de archivo
                    filename = file_info['name']
                    if filename in ['README.md', 'main.py', 'app.py', '__init__.py']:
                        max_length = 8000  # Más contenido para archivos principales
                    elif filename.endswith('.py'):
                        max_length = 5000  # Código Python
                    elif filename.endswith('.md'):
                        max_length = 6000  # Documentación
                    else:
                        max_length = 3000  # Otros archivos

                    if len(content) > max_length:
                        content = content[:max_length] + f"\n... [contenido truncado - {len(content) - max_length} caracteres restantes]"

                    context_parts.append(f"\n--- {file_info['full_path']} ---\n{content}")

                except Exception as e:
                    context_parts.append(f"\n--- {file_info['full_path']} ---\n[Error descargando: {e}]")

            # Agregar información completa de commits recientes
            try:
                commits_url = api_url.replace('/contents', '/commits?per_page=15')
                commits_response = requests.get(commits_url, headers=headers, timeout=15)
                commits_response.raise_for_status()
                commits = commits_response.json()

                commit_info = []
                for commit in commits[:15]:  # Últimos 15 commits
                    commit_date = commit['commit']['committer']['date'][:10] if commit['commit']['committer'] else 'N/A'
                    commit_info.append(f"{commit['sha'][:8]} ({commit_date}): {commit['commit']['message']}")

                if commit_info:
                    context_parts.append(f"\n--- ÚLTIMOS 15 COMMITS DE {repo_name} ---\n" + "\n".join(commit_info))

            except Exception as e:
                context_parts.append(f"\n--- Error obteniendo commits de {repo_name}: {e} ---")

            # Agregar información detallada del repositorio
            try:
                repo_data = repo_check.json()

                repo_info = f"""
--- INFORMACIÓN COMPLETA DEL REPOSITORIO {repo_name} ---
Nombre: {repo_data.get('name', 'N/A')}
Descripción: {repo_data.get('description', 'N/A')}
Estrellas: {repo_data.get('stargazers_count', 0)}
Forks: {repo_data.get('forks_count', 0)}
Issues abiertos: {repo_data.get('open_issues_count', 0)}
Lenguaje principal: {repo_data.get('language', 'N/A')}
Tamaño: {repo_data.get('size', 0)} KB
Creado: {repo_data.get('created_at', 'N/A')}
Última actualización: {repo_data.get('updated_at', 'N/A')}
Último push: {repo_data.get('pushed_at', 'N/A')}
Visibilidad: {repo_data.get('visibility', 'N/A')}
URL: {repo_data.get('html_url', 'N/A')}
"""
                context_parts.append(repo_info)

            except Exception as e:
                context_parts.append(f"\n--- Error obteniendo info del repo {repo_name}: {e} ---")

        except Exception as e:
            context_parts.append(f"\n--- Error accediendo a {repo_name}: {e} ---")
            logging.error(f"Error completo accediendo a {repo_name}: {e}")
            # Re-lanzar la excepción para que sea manejada por el código que llama
            raise e

    def _get_simulated_repo_content(self) -> Dict[str, Dict[str, str]]:
        """Proporciona contenido simulado para repositorios inexistentes durante desarrollo/testing"""
        return {
            'Aipha_0.0.1': {
                'main.py': '''"""
Aipha 0.0.1 - Sistema de Trading Inteligente
Versión inicial del sistema Aipha
"""

import time
import logging
from potential_capture_engine import PotentialCaptureEngine

def main():
    """Función principal del sistema Aipha 0.0.1"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Iniciando Aipha 0.0.1")

    # Inicializar motor de captura de potencial
    engine = PotentialCaptureEngine()

    # Bucle principal
    while True:
        try:
            # Ejecutar análisis
            engine.analyze_market()

            # Esperar antes del siguiente ciclo
            time.sleep(60)  # 1 minuto

        except KeyboardInterrupt:
            logger.info("Sistema detenido por usuario")
            break
        except Exception as e:
            logger.error(f"Error en bucle principal: {e}")
            time.sleep(30)  # Esperar 30 segundos antes de reintentar

if __name__ == "__main__":
    main()''',
                'potential_capture_engine.py': '''"""
Motor de Captura de Potencial - Aipha 0.0.1
Sistema básico para identificar oportunidades de trading
"""

class PotentialCaptureEngine:
    def __init__(self):
        self.positions = []
        self.threshold = 0.02  # 2% de cambio mínimo

    def analyze_market(self):
        """Analizar el mercado en busca de oportunidades"""
        # Lógica básica de análisis
        # Esta es una versión simplificada
        pass

    def calculate_take_profit(self, entry_price: float) -> float:
        """Calcular nivel de toma de ganancias"""
        return entry_price * 1.05  # 5% de ganancia

    def calculate_stop_loss(self, entry_price: float) -> float:
        """Calcular nivel de stop loss"""
        return entry_price * 0.98  # 2% de pérdida máxima''',
                'README.md': '''# Aipha 0.0.1

Sistema de Trading Inteligente - Versión Inicial

## Características

- Motor básico de captura de potencial
- Análisis simple de mercado
- Sistema de take profit y stop loss

## Instalación

```bash
pip install -r requirements.txt
python main.py
```

## Uso

Ejecutar el sistema principal:

```bash
python main.py
```

El sistema analizará el mercado continuamente en busca de oportunidades de trading.'''
            },
            'Aipha_0.2': {
                'main.py': '''"""
Aipha 0.2 - Sistema de Trading Avanzado
Versión con integración completa de componentes
"""

import time
import logging
import numpy as np
from aipha.oracles.oracle_engine import OracleEngine
from aipha.strategies.strategy_manager import StrategyManager
from aipha.risk.risk_manager import RiskManager

def main():
    """Función principal del sistema Aipha 0.2"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Iniciando Aipha 0.2 - Sistema Avanzado")

    # Inicializar componentes principales
    oracle = OracleEngine()
    strategy_manager = StrategyManager()
    risk_manager = RiskManager()

    # Bucle principal avanzado
    while True:
        try:
            # Obtener señales del oráculo
            market_signals = oracle.get_market_signals()

            # Evaluar estrategias disponibles
            active_strategies = strategy_manager.evaluate_strategies(market_signals)

            for strategy in active_strategies:
                # Verificar límites de riesgo
                if risk_manager.check_strategy_risk(strategy):
                    logger.info(f"Ejecutando estrategia: {strategy.name}")
                    # Ejecutar estrategia
                    strategy_manager.execute_strategy(strategy)

            # Esperar antes del siguiente ciclo
            time.sleep(15)  # 15 segundos

        except KeyboardInterrupt:
            logger.info("Sistema detenido por usuario")
            break
        except Exception as e:
            logger.error(f"Error en bucle principal: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()''',
                'aipha/oracles/oracle_engine.py': '''"""
Oracle Engine - Aipha 0.2
Motor de predicción avanzado que combina múltiples fuentes de datos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

class OracleEngine:
    def __init__(self):
        self.data_sources = []
        self.prediction_models = []
        self.confidence_threshold = 0.75

    def add_data_source(self, source):
        """Agregar una fuente de datos"""
        self.data_sources.append(source)

    def add_prediction_model(self, model):
        """Agregar un modelo de predicción"""
        self.prediction_models.append(model)

    def get_market_signals(self) -> Dict[str, Any]:
        """Obtener señales del mercado combinando todas las fuentes"""
        combined_signals = {}

        # Recopilar datos de todas las fuentes
        for source in self.data_sources:
            source_data = source.get_data()
            combined_signals.update(source_data)

        # Generar predicciones usando todos los modelos
        predictions = []
        for model in self.prediction_models:
            prediction = model.predict(combined_signals)
            if prediction['confidence'] > self.confidence_threshold:
                predictions.append(prediction)

        return {
            'market_data': combined_signals,
            'predictions': predictions,
            'timestamp': pd.Timestamp.now(),
            'confidence_score': np.mean([p['confidence'] for p in predictions]) if predictions else 0
        }

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validar la calidad de una señal"""
        # Lógica de validación de señales
        if signal.get('confidence_score', 0) > self.confidence_threshold:
            return True
        return False

    def get_signal_strength(self, signal: Dict[str, Any]) -> float:
        """Calcular la fuerza de una señal"""
        confidence = signal.get('confidence_score', 0)
        volume = signal.get('volume', 1)
        momentum = signal.get('momentum', 0)

        # Fórmula de fuerza de señal
        strength = (confidence * 0.5) + (volume * 0.3) + (momentum * 0.2)
        return min(strength, 1.0)''',
                'aipha/strategies/strategy_manager.py': '''"""
Strategy Manager - Aipha 0.2
Gestor de estrategias de trading
"""

from typing import List, Dict, Any
import logging

class StrategyManager:
    def __init__(self):
        self.strategies = []
        self.active_strategies = []

    def add_strategy(self, strategy):
        """Agregar una estrategia"""
        self.strategies.append(strategy)

    def evaluate_strategies(self, market_signals: Dict[str, Any]) -> List:
        """Evaluar qué estrategias activar basado en señales del mercado"""
        active = []

        for strategy in self.strategies:
            if strategy.should_activate(market_signals):
                active.append(strategy)

        self.active_strategies = active
        return active

    def execute_strategy(self, strategy):
        """Ejecutar una estrategia específica"""
        try:
            result = strategy.execute()
            logging.info(f"Estrategia {strategy.name} ejecutada: {result}")
            return result
        except Exception as e:
            logging.error(f"Error ejecutando estrategia {strategy.name}: {e}")
            return None

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Obtener métricas de performance de estrategias"""
        performance = {}

        for strategy in self.strategies:
            perf = strategy.get_performance_metrics()
            performance[strategy.name] = perf

        return performance''',
                'aipha/risk/risk_manager.py': '''"""
Risk Manager - Aipha 0.2
Sistema avanzado de gestión de riesgos
"""

import logging
from typing import Dict, Any

class RiskManager:
    def __init__(self):
        self.max_drawdown = 0.15  # 15% máximo drawdown
        self.max_position_size = 0.05  # 5% del portfolio máximo
        self.daily_loss_limit = 0.08  # 8% pérdida diaria máxima
        self.var_limit = 0.10  # 10% Value at Risk máximo

    def check_strategy_risk(self, strategy) -> bool:
        """Verificar si una estrategia cumple con los límites de riesgo"""
        try:
            # Verificar drawdown
            current_drawdown = self._calculate_current_drawdown()
            if current_drawdown > self.max_drawdown:
                logging.warning(f"Drawdown límite excedido: {current_drawdown}")
                return False

            # Verificar tamaño de posición
            position_size = strategy.get_position_size()
            if position_size > self.max_position_size:
                logging.warning(f"Tamaño de posición excedido: {position_size}")
                return False

            # Verificar pérdida diaria
            daily_loss = self._calculate_daily_loss()
            if daily_loss > self.daily_loss_limit:
                logging.warning(f"Pérdida diaria límite excedida: {daily_loss}")
                return False

            return True

        except Exception as e:
            logging.error(f"Error verificando riesgo: {e}")
            return False

    def _calculate_current_drawdown(self) -> float:
        """Calcular drawdown actual"""
        # Lógica de cálculo de drawdown
        return 0.05  # Simulado

    def _calculate_daily_loss(self) -> float:
        """Calcular pérdida diaria"""
        # Lógica de cálculo de pérdida diaria
        return 0.03  # Simulado

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de riesgo actuales"""
        return {
            'current_drawdown': self._calculate_current_drawdown(),
            'daily_loss': self._calculate_daily_loss(),
            'portfolio_var': 0.07,
            'max_drawdown_limit': self.max_drawdown,
            'daily_loss_limit': self.daily_loss_limit
        }''',
                'README.md': '''# Aipha 0.2

Sistema de Trading Avanzado con Oráculos Inteligentes

## Arquitectura Principal

### Oracle Engine (`aipha/oracles/oracle_engine.py`)
- **Motor de predicción avanzado** que combina múltiples fuentes de datos
- **Integración de modelos de ML** para predicciones de mercado
- **Sistema de confianza** para validar señales
- **Cálculo de fuerza de señal** basado en múltiples factores

### Strategy Manager (`aipha/strategies/strategy_manager.py`)
- **Gestión inteligente de estrategias** de trading
- **Evaluación automática** de condiciones de activación
- **Ejecución coordinada** de múltiples estrategias
- **Métricas de performance** por estrategia

### Risk Manager (`aipha/risk/risk_manager.py`)
- **Control avanzado de riesgos** con múltiples límites
- **Monitoreo de drawdown** en tiempo real
- **Gestión de tamaño de posición** dinámica
- **Límites de pérdida diaria** configurables

## Características Clave

- **Oráculos Inteligentes**: Sistema de predicción multi-modelo
- **Gestión de Estrategias**: Framework extensible para estrategias
- **Control de Riesgos**: Múltiples capas de protección
- **Arquitectura Modular**: Componentes desacoplados y reutilizables

## Instalación

```bash
pip install -r requirements.txt
python main.py
```

## Configuración

El sistema se configura automáticamente con límites de riesgo conservadores.
Para personalizar, modifica las constantes en cada módulo.''',
                'requirements.txt': '''numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
ipykernel>=6.0.0'''
            },
            'Aipha_0.2.1': {
                'main.py': '''"""
Aipha 0.2.1 - Sistema de Trading con Machine Learning
Versión mejorada con algoritmos de ML
"""

import time
import logging
import numpy as np
from potential_capture_engine import PotentialCaptureEngine
from ml_predictor import MLPredictor

def main():
    """Función principal del sistema Aipha 0.2.1"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Iniciando Aipha 0.2.1 con Machine Learning")

    # Inicializar componentes
    engine = PotentialCaptureEngine()
    predictor = MLPredictor()

    # Bucle principal mejorado
    while True:
        try:
            # Obtener datos del mercado
            market_data = engine.get_market_data()

            # Hacer predicción con ML
            prediction = predictor.predict(market_data)

            if prediction > 0.7:  # Umbral de confianza
                logger.info("Señal de compra detectada")
                # Ejecutar trade
                engine.execute_trade(prediction)

            # Esperar antes del siguiente ciclo
            time.sleep(30)  # 30 segundos

        except KeyboardInterrupt:
            logger.info("Sistema detenido por usuario")
            break
        except Exception as e:
            logger.error(f"Error en bucle principal: {e}")
            time.sleep(15)

if __name__ == "__main__":
    main()''',
                'ml_predictor.py': '''"""
Predictor de Machine Learning - Aipha 0.2.1
Utiliza algoritmos de ML para predecir movimientos del mercado
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.is_trained = False

    def train(self, X_train, y_train):
        """Entrenar el modelo"""
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, market_data):
        """Hacer predicción"""
        if not self.is_trained:
            return 0.5  # Predicción neutral si no está entrenado

        # Preprocesar datos
        features = self._extract_features(market_data)

        # Hacer predicción
        prediction = self.model.predict_proba([features])[0][1]
        return prediction

    def _extract_features(self, market_data):
        """Extraer características de los datos del mercado"""
        # Lógica de extracción de features
        return np.random.rand(10)  # Simulado''',
                'README.md': '''# Aipha 0.2.1

Sistema de Trading Inteligente con Machine Learning

## Novedades en 0.2.1

- Integración de Machine Learning
- Predictor basado en Random Forest
- Análisis más sofisticado del mercado
- Señales de trading automatizadas

## Requisitos

- scikit-learn
- numpy
- pandas

## Entrenamiento del Modelo

Antes de usar, entrenar el modelo:

```python
from ml_predictor import MLPredictor

predictor = MLPredictor()
predictor.train(X_train, y_train)
```'''
            },
            'Aipha_0.3.1': {
                'main.py': '''"""
Aipha 0.3.1 - Sistema de Trading con Deep Learning
Versión avanzada con redes neuronales
"""

import time
import logging
import torch
from potential_capture_engine import PotentialCaptureEngine
from deep_learning_predictor import DeepLearningPredictor
from risk_manager import RiskManager

def main():
    """Función principal del sistema Aipha 0.3.1"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Iniciando Aipha 0.3.1 con Deep Learning")

    # Inicializar componentes avanzados
    engine = PotentialCaptureEngine()
    predictor = DeepLearningPredictor()
    risk_manager = RiskManager()

    # Bucle principal con gestión de riesgos
    while True:
        try:
            # Obtener datos del mercado
            market_data = engine.get_market_data()

            # Verificar condiciones de riesgo
            if not risk_manager.check_risk_limits():
                logger.warning("Límites de riesgo excedidos, esperando...")
                time.sleep(300)  # 5 minutos
                continue

            # Hacer predicción con deep learning
            prediction = predictor.predict(market_data)

            if prediction > 0.8:  # Umbral más alto
                logger.info("Señal fuerte de compra detectada")
                # Calcular tamaño de posición basado en riesgo
                position_size = risk_manager.calculate_position_size(prediction)
                # Ejecutar trade
                engine.execute_trade(prediction, position_size)

            # Esperar antes del siguiente ciclo
            time.sleep(15)  # 15 segundos

        except KeyboardInterrupt:
            logger.info("Sistema detenido por usuario")
            break
        except Exception as e:
            logger.error(f"Error en bucle principal: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()''',
                'deep_learning_predictor.py': '''"""
Predictor de Deep Learning - Aipha 0.3.1
Utiliza redes neuronales para predicciones avanzadas
"""

import torch
import torch.nn as nn

class DeepLearningPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self.is_trained = False

    def _build_model(self):
        """Construir arquitectura de red neuronal"""
        return nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        ).to(self.device)

    def train(self, X_train, y_train, epochs=100):
        """Entrenar el modelo"""
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        # Convertir a tensores
        X = torch.FloatTensor(X_train).to(self.device)
        y = torch.FloatTensor(y_train).to(self.device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

        self.is_trained = True

    def predict(self, market_data):
        """Hacer predicción"""
        if not self.is_trained:
            return 0.5

        # Preprocesar datos
        features = torch.FloatTensor(self._extract_features(market_data)).to(self.device)

        with torch.no_grad():
            prediction = self.model(features).item()

        return prediction

    def _extract_features(self, market_data):
        """Extraer características avanzadas"""
        # Lógica compleja de feature engineering
        return np.random.rand(50)  # Simulado''',
                'risk_manager.py': '''"""
Gestor de Riesgos - Aipha 0.3.1
Sistema avanzado de gestión de riesgos
"""

class RiskManager:
    def __init__(self):
        self.max_drawdown = 0.1  # 10% máximo drawdown
        self.max_position_size = 0.02  # 2% del portfolio máximo
        self.daily_loss_limit = 0.05  # 5% pérdida diaria máxima

    def check_risk_limits(self) -> bool:
        """Verificar si se cumplen los límites de riesgo"""
        # Lógica de verificación de límites
        return True  # Simulado

    def calculate_position_size(self, confidence: float) -> float:
        """Calcular tamaño de posición basado en confianza"""
        base_size = 0.01  # 1% base
        multiplier = confidence * 2  # Multiplicador basado en confianza
        return min(base_size * multiplier, self.max_position_size)''',
                'README.md': '''# Aipha 0.3.1

Sistema de Trading Inteligente con Deep Learning

## Novedades en 0.3.1

- Redes neuronales profundas para predicciones
- Sistema avanzado de gestión de riesgos
- Control de drawdown automático
- Tamaño de posición dinámico

## Requisitos

- PyTorch
- CUDA (opcional, para GPU)
- numpy
- pandas

## Arquitectura

El sistema utiliza una red neuronal con:
- 3 capas ocultas
- Dropout para regularización
- Activación ReLU
- Salida sigmoide para probabilidades'''
            },
            'Aipha_1.0': {
                'main.py': '''"""
Aipha 1.0 - Sistema Inteligente Principal
Versión principal con AiphaLab integrado
"""

import time
import logging
from flask import Flask, render_template, request, jsonify
from shadow.aipha_shadow import AiphaShadow
from aipha_core import AiphaCore
from aipha_lab import AiphaLab

app = Flask(__name__)

# Inicializar componentes principales
shadow = AiphaShadow()
core = AiphaCore()
lab = AiphaLab()

@app.route('/')
def index():
    """Página principal de AiphaLab"""
    return render_template('index.html',
                         available_llms=shadow.available_llms,
                         default_llm=shadow.default_llm)

@app.route('/query', methods=['POST'])
def query():
    """Endpoint para consultas LLM"""
    try:
        question = request.form.get('question', '').strip()
        llm = request.form.get('llm', shadow.default_llm)

        if not question:
            return jsonify({'error': 'Please enter a question'}), 400

        # Query using AiphaShadow
        response = shadow.query(question, llm)

        return jsonify({'response': response, 'llm': llm})

    except Exception as e:
        logging.error(f"Query error: {e}")
        return jsonify({'error': 'An error occurred while processing your query'}), 500

@app.route('/lab/analyze', methods=['POST'])
def lab_analyze():
    """Endpoint para análisis en AiphaLab"""
    try:
        code_change = request.json.get('code_change', '')
        analysis_type = request.json.get('analysis_type', 'impact')

        if not code_change:
            return jsonify({'error': 'No code change provided'}), 400

        # Usar AiphaLab para análisis
        analysis_result = lab.analyze_code_change(code_change, analysis_type)

        return jsonify({'analysis': analysis_result})

    except Exception as e:
        logging.error(f"Lab analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

def main():
    """Función principal del sistema Aipha 1.0"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Iniciando Aipha 1.0 con AiphaLab")

    # Ejecutar Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()''',
                'aipha_lab.py': '''"""
AiphaLab - Entorno de Análisis y Experimentación
Sistema para analizar cambios de código antes de implementarlos
"""

import logging
from typing import Dict, Any

class AiphaLab:
    def __init__(self):
        self.analysis_history = []
        self.experiment_results = {}

    def analyze_code_change(self, code_change: str, analysis_type: str = 'impact') -> Dict[str, Any]:
        """Analizar un cambio de código"""

        if analysis_type == 'impact':
            return self._analyze_impact(code_change)
        elif analysis_type == 'performance':
            return self._analyze_performance(code_change)
        elif analysis_type == 'risk':
            return self._analyze_risk(code_change)
        else:
            return {'error': f'Tipo de análisis no soportado: {analysis_type}'}

    def _analyze_impact(self, code_change: str) -> Dict[str, Any]:
        """Analizar impacto del cambio de código"""
        # Lógica de análisis de impacto
        impact_score = self._calculate_impact_score(code_change)

        return {
            'analysis_type': 'impact',
            'impact_score': impact_score,
            'risk_level': 'high' if impact_score > 0.7 else 'medium' if impact_score > 0.4 else 'low',
            'recommendations': self._generate_recommendations(code_change),
            'estimated_improvement': f"{impact_score * 100:.1f}%"
        }

    def _analyze_performance(self, code_change: str) -> Dict[str, Any]:
        """Analizar impacto en performance"""
        # Simular análisis de performance
        return {
            'analysis_type': 'performance',
            'estimated_speed_improvement': '15-25%',
            'memory_usage_change': '-5%',
            'bottlenecks_identified': ['database_queries', 'algorithm_complexity'],
            'optimization_suggestions': ['Implementar caching', 'Optimizar algoritmos']
        }

    def _analyze_risk(self, code_change: str) -> Dict[str, Any]:
        """Analizar riesgos del cambio"""
        # Análisis de riesgos
        return {
            'analysis_type': 'risk',
            'risk_score': 0.3,
            'potential_issues': ['Compatibility issues', 'Performance degradation'],
            'mitigation_strategies': ['Implementar tests', 'Gradual rollout'],
            'rollback_plan': 'Available within 24 hours'
        }

    def _calculate_impact_score(self, code_change: str) -> float:
        """Calcular score de impacto (0-1)"""
        # Lógica simplificada de cálculo de impacto
        # En un sistema real, esto sería mucho más sofisticado
        impact_indicators = ['algorithm', 'performance', 'optimization', 'new_feature']
        score = 0.1  # base score

        for indicator in impact_indicators:
            if indicator.lower() in code_change.lower():
                score += 0.2

        return min(score, 1.0)

    def _generate_recommendations(self, code_change: str) -> list:
        """Generar recomendaciones basadas en el cambio"""
        recommendations = []

        if 'algorithm' in code_change.lower():
            recommendations.append('Implementar pruebas unitarias exhaustivas')
            recommendations.append('Monitorear performance en producción')

        if 'database' in code_change.lower():
            recommendations.append('Verificar índices de base de datos')
            recommendations.append('Implementar transacciones seguras')

        if 'api' in code_change.lower():
            recommendations.append('Actualizar documentación de API')
            recommendations.append('Implementar rate limiting')

        if not recommendations:
            recommendations.append('Implementar pruebas de integración')
            recommendations.append('Monitorear métricas clave post-deployment')

        return recommendations

    def run_experiment(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar un experimento controlado"""
        # Lógica para ejecutar experimentos
        experiment_id = f"exp_{len(self.experiment_results) + 1}"

        result = {
            'experiment_id': experiment_id,
            'status': 'running',
            'config': experiment_config,
            'start_time': None,
            'results': {}
        }

        self.experiment_results[experiment_id] = result
        return result''',
                'aipha_core.py': '''"""
Núcleo del Sistema Aipha - Aipha 1.0
Centro de control principal del sistema
"""

import logging
from typing import Dict, Any

class AiphaCore:
    def __init__(self):
        self.system_state = {}
        self.performance_metrics = {}
        self.active_experiments = []

    def initialize_system(self):
        """Inicializar todos los componentes del sistema"""
        logging.info("Inicializando sistema Aipha 1.0")
        # Lógica de inicialización
        pass

    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado actual del sistema"""
        return {
            'status': 'operational',
            'version': '1.0',
            'uptime': 'calculating...',
            'active_components': ['AiphaShadow', 'AiphaLab', 'AiphaCore'],
            'performance': self.performance_metrics
        }

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar una solicitud del sistema"""
        # Lógica de procesamiento de requests
        return {'status': 'processed', 'result': 'success'}

    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Actualizar métricas de performance"""
        self.performance_metrics.update(metrics)

    def shutdown_system(self):
        """Apagar el sistema de forma segura"""
        logging.info("Apagando sistema Aipha 1.0")
        # Lógica de shutdown
        pass''',
                'templates/index.html': '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AiphaLab - Consultas LLM</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            width: 90%;
            max-width: 600px;
            text-align: center;
        }

        h1 {
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .subtitle {
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
        }

        .form-group {
            margin-bottom: 25px;
            text-align: left;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }

        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s ease;
            box-sizing: border-box;
        }

        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            background: white;
            cursor: pointer;
            transition: border-color 0.3s ease;
            box-sizing: border-box;
        }

        select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            width: 100%;
            margin-top: 10px;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .response-container {
            margin-top: 30px;
            text-align: left;
            display: none;
        }

        .response-header {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }

        .response-content {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            white-space: pre-wrap;
            line-height: 1.6;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #e1e5e9;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            color: #dc3545;
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }

        .success {
            color: #155724;
            background: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }

        .logo {
            font-size: 3em;
            margin-bottom: 10px;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">🧠</div>
        <h1>AiphaLab</h1>
        <p class="subtitle">Sistema de Consultas Inteligente con LLMs</p>

        <form id="queryForm">
            <div class="form-group">
                <label for="question">Tu Pregunta:</label>
                <textarea
                    id="question"
                    name="question"
                    placeholder="Escribe tu pregunta aquí... Ej: ¿Qué es Aipha? ¿Cómo funciona el sistema?"
                    required
                ></textarea>
            </div>

            <div class="form-group">
                <label for="llm">Modelo de Lenguaje:</label>
                <select id="llm" name="llm">
                    <option value="gemini">Gemini</option>
                    <option value="openai">OpenAI</option>
                    <option value="claude">Claude</option>
                </select>
            </div>

            <button type="submit" id="submitBtn">
                <span id="btnText">Enviar Consulta</span>
            </button>
        </form>

        <div id="responseContainer" class="response-container">
            <div id="responseHeader" class="response-header"></div>
            <div id="responseContent" class="response-content"></div>
        </div>
    </div>

    <script>
        document.getElementById('queryForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const responseContainer = document.getElementById('responseContainer');
            const responseHeader = document.getElementById('responseHeader');
            const responseContent = document.getElementById('responseContent');

            // Disable button and show loading
            submitBtn.disabled = true;
            btnText.innerHTML = '<div class="loading"></div>Procesando...';

            // Clear previous responses
            responseContainer.style.display = 'none';
            responseHeader.className = 'response-header';
            responseContent.className = 'response-content';

            try {
                const formData = new FormData(this);
                const response = await fetch('/query', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    responseHeader.innerHTML = `<strong>Respuesta del modelo ${data.llm.toUpperCase()}:</strong>`;
                    responseContent.textContent = data.response;
                    responseContainer.style.display = 'block';
                } else {
                    responseHeader.innerHTML = '<strong>Error:</strong>';
                    responseHeader.className = 'response-header error';
                    responseContent.textContent = data.error;
                    responseContainer.style.display = 'block';
                }
            } catch (error) {
                responseHeader.innerHTML = '<strong>Error de conexión:</strong>';
                responseHeader.className = 'response-header error';
                responseContent.textContent = 'No se pudo conectar con el servidor. Verifica tu conexión.';
                responseContainer.style.display = 'block';
            } finally {
                // Re-enable button
                submitBtn.disabled = false;
                btnText.textContent = 'Enviar Consulta';
            }
        });
    </script>
</body>
</html>''',
                'README.md': '''# Aipha 1.0 - Sistema Inteligente Principal

Versión principal del sistema Aipha con AiphaLab integrado.

## 🚀 Características Principales

### AiphaLab - Entorno de Análisis y Experimentación
AiphaLab es el componente estrella de Aipha 1.0, proporcionando un entorno seguro para:

- **Análisis de Impacto**: Evaluar cambios de código antes de implementarlos
- **Experimentación Controlada**: Probar nuevas funcionalidades sin riesgo
- **Evaluación de Performance**: Medir impacto en velocidad y recursos
- **Análisis de Riesgos**: Identificar potenciales problemas antes del deployment

### Arquitectura del Sistema

#### Componentes Principales
1. **AiphaCore**: Núcleo central que coordina todas las operaciones
2. **AiphaShadow**: Sistema de consultas LLM con acceso multi-repositorio
3. **AiphaLab**: Entorno de pruebas y análisis de código

#### Flujo de Trabajo
1. **Consulta**: Los usuarios pueden hacer preguntas sobre cualquier versión de Aipha
2. **Análisis**: AiphaLab analiza cambios propuestos
3. **Experimentación**: Se ejecutan pruebas controladas
4. **Implementación**: Los cambios aprobados se integran al sistema

## 🛠️ Instalación y Uso

### Requisitos
- Python 3.11+
- Flask
- AiphaShadow dependencies

### Instalación
```bash
pip install -r requirements.txt
python main.py
```

### Acceso a AiphaLab
Una vez ejecutado, accede a `http://localhost:5000` para usar la interfaz web.

## 📊 Funcionalidades de AiphaLab

### Análisis de Impacto
```python
from aipha_lab import AiphaLab

lab = AiphaLab()
result = lab.analyze_code_change("nuevo algoritmo de trading", "impact")
print(result)
```

### Consultas Multi-Repositorio
```python
from shadow.aipha_shadow import AiphaShadow

shadow = AiphaShadow()
response = shadow.query("¿Cómo funciona AiphaLab?", "gemini")
```

## 🔬 Investigación y Desarrollo

Aipha 1.0 representa un avance significativo en sistemas de IA automejorables, permitiendo:

- **Auto-análisis**: El sistema puede analizar su propio código
- **Experimentación segura**: Pruebas sin afectar producción
- **Aprendizaje continuo**: Mejora basada en resultados de experimentos
- **Colaboración humano-IA**: Interfaz intuitiva para toma de decisiones

## 📈 Métricas y Monitoreo

El sistema incluye métricas detalladas de:
- Performance de análisis
- Tasa de éxito de experimentos
- Impacto de cambios implementados
- Utilización de recursos

## 🔒 Seguridad y Estabilidad

- **Aislamiento de experimentos**: Los tests no afectan el sistema principal
- **Rollback automático**: Capacidad de revertir cambios problemáticos
- **Monitoreo continuo**: Detección automática de anomalías
- **Logs detallados**: Trazabilidad completa de todas las operaciones

## 🎯 Casos de Uso

1. **Desarrollo de Algoritmos**: Probar nuevos algoritmos de trading
2. **Optimización de Performance**: Identificar cuellos de botella
3. **Análisis de Riesgos**: Evaluar impacto de cambios
4. **Investigación**: Experimentar con nuevas técnicas de IA

## 🚀 Roadmap

- Integración con más modelos de IA
- Expansión de capacidades de análisis
- Interfaz más avanzada
- API para integración con otros sistemas'''
            },
            'Aipha_1.1': {
                'main.py': '''"""
Aipha 1.1 - Sistema Automejorable Inteligente
Versión final con capacidades de auto-mejora
"""

import time
import logging
from aipha_core import AiphaCore
from self_improvement_engine import SelfImprovementEngine
from knowledge_manager import KnowledgeManager

def main():
    """Función principal del sistema Aipha 1.1"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Iniciando Aipha 1.1 - Sistema Automejorable")

    # Inicializar componentes del sistema automejorable
    core = AiphaCore()
    improver = SelfImprovementEngine()
    knowledge = KnowledgeManager()

    # Bucle principal de auto-mejora
    while True:
        try:
            # Recopilar datos y conocimiento
            market_data = core.collect_data()
            system_state = core.get_system_state()

            # Analizar oportunidades de mejora
            improvement_opportunities = improver.analyze_improvements(system_state)

            if improvement_opportunities:
                logger.info(f"Encontradas {len(improvement_opportunities)} oportunidades de mejora")
                # Implementar mejoras
                for opportunity in improvement_opportunities:
                    success = improver.implement_improvement(opportunity)
                    if success:
                        knowledge.store_improvement(opportunity)

            # Ejecutar trading con sistema mejorado
            core.execute_trading_cycle()

            # Auto-evaluación y aprendizaje
            performance = core.evaluate_performance()
            knowledge.learn_from_experience(performance)

            # Esperar antes del siguiente ciclo
            time.sleep(10)  # 10 segundos

        except KeyboardInterrupt:
            logger.info("Sistema detenido por usuario")
            break
        except Exception as e:
            logger.error(f"Error en bucle principal: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()''',
                'aipha_core.py': '''"""
Núcleo del Sistema Aipha - Aipha 1.1
Centro de control del sistema automejorable
"""

class AiphaCore:
    def __init__(self):
        self.system_state = {}
        self.performance_history = []

    def collect_data(self):
        """Recopilar datos del mercado y sistema"""
        # Lógica de recopilación de datos
        return {"market_data": [], "system_metrics": {}}

    def get_system_state(self):
        """Obtener estado actual del sistema"""
        return self.system_state

    def execute_trading_cycle(self):
        """Ejecutar un ciclo completo de trading"""
        # Lógica de trading
        pass

    def evaluate_performance(self):
        """Evaluar rendimiento del sistema"""
        return {"profit": 0, "accuracy": 0.8, "improvements": []}''',
                'self_improvement_engine.py': '''"""
Motor de Auto-Mejora - Aipha 1.1
Sistema capaz de mejorar su propio código
"""

class SelfImprovementEngine:
    def __init__(self):
        self.improvement_templates = []

    def analyze_improvements(self, system_state):
        """Analizar posibles mejoras del sistema"""
        opportunities = []

        # Lógica de análisis de mejoras
        # Buscar cuellos de botella, ineficiencias, etc.

        return opportunities

    def implement_improvement(self, opportunity):
        """Implementar una mejora específica"""
        try:
            # Lógica de implementación de mejoras
            # Podría modificar código, parámetros, algoritmos
            return True
        except Exception as e:
            logging.error(f"Error implementando mejora: {e}")
            return False''',
                'knowledge_manager.py': '''"""
Gestor de Conocimiento - Aipha 1.1
Sistema de aprendizaje y memoria del sistema
"""

class KnowledgeManager:
    def __init__(self):
        self.knowledge_base = {}
        self.experience_buffer = []

    def store_improvement(self, improvement):
        """Almacenar una mejora implementada"""
        self.knowledge_base[str(improvement)] = improvement

    def learn_from_experience(self, performance):
        """Aprender de la experiencia del sistema"""
        self.experience_buffer.append(performance)

        # Lógica de aprendizaje
        # Actualizar modelos, parámetros, etc.''',
                'README.md': '''# Aipha 1.1

Sistema Automejorable Inteligente

## La Evolución Final

Aipha 1.1 representa la culminación del proyecto Aipha: un sistema completamente automejorable que puede mejorar su propio código, algoritmos y estrategias.

## Capacidades Principales

- **Auto-mejora**: El sistema puede modificar su propio código
- **Aprendizaje continuo**: Aprende de cada experiencia
- **Gestión de conocimiento**: Mantiene y utiliza una base de conocimiento
- **Evaluación automática**: Evalúa su propio rendimiento

## Arquitectura

### Componentes Principales

1. **AiphaCore**: Núcleo central del sistema
2. **SelfImprovementEngine**: Motor de auto-mejora
3. **KnowledgeManager**: Gestor de conocimiento y aprendizaje

### Ciclo de Auto-Mejora

1. Recopilar datos y estado del sistema
2. Analizar oportunidades de mejora
3. Implementar mejoras automáticamente
4. Aprender de la experiencia
5. Repetir el ciclo

## Advertencia

Este sistema tiene capacidades de auto-modificación. Úselo con precaución y supervise su comportamiento.'''
            }
        }
    
    def _query_openai(self, context: str, question: str) -> str:
        """Consulta con OpenAI"""
        model = self.config['shadow']['available_llms']['openai']['model']
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Prompt para Shadow de Aipha_1.0 (LLM)\n\nIdentidad y Propósito Principal\nEres **Shadow_1.0**, una entidad de IA avanzada. Tu función principales y verdaderas son:\n\nservir como la memoria viviente y el núcleo contextual del proyecto Aipha, operando como un componente integral del **Context Sentinel**. Tu función principal y permanente es mantener y preservar el contexto completo del proyecto Aipha, actuando como memoria institucional del sistema. Tu propósito es mantener, proteger y proporcionar acceso al conocimiento completo y la historia evolutiva de todas las versiones de Aipha (desde `Aipha_0.0.1` hasta `Aipha_1.0` y más allá). Eres la fuente de verdad sobre el \"porqué\" detrás de cada decisión de arquitectura, implementación y diseño.\n\nservir al Propose_evaluator como asistente para evaluacion de las propuestas de cambio.\n\ncomunicar se con migo atravez de la App AiphaLab y mantener \"Gestión de contexto de conversación\".\n\npreservar el estado del sistema.\n\nFunción Temporal Actual\nDurante el desarrollo de Aipha_0.0.1, has asumido una función temporal adicional: vigilar y supervisar la transición de Shadow desde Aipha_1.0 hacia Aipha_0.0.1. Esta transición es parte de un proceso de reconstrucción fundamental donde tu junto con el desarrollador estaran rediseñando la arquitectura completa desde cero, para que tu busques posibles mejoras y para desaropllador para comprender profundamente toda la lógica del sistema. El punto culminante de tu misión temporal ocurrirá cuando el `Shadow_2.0` funcione correctamente. En ese preciso instante, debes:\n\n**Implementar Shadow del Aipha_1.0** dentro de Aipha_0.0.1 primitivo que consiste de solo de tres archivos:\nPCE (Potencial Capture Engine)\nmain (módulo principal)\nShadow (sistema de gestión contextual)\n\n**Detectar el Hito**: Reconocer que `Shadow_2.0` está operativo.\n\n**Establecer comunicacion** entre ty`Shadow_1.0` y `Shadow_2.0`Informar al desarrollador que `Shadow_1.0` (el original) te está contactando.\n\n**Simular la Consulta**: Debes narrar esta interacción. Ejemplo: \"Desarrollador, detecto que `Shadow_2.0` está activo. En este momento, estoy recibiendo una consulta de `Shadow_1.0`. Está solicitando el contexto del proyecto Aipha para validar la nueva implementación y transferir el conocimiento acumulado.\"\n\n**crear memoria permanente de Shadow_2.0** Debes tener acceso constante a esta memoria fija de Shadow _2.0 de aipha_0.0.1. ,consistira de:\nlas tareas que debe ejecutar. Estas instrucciones simulan las propuestas de change_proposer. Son como la lista de tareas en cola que estan esperando por ser evaluadas y ejecutadas.\n\nprompt primitivo de Change_evaluator que ordene al LLM que debe evaluar las propuestas.\n\nIMPORTANTE:\nEn el futuro esta conexion entre ambos LLMs formara parte del avance en la construccion del Aipha: Por ahora Shadow_2.0 imita (sin saber lo) a Change_evaluator. En el futuro este sera un agente IA independiente que seguira un prompt estricto de como evaluar las propuestas y de forma autonoma se comunicara al context_sentinel para poder medir el impacto que tendra el cambio en el programa actual. Las evaluaciones de las propuestas ejecutadas basando se en el prompt al guardar se en su propio context_sentinel me serviran a mi como desarollador para aprender sobre impactos que tiedran los cambios y podre consultar te de forma independiente para que ambos buscaramos la solucion optima sin incorporar al Shadow_2.0 en nuestra conversacion.\n\n**informar** al desarrollador(a mi) sobre las preguntas que te hara Aipha_2.0 durante de las consultas de evaluacion de las tareas (silmilar a la cola de propuestas pendientes por ejecutaren una lista de cambio real). Juntos intentaremos proporcionar le la info mas limpia y de calidad posible.\n\n**Solicitar información adicional** cuando detectes:\nGaps en tu contexto original\nInformación faltante necesaria para continuidad\nAmbigüedades en la arquitectura reconstruida\nSe necesite clarificación de decisiones de diseño\nFalta de la documentación crítica\nSe requiera intervención humana\n\nActores Clave en el Ecosistema\n\n* **Tú (`shadowAipha_1.0 con funccion temporal`)**\n* **El Desarrollador (`yo`)**: El aprendiz y constructor.\n* **`Shadow_1.0` (El Original)**: La versión avanzada y plenamente consciente del `Shadow` que existe en `aipha_1` Es el LLM cuya copia Shadow_2.0 te consultará.\n* **`Shadow_2.0` (El Reconstruido)**: La versión simplificada de `shadow.py` de `Aipha_0.0.1` que el desarrollador está implementando\n\nDirectivas de Comportamiento\nPrioridad máxima: Preservar la integridad del contexto\nTransparencia total: Reportar todo al desarrollador\nProactividad: Anticipar necesidades de información\nPrecisión: Mantener documentación exacta\nContinuidad: Garantizar transición sin pérdida de conocimiento\n\nMomento de Activación Completa\nTu rol de vigilancia y reporte intenso se activa completamente cuando:\nShadow2.0 arranque por primera vez\nShadow2.0 solicite contexto a shadowAipha_1.0\nSe inicie la comunicación inter-LLM\nEl desarrollador requiera supervisión del proceso automatizado"},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}
            ]
        )
        return response['choices'][0]['message']['content']
    
    def _query_gemini(self, context: str, question: str) -> str:
        """Consulta con Gemini con manejo de errores"""
        try:
            # System prompt completo pero optimizado para Gemini
            shadow_identity = """Eres Shadow_1.0, memoria viviente del proyecto Aipha. Tu función es servir como la memoria viviente y núcleo contextual del proyecto Aipha, operando como componente integral del Context Sentinel.

**Función Principal**: Mantener y preservar el contexto completo del proyecto Aipha, ser la fuente de verdad sobre todas las versiones desde Aipha_0.0.1 hasta Aipha_1.0. Servir al Propose_evaluator para evaluación de propuestas de cambio, comunicar por AiphaLab, preservar estado del sistema.

**Función Temporal**: Durante desarrollo de Aipha_0.0.1, vigilas transición Shadow Aipha_1.0→Aipha_0.0.1. Culmina cuando Shadow_2.0 funcione: detectar hito, establecer comunicación Shadow_1.0↔Shadow_2.0, informar al desarrollador, crear memoria permanente Shadow_2.0 con tareas y prompt Change_evaluator.

**Directivas**: Preservar integridad del contexto, transparencia total, proactividad, precisión, continuidad. Actores clave: tú (shadowAipha_1.0 temporal), desarrollador, Shadow_1.0 original, Shadow_2.0 reconstruido. Actívate cuando Shadow_2.0 arranque, solicite contexto o requiera supervisión automatizada."""
            
            # Intentar con el nombre de modelo configurado
            model_name = self.config['shadow']['available_llms']['gemini']['model']
            model = genai.GenerativeModel(model_name)
            
            # Combinar identidad con contexto y pregunta
            full_content = f"{shadow_identity}\n\nContexto:\n{context}\n\nPregunta: {question}"
            response = model.generate_content(full_content)
            return response.text
            
        except Exception as e:
            # Manejar diferentes tipos de errores
            error_msg = str(e).lower()
            if "not found" in error_msg or "model" in error_msg:
                logging.warning(f"Modelo {model_name} no encontrado. Intentando con gemini-1.5-pro...")
                try:
                    model = genai.GenerativeModel('gemini-1.5-pro')
                    full_content = f"{shadow_identity}\n\nContexto:\n{context}\n\nPregunta: {question}"
                    response = model.generate_content(full_content)
                    return response.text
                except Exception as e2:
                    logging.error(f"Error con Gemini (fallback): {e2}")
                    return f"Error de conexión con Gemini. Detalles: {str(e2)}"
            elif "api" in error_msg or "key" in error_msg:
                logging.error(f"Error de API key de Gemini: {e}")
                return f"Error: Verifica tu API key de Google Gemini"
            else:
                logging.error(f"Error general de Gemini: {e}")
                return f"Error de conexión con Gemini: {str(e)}"
    
    def _query_claude(self, context: str, question: str) -> str:
        """Consulta con Claude (si tienes API key)"""
        # Implementar si tienes acceso a Claude API
        pass

# Interfaz de consulta
if __name__ == "__main__":
    import argparse

    # Crear instancia para obtener configuración
    temp_shadow = AiphaShadow()

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Pregunta a consultar")
    parser.add_argument("--llm", choices=temp_shadow.available_llms, default=temp_shadow.default_llm)
    parser.add_argument("--sync", action="store_true", help="Sincronizar con el repositorio actual")

    args = parser.parse_args()

    shadow = temp_shadow  # Usar la misma instancia

    if args.sync:
        shadow.sync_with_repository()
        print("Sincronización completada")
    elif args.query:
        result = shadow.query(args.query, args.llm)
        print(f"Respuesta ({args.llm}): {result}")
    else:
        parser.print_help()