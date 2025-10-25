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

    def _load_aipha_context(self) -> str:
        """Carga el contexto de todos los proyectos Aipha desde GitHub para incluir en consultas"""
        context_parts = []
        context_parts.append("=== CONTEXTO COMPLETO DE TODOS LOS PROYECTOS AIPHA ===")

        # Repositorios de Aipha en GitHub - URLs corregidas para acceso real
        # NOTA: Los repositorios deben existir en GitHub para que Gemini tenga acceso real
        aipha_repos = [
            ('Aipha_0.0.1', 'https://api.github.com/repos/vaclav/Aipha_0.0.1/contents'),
            ('Aipha_0.2.1', 'https://api.github.com/repos/vaclav/Aipha_0.2.1/contents'),
            ('Aipha_0.3.1', 'https://api.github.com/repos/vaclav/Aipha_0.3.1/contents'),
            ('Aipha_1.1', 'https://api.github.com/repos/vaclav/Aipha_1.1/contents')
        ]

        # Intentar también con diferentes nombres de usuario si los repositorios existen
        alternative_repos = [
            ('Aipha_0.0.1', 'https://api.github.com/repos/Aipha/Aipha_0.0.1/contents'),
            ('Aipha_0.2.1', 'https://api.github.com/repos/Aipha/Aipha_0.2.1/contents'),
            ('Aipha_0.3.1', 'https://api.github.com/repos/Aipha/Aipha_0.3.1/contents'),
            ('Aipha_1.1', 'https://api.github.com/repos/Aipha/Aipha_1.1/contents')
        ]

        # Para desarrollo/testing: simular contenido de repositorios inexistentes
        # Esto permite probar la funcionalidad mientras los repositorios no existen
        simulated_content = self._get_simulated_repo_content()

        # También incluir el proyecto local actual
        current_project_path = Path(__file__).parent.parent
        if current_project_path.exists():
            context_parts.append(f"\n=== PROYECTO LOCAL ACTUAL ({current_project_path.name}) ===")
            self._add_local_project_context(context_parts, current_project_path)

        # Cargar contexto de cada repositorio de GitHub - intentar múltiples URLs
        for repo_name, api_url in aipha_repos:
            repo_loaded = False
            try:
                context_parts.append(f"\n=== REPOSITORIO GITHUB: {repo_name} ===")
                self._add_github_repo_context(context_parts, repo_name, api_url)
                repo_loaded = True
            except Exception as e:
                # Intentar con URL alternativa
                alt_repo = next((alt for alt in alternative_repos if alt[0] == repo_name), None)
                if alt_repo:
                    try:
                        context_parts.append(f"\n--- Intentando URL alternativa para {repo_name} ---")
                        self._add_github_repo_context(context_parts, repo_name, alt_repo[1])
                        repo_loaded = True
                    except Exception as e2:
                        # Usar contenido simulado para desarrollo/testing
                        context_parts.append(f"\n--- CONTENIDO SIMULADO PARA {repo_name} (repositorio no encontrado en GitHub) ---")
                        simulated_data = simulated_content.get(repo_name, {})
                        if simulated_data:
                            context_parts.append(f"--- ARCHIVOS SIMULADOS EN {repo_name}: {len(simulated_data)} ---")
                            for file_path, content in simulated_data.items():
                                context_parts.append(f"\n--- {file_path} ---\n{content}")
                            context_parts.append(f"\n--- NOTA: Este es contenido SIMULADO para testing. Para contenido real, crea el repositorio {repo_name} en GitHub ---")
                            repo_loaded = True
                        else:
                            context_parts.append(f"\n--- Error cargando {repo_name} (ambas URLs fallaron): {e} / {e2} ---")
                else:
                    context_parts.append(f"\n--- Error cargando {repo_name}: {e} ---")

            if not repo_loaded:
                # Si no se pudo cargar ni simular, agregar información de que el repo no existe
                context_parts.append(f"\n--- REPOSITORIO {repo_name} NO ENCONTRADO ---")
                context_parts.append("Este repositorio no existe en GitHub o no es accesible.")
                context_parts.append("Para que Gemini tenga acceso real al código, el repositorio debe existir en GitHub.")

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
                {"role": "system", "content": "Eres un experto en Aipha"},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}
            ]
        )
        return response['choices'][0]['message']['content']
    
    def _query_gemini(self, context: str, question: str) -> str:
        """Consulta con Gemini con manejo de errores"""
        try:
            # Intentar con el nombre de modelo antiguo (por si acerta)
            model_name = self.config['shadow']['available_llms']['gemini']['model']
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(f"Contexto:\n{context}\n\nPregunta: {question}")
            return response.text
        except Exception as e:
            # Si falla por el modelo no encontrado, intentar con un nombre actualizado
            if "not found" in str(e) or "gemini-pro" in str(e):
                logging.warning(f"Modelo {model_name} no encontrado. Intentando con 'gemini-2.5-pro'...")
                try:
                    # Usar un nombre de modelo que sí exista en la API actual
                    model = genai.GenerativeModel('gemini-1.5-pro')
                    response = model.generate_content(f"Contexto:\n{context}\n\nPregunta: {question}")
                    return response.text
                except Exception as e2:
                    logging.error(f"Error con Gemini (modelo actualizado): {e2}")
                    return f"Error: No se pudo contactar con Gemini. Verifica tu API key y el modelo."
            else:
                logging.error(f"Error de Gemini: {e}")
                return f"Error al contactar con Gemini: {e}"
    
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