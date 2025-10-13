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

        # Preparar contexto
        context = "\n---\n".join([doc for doc in results['documents'][0]])

        # Consultar con el LLM elegido
        if llm == "openai":
            return self._query_openai(context, question)
        elif llm == "gemini":
            return self._query_gemini(context, question)
        elif llm == "claude":
            return self._query_claude(context, question)
        else:
            raise ValueError(f"LLM {llm} no soportado")
    
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