# aipha_shadow.py
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

class AiphaShadow:
    def __init__(self, config_path: str = 'shadow/config_shadow.yaml'):
        """Inicializa el sistema shadow con múltiples LLMs"""
        # Cargar configuración existente
        self.config = self._load_config(config_path)
        
        # Inicializar ChromaDB (como en tu sistema actual)
        self.client = chromadb.PersistentClient(
            path=str(self.config['knowledge_manager']['chroma_persist_dir'])
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config['knowledge_manager']['collection_name'] + "_shadow"
        )
        
        # Modelo de embeddings (igual al actual)
        self.embedder = SentenceTransformer(
            self.config['knowledge_manager']['embedding_model']
        )
        
        # Configurar múltiples LLMs
        self._setup_llms()
        
        logging.info("AiphaShadow inicializado")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga configuración desde YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_llms(self):
        """Configura múltiples LLMs"""
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Gemini
        if os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Claude (si tienes API key)
        # if os.getenv("ANTHROPIC_API_KEY"):
        #     # Configurar Claude
    
    def sync_with_repository(self, repo_path: str = './'):
        """Sincroniza con el repositorio actual"""
        # Escanear archivos modificados y actualizar embeddings
        # Por ahora, un ejemplo simple
        logging.info("Sincronizando con el repositorio...")
        # Aquí iría la lógica para detectar cambios
        # y actualizar embeddings en ChromaDB
    
    def query(self, question: str, llm: str = "openai") -> str:
        """Consulta usando el LLM especificado"""
        # Recuperar contexto relevante desde ChromaDB
        results = self.collection.query(
            query_texts=[question],
            n_results=5
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
        response = openai.ChatCompletion.create(
            model=self.config['knowledge_manager']['llm_model'],
            messages=[
                {"role": "system", "content": "Eres un experto en Aipha"},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}
            ]
        )
        return response['choices'][0]['message']['content']
    
    def _query_gemini(self, context: str, question: str) -> str:
        """Consulta con Gemini"""
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"Contexto:\n{context}\n\nPregunta: {question}")
        return response.text
    
    def _query_claude(self, context: str, question: str) -> str:
        """Consulta con Claude (si tienes API key)"""
        # Implementar si tienes acceso a Claude API
        pass

# Interfaz de consulta
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Pregunta a consultar")
    parser.add_argument("--llm", choices=["openai", "gemini", "claude"], default="openai")
    parser.add_argument("--sync", help="Ruta del repositorio para sincronizar")

    args = parser.parse_args()

    shadow = AiphaShadow()

    if args.sync:
        shadow.sync_with_repository(args.sync)
        print(f"Sincronización completada para {args.sync}")
    elif args.query:
        result = shadow.query(args.query, args.llm)
        print(f"Respuesta ({args.llm}): {result}")
    else:
        parser.print_help()