# aipha_shadow_simple.py - Versión simplificada sin ChromaDB
from dotenv import load_dotenv
load_dotenv()

import os
import openai
import google.generativeai as genai
import yaml
import logging
from typing import Dict, Any, List
from datetime import datetime

class AiphaShadowSimple:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config_shadow.yaml')
        
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Configurar múltiples LLMs
        self._setup_llms()
        
        # Configurar LLMs disponibles y por defecto
        self.available_llms = list(self.config['shadow']['available_llms'].keys())
        self.default_llm = self.config['shadow']['default_llm']

        logging.info("AiphaShadowSimple inicializado")
    
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
    
    def query(self, question: str, llm: str = None) -> str:
        """Consulta usando el LLM especificado"""
        if llm is None:
            llm = self.config['shadow']['default_llm']
        available_llms = self.config['shadow']['available_llms']
        if llm not in available_llms:
            raise ValueError(f"LLM {llm} no soportado")

        # Sin contexto de ChromaDB, solo contexto básico
        context = self._load_basic_context()

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

        # Contexto básico
        context = self._load_basic_context()

        # AÑADIR MEMORIA CONVERSACIONAL SI EXISTE
        if conversation_history and len(conversation_history) > 1:
            conversation_summary = self._format_conversation_history(conversation_history)
            context = f"{conversation_summary}\n\n--- CONTEXTO BÁSICO AIPHA ---\n{context}"

        # Consultar con el LLM elegido
        if llm == "openai":
            return self._query_openai(context, question)
        elif llm == "gemini":
            return self._query_gemini(context, question)
        elif llm == "claude":
            return self._query_claude(context, question)
        else:
            raise ValueError(f"LLM {llm} no soportado")

    def _load_basic_context(self) -> str:
        """Carga contexto básico sin ChromaDB"""
        return """=== CONTEXTO BÁSICO DE AIPHA ===

Aipha es un proyecto de sistema de trading inteligente que evoluciona desde versiones básicas hasta sistemas automejorables.

Versiones principales:
- Aipha_0.0.1: Sistema básico de trading
- Aipha_0.2: Sistema avanzado con oráculos
- Aipha_0.3.1: Sistema con deep learning
- Aipha_1.0: Sistema con AiphaLab
- Aipha_1.1: Sistema automejorable

Shadow_1.0 es la memoria viviente del proyecto Aipha, actuando como Context Sentinel."""

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
    
    def _query_openai(self, context: str, question: str) -> str:
        """Consulta con OpenAI"""
        model = self.config['shadow']['available_llms']['openai']['model']
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres Shadow_1.0, la memoria viviente del proyecto Aipha"},
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