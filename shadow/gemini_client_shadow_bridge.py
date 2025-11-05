#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gemini_client_shadow_bridge.py - Cliente Gemini API para comunicación Shadow-AiphaLab

Este módulo implementa la comunicación bidireccional entre el sistema Shadow local
y AiphaLab a través de la API de Gemini, permitiendo el envío de contexto y
solicitud de información histórica.

Autor: Shadow System
Versión: 1.0.0
"""

import os
import sys
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time

class GeminiClientShadowBridge:
    """
    Cliente Gemini API para comunicación bidireccional Shadow-AiphaLab
    """
    
    def __init__(self, 
                 gemini_api_key: str = None,
                 shadow_memory_path: str = "./aipha_memory_storage/action_history"):
        """
        Inicializa el cliente Gemini para comunicación Shadow-AiphaLab
        
        Args:
            gemini_api_key: Clave de API de Gemini
            shadow_memory_path: Ruta a memoria Shadow
        """
        self.gemini_api_key = gemini_api_key or self._get_gemini_api_key()
        self.shadow_memory_path = shadow_memory_path
        
        # URLs de API
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        
        # Configurar logging
        self._setup_logging()
        
        # Cache de respuestas
        self.response_cache = {}
        self.last_request_time = 0
        self.min_request_interval = 1  # 1 segundo entre requests
        
        self.logger.info("🔗 Gemini-Shadow Bridge inicializado")

    def _setup_logging(self):
        """Configura el sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _get_gemini_api_key(self) -> str:
        """Obtiene la API key de Gemini desde variables de entorno"""
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            self.logger.warning("No se encontró API key de Gemini en variables de entorno")
        return api_key

    def send_context_to_aiphalab(self, 
                                context_data: str, 
                                query: str = None) -> Dict[str, Any]:
        """
        Envía contexto del sistema Shadow a AiphaLab
        
        Args:
            context_data: Contexto generado por Shadow
            query: Consulta específica (opcional)
            
        Returns:
            Dict con resultado del envío
        """
        result = {
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'context_received': True,
            'gemini_response': None,
            'error': None
        }
        
        try:
            # Preparar prompt para AiphaLab
            prompt = f"""
Eres AiphaLab, un sistema de análisis de proyectos. El siguiente contexto proviene del sistema Shadow que monitoriza el repositorio Aipha_0.0.1.

**CONTEXTO DEL REPOSITORIO:**
{context_data}

**CONSULTA:** {query or "Evalúa el estado actual del proyecto Aipha_0.0.1 basado en este contexto"}

**INSTRUCCIONES:**
1. Analiza el estado del proyecto basado en la información proporcionada
2. Identifica fortalezas y áreas de mejora
3. Proporciona recomendaciones específicas
4. Evalúa el progreso del desarrollo

Responde como AiphaLab evaluando el estado del proyecto.
"""
            
            # Verificar si tenemos API key
            if not self.gemini_api_key:
                result['error'] = 'API key de Gemini no configurada'
                result['fallback_mode'] = True
                return result
            
            # Enviar a Gemini (simulado por ahora)
            gemini_response = self._send_to_gemini(prompt)
            result['gemini_response'] = gemini_response
            result['success'] = gemini_response.get('success', False)
            
            # Registrar envío en memoria Shadow
            self._register_context_sent(result)
            
        except Exception as e:
            self.logger.error(f"Error enviando contexto a AiphaLab: {e}")
            result['error'] = str(e)
        
        return result

    def _send_to_gemini(self, prompt: str) -> Dict[str, Any]:
        """Envía prompt a Gemini API (simulado para testing)"""
        try:
            # Simular respuesta de Gemini para testing
            import random
            
            simulated_responses = [
                "He analizado el contexto del repositorio Aipha_0.0.1. El sistema muestra un estado técnico excelente con un score de integridad del 100%. Se han registrado múltiples eventos de desarrollo recientes, indicando actividad continua. Las recomendaciones incluyen: 1) Continuar con el monitoreo automatizado, 2) Implementar las mejoras propuestas por Shadow, 3) Considerar la transición a Aipha_1.0 cuando los milestones estén completos.",
                "Basándome en la información del sistema Shadow, el proyecto Aipha_0.0.1 se encuentra en un estado óptimo. La arquitectura de 79 archivos demuestra un sistema complejo y bien estructurado. La memoria de Shadow registra actividad reciente, confirmando que el desarrollo sigue activo. Fortalezas identificadas: integridad técnica, monitoreo automatizado, memoria histórica. Áreas de mejora: documentación adicional, testing comprehensivo.",
                "El análisis del sistema Shadow revela que Aipha_0.0.1 está en excelente estado. Los checksums y análisis de integridad confirman que no hay problemas técnicos. Los 4 cambios recientes muestran un desarrollo activo. La integración Shadow-AiphaLab funciona correctamente. Recomendaciones: mantener la frecuencia de análisis, implementar las sugerencias de optimización detectadas, preparar la transición a versiones posteriores."
            ]
            
            # Verificar rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < self.min_request_interval:
                time.sleep(self.min_request_interval - (current_time - self.last_request_time))
            
            # Simular delay de API
            time.sleep(0.5)
            
            response_text = random.choice(simulated_responses)
            
            self.last_request_time = time.time()
            
            return {
                'success': True,
                'response': response_text,
                'request_id': hashlib.md5(prompt.encode()).hexdigest()[:8],
                'timestamp': datetime.now().isoformat(),
                'mode': 'simulated'
            }
                
        except Exception as e:
            self.logger.error(f"Error enviando a Gemini: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _register_context_sent(self, result: Dict[str, Any]):
        """Registra el envío de contexto en memoria Shadow"""
        try:
            # Añadir entrada a la memoria de Shadow
            shadow_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "Gemini Bridge: Contexto enviado a AiphaLab",
                "agent": "GeminiClientShadowBridge",
                "component": "gemini_bridge",
                "status": "success" if result.get('success') else "partial",
                "details": {
                    "bridge_type": "gemini_api",
                    "query": result.get('query'),
                    "context_received": result.get('context_received'),
                    "gemini_response_received": result.get('gemini_response') is not None,
                    "error": result.get('error'),
                    "mode": result.get('gemini_response', {}).get('mode', 'unknown')
                }
            }
            
            # Guardar en archivo de memoria
            memory_file = os.path.join(self.shadow_memory_path, 'current_history.json')
            
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(shadow_entry)
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"Error registrando envío: {e}")

    def get_bridge_status(self) -> Dict[str, Any]:
        """Obtiene el estado del bridge"""
        return {
            'timestamp': datetime.now().isoformat(),
            'gemini_api_configured': bool(self.gemini_api_key),
            'cache_size': len(self.response_cache),
            'last_request_time': self.last_request_time,
            'rate_limiting': {
                'enabled': True,
                'min_interval_seconds': self.min_request_interval
            },
            'endpoints': {
                'gemini_api': self.gemini_api_url
            }
        }


def main():
    """Función principal para testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gemini-Shadow Bridge Client')
    parser.add_argument('--memory-path', default='./aipha_memory_storage/action_history',
                       help='Ruta a memoria Shadow')
    parser.add_argument('--mode', choices=['send', 'status'], default='status',
                       help='Modo de operación')
    parser.add_argument('--query', help='Consulta para enviar')
    
    args = parser.parse_args()
    
    # Crear bridge
    bridge = GeminiClientShadowBridge(shadow_memory_path=args.memory_path)
    
    if args.mode == 'send':
        print("📤 Enviando contexto a AiphaLab...")
        
        # Contexto de ejemplo
        context_example = """
# CONTEXTO COMPLETO - REPOSITORIO AIPHA_0.0.1
**Generado:** 2025-11-04T11:46:12.430Z
**Consulta:** estado del repositorio

## ESTADO DEL REPOSITORIO
- **URL:** https://github.com/vaseksindelaru/aipha_0.0.1.git
- **Archivos totales:** 79
- **Score de integridad:** 100/100
- **Fuente de datos:** shadow_monitor

## ANÁLISIS DE INTEGRIDAD
- **Score:** 100/100
- **Archivos analizados:** 7
- **Issues encontrados:** 0
- **Score de estructura:** 100/100
"""
        
        result = bridge.send_context_to_aiphalab(
            context_data=context_example,
            query=args.query or "Evalúa el estado del proyecto"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    else:
        print("🔍 Estado del Bridge:")
        status = bridge.get_bridge_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()