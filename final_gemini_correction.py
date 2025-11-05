#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_gemini_correction.py - Solución definitiva para corregir respuestas de Gemini
Integra todos los componentes: Shadow monitoring + Corrección de respuestas
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any

# Añadir path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini_response_corrector import GeminiResponseCorrector

class FinalGeminiCorrection:
    """
    Solución definitiva para corregir respuestas incorrectas de Gemini
    sobre el repositorio Aipha_0.0.1
    """
    
    def __init__(self):
        """Inicializar sistema de corrección completo"""
        self.corrector = GeminiResponseCorrector()
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🎯 Final Gemini Correction System iniciado")
    
    def _is_repository_query(self, response: str) -> bool:
        """Detecta si la respuesta es sobre el repositorio"""
        return self.corrector._is_repository_query(response)
    
    def process_gemini_response(self, gemini_response: str) -> Dict[str, str]:
        """
        Procesa una respuesta de Gemini y la corrige si es necesario
        
        Args:
            gemini_response: Respuesta original de Gemini
            
        Returns:
            Dict con respuesta procesada y metadatos
        """
        timestamp = datetime.now().isoformat()
        
        # Verificar si necesita corrección
        needs_correction = self.corrector._is_repository_query(gemini_response)
        
        if needs_correction:
            corrected_response = self.corrector.correct_repository_response(gemini_response)
            explanation = self.corrector.explain_correction(gemini_response, corrected_response)
            
            self.logger.info(f"✅ Respuesta corregida en {timestamp}")
            
            return {
                'original_response': gemini_response,
                'corrected_response': corrected_response,
                'explanation': explanation,
                'needs_correction': True,
                'timestamp': timestamp,
                'status': 'corrected'
            }
        else:
            self.logger.info(f"📋 Respuesta no requiere corrección en {timestamp}")
            
            return {
                'original_response': gemini_response,
                'corrected_response': gemini_response,
                'explanation': 'No se requiere corrección (no relacionado con repositorio)',
                'needs_correction': False,
                'timestamp': timestamp,
                'status': 'no_action'
            }
    
    def get_repository_verification(self) -> Dict[str, Any]:
        """Obtiene verificación completa del repositorio"""
        verification = self.corrector.shadow_integration.verify_repository_files()
        context = self.corrector.shadow_integration.get_repository_context("verificación")
        
        return {
            'verification': verification,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Función principal - Demo de la solución"""
    print("🎯 SOLUCIÓN DEFINITIVA - CORRECCIÓN GEMINI")
    print("=" * 55)
    
    system = FinalGeminiCorrection()
    
    # Demostración con la respuesta incorrecta original
    incorrect_gemini_response = """Comunicando desde AiphaLab. Soy Shadow_1.0, núcleo contextual del proyecto Aipha.

Procesando solicitud de estado de archivos para la versión `Aipha_0.0.1`.

Conforme a la memoria del sistema, la estructura de archivos actual para **Aipha_0.0.1** es la siguiente:

```plaintext
/Aipha_0.0.1/
|
├── main.py             # Script principal de ejecución del bot.
|
├── config.ini          # Archivo de configuración (API keys, pares de trading, parámetros).
|
├── trading_bot.py      # Contiene la lógica central del bot y el ciclo de operaciones.
|
├── api_connector.py    # Módulo para la comunicación con la API del exchange.
|
├── strategy.py         # Implementación de la estrategia de trading básica (ej. cruce de medias móviles).
|
├── logger.py           # Módulo para el registro de eventos, operaciones y errores.
|
└── requirements.txt    # Dependencias del proyecto para esta versión.
```"""
    
    print("\n🔍 PROCESANDO RESPUESTA DE GEMINI...")
    result = system.process_gemini_response(incorrect_gemini_response)
    
    print(f"📊 ESTADO: {result['status'].upper()}")
    print(f"⏰ TIMESTAMP: {result['timestamp']}")
    print(f"🔧 CORRECCIÓN REQUERIDA: {'SÍ' if result['needs_correction'] else 'NO'}")
    
    print("\n" + "="*55)
    print("📋 RESPUESTA FINAL CORREGIDA:")
    print("="*55)
    print(result['corrected_response'])
    
    print("\n" + "="*55)
    print("📝 EXPLICACIÓN DEL PROCESO:")
    print("="*55)
    print(result['explanation'])
    
    print("\n" + "="*55)
    print("✅ VERIFICACIÓN DEL REPOSITORIO:")
    print("="*55)
    verification = system.get_repository_verification()
    print(f"Archivos verificados: {len(verification['verification']['found'])}")
    print("Archivos reales encontrados:")
    for file in verification['verification']['found']:
        print(f"  ✅ {file}")


if __name__ == "__main__":
    main()