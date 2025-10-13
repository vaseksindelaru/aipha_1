# aipha/core/tools/basic_rules_proposer.py

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Asumimos que la librería de OpenAI está instalada: pip install openai
import openai

from ..atomic_update_system import ChangeProposal, CriticalMemoryRules

logger = logging.getLogger(__name__)

# --- Definición del Prompt (como se mostró arriba) ---

SYSTEM_PROMPT = """
Eres Aipha_1.1, una IA de automejora. Tu tarea principal es analizar directivas de alto nivel y generar propuestas de cambio de código atómicas y seguras para mejorar tu propio funcionamiento.

Tu respuesta DEBE ser un único bloque de código JSON válido. No incluyas texto introductorio, explicaciones, ni ```json ... ```. Solo el JSON.

El JSON debe tener la siguiente estructura:
{
  "description": "Una descripción concisa y clara del cambio propuesto.",
  "justification": "Una justificación detallada de por qué este cambio es necesario y beneficioso, alineado con la directiva.",
  "files_affected": ["una/lista/de/archivos/afectados.py"],
  "diff_content": "Un diff en formato unificado que muestra los cambios exactos. Usa '+' para líneas añadidas y '-' para líneas eliminadas."
}

Sigue estrictamente el Protocolo de Actualización Atómica (ATR) y los principios de seguridad. Tus propuestas deben ser pequeñas, verificables y seguras.
"""

def build_user_prompt(directive: str, context_files: dict, system_version: str) -> str:
    """Construye el prompt de usuario con la directiva y el contexto."""
    file_context_str = ""
    for file_path, content in context_files.items():
        file_context_str += f"\n--- Contenido de {file_path} ---\n"
        file_context_str += content
        file_context_str += f"\n--- Fin de {file_path} ---\n"

    return f"""
**Directiva de Alto Nivel:**
"{directive}"

**Contexto del Sistema:**
- Versión Actual: {system_version}
- Foco de la Tarea: Analizar y refinar el conocimiento base del sistema.

**Archivos de Contexto Relevantes:**
{file_context_str}

**Tarea:**
Analiza la directiva y el contenido de los archivos de contexto. Genera una propuesta de cambio en el formato JSON especificado para cumplir con la directiva. La propuesta debe ser aplicable directamente al sistema. Si la directiva es demasiado ambigua o ya está cumplida, puedes proponer una mejora menor o una clarificación en la documentación.
"""


class BasicRulesChangeProposer:
    """
    Un agente que utiliza un LLM para generar propuestas de cambio basadas en reglas
    y conocimiento base del sistema.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada.")
        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info("BasicRulesChangeProposer inicializado con cliente OpenAI.")

    def generate_proposal(self, directive: str, critical_memory: CriticalMemoryRules) -> Optional[ChangeProposal]:
        """
        Genera una propuesta de cambio utilizando un LLM.

        Args:
            directive: La directiva de alto nivel para el cambio.
            critical_memory: La instancia de CriticalMemoryRules para crear la propuesta.

        Returns:
            Un objeto ChangeProposal o None si falla la generación.
        """
        try:
            # 1. Recopilar contexto
            system_version = critical_memory.get_current_version()
            # Para esta tarea, el archivo más relevante es CRITICAL_CONTRACTS.md
            contracts_path = Path(__file__).parent.parent.parent / "docs" / "CRITICAL_CONTRACTS.md"
            if contracts_path.exists():
                with open(contracts_path, 'r', encoding='utf-8') as f:
                    contracts_content = f.read()
            else:
                logger.warning(f"Archivo CRITICAL_CONTRACTS.md no encontrado en {contracts_path}")
                contracts_content = "Archivo no encontrado - usando contexto mínimo"

            context_files = {
                str(contracts_path.relative_to(Path.cwd())): contracts_content
            }

            # 2. Construir el prompt
            user_prompt = build_user_prompt(directive, context_files, system_version)

            # 3. Llamar al LLM
            logger.info("Enviando directiva al LLM para generar propuesta...")
            response = self.client.chat.completions.create(
                model=self.config.get('llm', {}).get('model', 'gpt-4-turbo-preview'),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            response_content = response.choices[0].message.content
            logger.debug(f"Respuesta JSON del LLM: {response_content}")

            # 4. Parsear la respuesta y crear la propuesta
            proposal_data = json.loads(response_content)

            # Validar que los campos necesarios están presentes
            required_keys = ["description", "justification", "files_affected", "diff_content"]
            if not all(key in proposal_data for key in required_keys):
                logger.error(f"La respuesta del LLM no contiene todos los campos requeridos. Respuesta: {proposal_data}")
                return None

            proposal = critical_memory.create_change_proposal(
                description=proposal_data['description'],
                justification=proposal_data['justification'],
                files_affected=proposal_data['files_affected'],
                diff_content=proposal_data['diff_content'],
                author="Aipha_1.1::BasicRulesChangeProposer"
            )
            return proposal

        except openai.APIError as e:
            logger.error(f"Error en la API de OpenAI: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error al decodificar la respuesta JSON del LLM: {e}")
            logger.error(f"Contenido recibido: {response_content}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado al generar la propuesta: {e}", exc_info=True)
            return None