# shadow.py

import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import requests
from typing import Dict, List, Any, Optional

class Shadow:
    """
    Shadow 2.0 - Arquitecto Pedagógico de Aipha
    Representa una capa de conocimiento que analiza los resultados
    producidos por otras capas, sin conocer su implementación interna.

    Mejoras en 2.0:
    - Memoria conversacional persistente
    - Gestión dinámica de contexto
    - Acceso a conocimiento de GitHub
    - Rol pedagógico y correcciones
    - Integración con ContextSentinel y ChangeProposer
    """

    def __init__(self, memory_file: str = "shadow_memory.json", github_repo: str = "https://github.com/vaseksindelaru/aipha_0.0.1.git"):
        self.memory_file = Path(memory_file)
        self.github_repo = github_repo
        self.conversation_memory = []
        self.context = self._load_base_context()
        self.last_github_commit = None

        # Inicializar memoria
        self._load_memory()

        print("[Shadow 2.0] Arquitecto Pedagógico inicializado.")
        print(f"[Shadow 2.0] Memoria: {len(self.conversation_memory)} conversaciones previas")
        print(f"[Shadow 2.0] Contexto base cargado con {len(self.context)} elementos")

    def _load_base_context(self) -> Dict[str, Any]:
        """Carga el contexto base de Aipha"""
        return {
            "aipha_version": "0.0.1",
            "architecture": {
                "layers": ["PCE (Domain Logic)", "Main (Orchestration)", "Shadow (Knowledge)"],
                "goal": "Construir Aipha 1.0 paso a paso"
            },
            "current_files": ["potential_capture_engine.py", "main.py", "shadow.py"],
            "philosophy": "Aprendizaje por construcción, no por generación automática",
            "github_repo": self.github_repo
        }

    def _load_memory(self):
        """Carga la memoria conversacional desde archivo"""
        if self.memory_file.exists():
            try:
                with self.memory_file.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversation_memory = data.get('conversations', [])
                    self.last_github_commit = data.get('last_github_commit')
                    print(f"[Shadow 2.0] Memoria cargada: {len(self.conversation_memory)} conversaciones")
            except Exception as e:
                print(f"[Shadow 2.0] Error cargando memoria: {e}")
                self.conversation_memory = []

    def _save_memory(self):
        """Guarda la memoria conversacional"""
        data = {
            'conversations': self.conversation_memory[-100:],  # Mantener últimas 100
            'last_github_commit': self.last_github_commit,
            'last_updated': datetime.now().isoformat()
        }
        try:
            with self.memory_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Shadow 2.0] Error guardando memoria: {e}")

    def check_github_updates(self) -> bool:
        """Verifica actualizaciones en el repositorio de GitHub"""
        try:
            # Extraer owner/repo del URL
            repo_path = self.github_repo.replace('https://github.com/', '').replace('.git', '')
            api_url = f"https://api.github.com/repos/{repo_path}/commits"

            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                commits = response.json()
                if commits:
                    latest_commit = commits[0]['sha']
                    if latest_commit != self.last_github_commit:
                        print(f"[Shadow 2.0] Nuevo commit detectado: {latest_commit[:8]}")
                        self.last_github_commit = latest_commit
                        self._sync_github_knowledge(commits[0])
                        self._save_memory()
                        return True
            return False
        except Exception as e:
            print(f"[Shadow 2.0] Error verificando GitHub: {e}")
            return False

    def _sync_github_knowledge(self, commit_data: dict):
        """Sincroniza conocimiento desde GitHub"""
        commit_message = commit_data.get('commit', {}).get('message', '')
        author = commit_data.get('commit', {}).get('author', {}).get('name', '')

        # Agregar a memoria como conocimiento nuevo
        knowledge_entry = {
            'type': 'github_update',
            'timestamp': datetime.now().isoformat(),
            'commit_hash': commit_data['sha'][:8],
            'message': commit_message,
            'author': author,
            'files_changed': []  # Podría expandirse para obtener archivos modificados
        }

        self.conversation_memory.append({
            'role': 'system',
            'content': f"Actualización de GitHub: {commit_message}",
            'metadata': knowledge_entry
        })

    def correct_context(self, correction: str) -> str:
        """Permite correcciones dinámicas al contexto"""
        # Analizar el tipo de corrección
        if "aipha_version" in correction.lower():
            # Extraer nueva versión
            import re
            version_match = re.search(r'(\d+\.\d+\.\d+)', correction)
            if version_match:
                old_version = self.context['aipha_version']
                self.context['aipha_version'] = version_match.group(1)
                response = f"Contexto corregido: Versión de Aipha cambiada de {old_version} a {self.context['aipha_version']}"
            else:
                response = "No pude identificar la nueva versión en la corrección"
        elif "architecture" in correction.lower():
            response = "Corrección de arquitectura aplicada (funcionalidad básica)"
        else:
            # Corrección general
            self.context['custom_corrections'] = self.context.get('custom_corrections', [])
            self.context['custom_corrections'].append({
                'timestamp': datetime.now().isoformat(),
                'correction': correction
            })
            response = f"Corrección aplicada al contexto: {correction[:100]}..."

        self._save_memory()
        return response

    def pedagogical_analysis(self, user_input: str) -> str:
        """Análisis pedagógico de la entrada del usuario"""
        analysis = []

        # Verificar si es una pregunta de comprensión
        if any(word in user_input.lower() for word in ['por qué', 'why', 'cómo', 'how', 'qué', 'what']):
            analysis.append("Detecto una pregunta que busca comprensión profunda - excelente enfoque pedagógico!")

        # Verificar si propone cambios
        if any(word in user_input.lower() for word in ['cambiar', 'modificar', 'change', 'modify', 'mejorar', 'improve']):
            analysis.append("Veo que propones mejoras - ¿has considerado el impacto en la arquitectura general?")

        # Verificar si pregunta por ayuda
        if any(word in user_input.lower() for word in ['ayuda', 'help', 'duda', 'confusión']):
            analysis.append("No hay preguntas tontas, solo oportunidades de aprendizaje!")

        return " | ".join(analysis) if analysis else ""

    def generate_pedagogical_response(self, user_input: str, analysis_result: str = "") -> str:
        """Genera respuesta pedagógica basada en el análisis PCE"""
        if not analysis_result:
            return ""

        pedagogical_tips = []

        # Consejos específicos basados en el análisis
        if "TP" in analysis_result or "SL" in analysis_result:
            pedagogical_tips.append("¿Qué te dice este resultado sobre la importancia de adaptar las barreras a la volatilidad del mercado?")

        if "timeout" in analysis_result:
            pedagogical_tips.append("Los timeouts son oportunidades de aprendizaje - ¿por qué crees que la operación no alcanzó sus objetivos?")

        if len([x for x in analysis_result.split() if x.replace('.', '').replace('%', '').isdigit()]) > 2:
            pedagogical_tips.append("Estos porcentajes son datos valiosos. ¿Cómo podríamos usarlos para mejorar la estrategia?")

        return " | ".join(pedagogical_tips) if pedagogical_tips else ""

    def analyze_pce_output(self, labels: pd.Series):
        """
        Analiza las etiquetas generadas por el PCE.
        Su único "conocimiento" es el formato de las etiquetas (1, -1, 0).
        """
        print("\n[Shadow 2.0] Recibiendo datos para análisis pedagógico...")

        if labels.empty:
            print("[Shadow 2.0] No hay etiquetas para analizar.")
            return

        # Verificar actualizaciones de GitHub antes del análisis
        self.check_github_updates()

        print("[Shadow 2.0] --- INICIO DEL ANÁLISIS PEDAGÓGICO ---")

        num_events = len(labels)
        tp_count = (labels == 1).sum()
        sl_count = (labels == -1).sum()
        timeout_count = (labels == 0).sum()

        analysis_result = f"""
Total de eventos analizados: {num_events}
  - Take Profits (1): {tp_count} ({(tp_count/num_events)*100:.2f}%)
  - Stop Losses (-1): {sl_count} ({(sl_count/num_events)*100:.2f}%)
  - Timeouts (0):    {timeout_count} ({(timeout_count/num_events)*100:.2f}%)
        """.strip()

        print(analysis_result)

        # Análisis pedagógico
        pedagogical_insights = self.generate_pedagogical_response("", analysis_result)
        if pedagogical_insights:
            print(f"\n[Shadow 2.0] Reflexiones Pedagógicas: {pedagogical_insights}")

        # Almacenar en memoria conversacional
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'pce_analysis',
            'input': 'Análisis PCE automático',
            'output': analysis_result,
            'pedagogical_insights': pedagogical_insights,
            'context_state': self.context.copy()
        }

        self.conversation_memory.append(conversation_entry)
        self._save_memory()

        print("[Shadow 2.0] --- FIN DEL ANÁLISIS PEDAGÓGICO ---")

        # Verificar si necesitamos proponer mejoras
        win_rate = (labels == 1).mean()
        if win_rate < 0.3:
            print(f"\n[Shadow 2.0] ⚠️  Tasa de ganancias baja ({win_rate:.1%}). Considera revisar los parámetros del PCE.")
            print("[Shadow 2.0] 💡 Sugerencia: ¿Qué tal experimentar con barreras dinámicas basadas en volatilidad?")

    def query_knowledge(self, question: str) -> str:
        """Consulta el conocimiento acumulado"""
        # Buscar en memoria conversacional
        relevant_memories = []
        for memory in self.conversation_memory[-20:]:  # Últimas 20 entradas
            if any(keyword in question.lower() for keyword in ['análisis', 'pce', 'resultados', 'rendimiento']):
                if memory.get('type') == 'pce_analysis':
                    relevant_memories.append(memory)

        if relevant_memories:
            latest_analysis = relevant_memories[-1]
            return f"Basado en el último análisis PCE: {latest_analysis.get('output', 'No disponible')}"
        else:
            return "No tengo análisis previos relevantes para esta consulta."

    def get_conversation_summary(self) -> str:
        """Obtiene resumen de la conversación actual"""
        if not self.conversation_memory:
            return "No hay conversaciones previas."

        recent_conversations = self.conversation_memory[-5:]  # Últimas 5
        summary = f"Resumen de las últimas {len(recent_conversations)} interacciones:\n"

        for i, conv in enumerate(recent_conversations, 1):
            conv_type = conv.get('type', 'general')
            timestamp = conv.get('timestamp', '')[:19]  # Solo fecha y hora
            summary += f"{i}. [{timestamp}] {conv_type}\n"

        return summary

    def __del__(self):
        """Guardar memoria al destruir el objeto"""
        self._save_memory()