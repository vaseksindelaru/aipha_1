#!/usr/bin/env python3
# gemini_sync.py

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path

class GeminiSync:
    """
    Sincroniza el contexto de Aipha_0.0.1 con shadowAipha_1.0 (Gemini)
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no configurada. Usa variable de entorno o pasa api_key")

        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.repo_path = Path("/home/vaclav/Aipha_0.0.1")
        self.log_file = self.repo_path / "shadow" / "gemini_sync.log"

    def log(self, message):
        """Registra evento en log"""
        timestamp = datetime.utcnow().isoformat() + "Z"
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())

        # Guardar en archivo
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'a') as f:
            f.write(log_entry)

    def generate_context(self):
        """Genera contexto completo del repositorio"""
        self.log("Generando contexto...")

        context_parts = []

        # Header
        context_parts.append("""
# ACTUALIZACIÓN DE CONTEXTO PARA shadowAipha_1.0

**Timestamp:** {timestamp}
**Fuente:** Shadow Monitor - Aipha_0.0.1
**Tipo:** Sincronización automática

## INSTRUCCIÓN CRÍTICA:
Este es el estado ACTUAL del repositorio. REEMPLAZA tu contexto anterior.
""".format(timestamp=datetime.utcnow().isoformat() + "Z"))

        # Archivos principales
        main_files = [
            'README.md',
            'config.json',
            'config_loader.py',
            'main.py',
            'potential_capture_engine.py',
            'shadow.py'
        ]

        context_parts.append("\n## ARCHIVOS DEL SISTEMA:\n")

        for filename in main_files:
            filepath = self.repo_path / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Determinar tipo de código
                if filename.endswith('.py'):
                    lang = 'python'
                elif filename.endswith('.json'):
                    lang = 'json'
                elif filename.endswith('.md'):
                    lang = 'markdown'
                else:
                    lang = 'text'

                context_parts.append(f"\n### {filename}\n```{lang}\n{content}\n```\n")

        # Estado de Git
        try:
            import subprocess
            git_log = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=format:%H|%an|%ad|%s'],
                cwd=self.repo_path
            ).decode('utf-8')

            if git_log:
                hash, author, date, message = git_log.split('|')
                context_parts.append(f"""
## ÚLTIMO COMMIT:
- Hash: {hash[:8]}
- Autor: {author}
- Fecha: {date}
- Mensaje: {message}
""")
        except:
            pass

        # Resumen de arquitectura
        context_parts.append("""
## ARQUITECTURA ACTUAL:
**Sistema:** Aipha_0.0.1 - Potential Capture Engine (PCE)

**Componentes:**
1. `potential_capture_engine.py` - Motor con lógica TP/SL
2. `shadow.py` - Analizador de resultados
3. `config_loader.py` - Gestor de configuración
4. `main.py` - Orquestador principal
5. `config.json` - Parámetros centralizados

**Funcionalidad:** Sistema de análisis de eventos con barreras configurables (NO conecta a exchanges)

**ARCHIVOS QUE NO EXISTEN:** api_connector.py, logger.py
""")

        return ''.join(context_parts)

    def send_to_gemini(self, context):
        """Envía contexto a Gemini API"""
        self.log("Enviando contexto a Gemini...")

        headers = {
            "Content-Type": "application/json"
        }

        # Preparar payload
        payload = {
            "contents": [{
                "parts": [{
                    "text": context + "\n\nConfirma actualización listando archivos principales del sistema."
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048
            }
        }

        # Hacer request
        url_with_key = f"{self.api_url}?key={self.api_key}"

        try:
            response = requests.post(url_with_key, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()

            # Extraer respuesta
            if 'candidates' in result and len(result['candidates']) > 0:
                gemini_response = result['candidates'][0]['content']['parts'][0]['text']
                self.log(f"✓ Gemini respondió: {gemini_response[:200]}...")
                return True, gemini_response
            else:
                self.log("✗ Respuesta de Gemini vacía o inválida")
                return False, "No response"

        except requests.exceptions.RequestException as e:
            self.log(f"✗ Error al comunicar con Gemini: {e}")
            return False, str(e)

    def sync(self):
        """Ejecuta sincronización completa"""
        self.log("=" * 60)
        self.log("INICIANDO SINCRONIZACIÓN CON GEMINI")
        self.log("=" * 60)

        # Generar contexto
        context = self.generate_context()
        self.log(f"Contexto generado: {len(context)} caracteres")

        # Enviar a Gemini
        success, response = self.send_to_gemini(context)

        if success:
            self.log("✓ SINCRONIZACIÓN EXITOSA")

            # Guardar copia del contexto enviado
            backup_file = self.repo_path / "shadow" / f"context_sent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(backup_file, 'w') as f:
                f.write(context)
            self.log(f"Copia guardada en: {backup_file}")

            return True
        else:
            self.log("✗ SINCRONIZACIÓN FALLIDA")
            return False

def main():
    """Punto de entrada principal"""
    if len(sys.argv) > 1 and sys.argv[1] == '--setup':
        print("""
CONFIGURACIÓN DE GEMINI SYNC
============================

1. Obtén tu API key de Gemini:
   https://makersuite.google.com/app/apikey

2. Configura la variable de entorno:
   export GEMINI_API_KEY="tu-api-key-aqui"

3. O agrégala a tu ~/.bashrc:
   echo 'export GEMINI_API_KEY="tu-api-key-aqui"' >> ~/.bashrc
   source ~/.bashrc

4. Prueba la conexión:
   python3 gemini_sync.py --test

5. Ejecuta sincronización:
   python3 gemini_sync.py
""")
        return

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        try:
            sync = GeminiSync()
            print("✓ API key configurada correctamente")
            print(f"✓ Repositorio encontrado: {sync.repo_path}")
            print("✓ Listo para sincronizar")
        except Exception as e:
            print(f"✗ Error: {e}")
        return

    # Sincronización normal
    try:
        sync = GeminiSync()
        success = sync.sync()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR FATAL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()