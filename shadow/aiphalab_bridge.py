#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aiphalab_bridge.py - Puente de comunicación entre AiphaLab y Shadow Memory

Este módulo proporciona una interfaz para que AiphaLab pueda consultar
y obtener información contextual de la memoria de Shadow.

Permite consultas estructuradas sobre:
- Eventos de desarrollo y evolución del código
- Estado actual del proyecto Aipha
- Contexto histórico de decisiones tomadas
- Información de commits y cambios realizados

Autor: Shadow System
Versión: 1.0.0
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import argparse

class AiphaLabBridge:
    """
    Puente de comunicación entre AiphaLab y Shadow Memory System
    """

    def __init__(self, shadow_memory_path: str):
        self.shadow_memory_path = shadow_memory_path
        self.memory_file = os.path.join(shadow_memory_path, 'current_history.json')
        self.logger = logging.getLogger(__name__)

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def query_shadow_memory(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consulta la memoria de Shadow con parámetros específicos

        Args:
            query_params: Parámetros de consulta

        Returns:
            Dict con resultados de la consulta
        """
        try:
            # Cargar memoria de Shadow
            memory_data = self._load_shadow_memory()

            if not memory_data:
                return {
                    "status": "error",
                    "message": "No se pudo cargar la memoria de Shadow",
                    "data": None
                }

            # Aplicar filtros de consulta
            filtered_data = self._apply_filters(memory_data, query_params)

            # Formatear respuesta
            response = {
                "status": "success",
                "query_timestamp": datetime.now().isoformat(),
                "total_entries": len(memory_data),
                "filtered_entries": len(filtered_data),
                "data": filtered_data
            }

            # Agregar metadatos si se solicita
            if query_params.get('include_metadata', False):
                response['metadata'] = self._generate_metadata(memory_data, filtered_data)

            return response

        except Exception as e:
            self.logger.error(f"Error en consulta de Shadow Memory: {e}")
            return {
                "status": "error",
                "message": f"Error interno: {str(e)}",
                "data": None
            }

    def _load_shadow_memory(self) -> List[Dict[str, Any]]:
        """Carga la memoria de Shadow desde el archivo"""
        try:
            if not os.path.exists(self.memory_file):
                self.logger.warning(f"Archivo de memoria no encontrado: {self.memory_file}")
                return []

            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)

            # Validar que es una lista
            if not isinstance(memory_data, list):
                self.logger.error("Formato de memoria inválido: no es una lista")
                return []

            return memory_data

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error cargando memoria de Shadow: {e}")
            return []

    def _apply_filters(self, memory_data: List[Dict[str, Any]], query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Aplica filtros a los datos de memoria"""
        filtered_data = memory_data.copy()

        # Filtro por categoría
        if 'category' in query_params:
            category = query_params['category']
            filtered_data = [entry for entry in filtered_data
                           if entry.get('entry_category') == category or
                              entry.get('component') == category]

        # Filtro por componente
        if 'component' in query_params:
            component = query_params['component']
            filtered_data = [entry for entry in filtered_data
                           if entry.get('source_component') == component or
                              entry.get('component') == component]

        # Filtro por rango de tiempo
        if 'time_range' in query_params:
            time_range = query_params['time_range']
            filtered_data = self._filter_by_time(filtered_data, time_range)

        # Filtro por versión
        if 'version' in query_params:
            version = query_params['version']
            filtered_data = [entry for entry in filtered_data
                           if entry.get('version_id') == version]

        # Filtro por texto (búsqueda en mensajes y contenido)
        if 'search_text' in query_params:
            search_text = query_params['search_text'].lower()
            filtered_data = [entry for entry in filtered_data
                           if self._contains_text(entry, search_text)]

        # Ordenar por timestamp (más reciente primero por defecto)
        filtered_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Limitar resultados
        limit = query_params.get('limit', 50)
        filtered_data = filtered_data[:limit]

        return filtered_data

    def _filter_by_time(self, data: List[Dict[str, Any]], time_range: str) -> List[Dict[str, Any]]:
        """Filtra entradas por rango de tiempo"""
        try:
            # Parsear rango de tiempo
            if time_range.endswith('h'):
                hours = int(time_range[:-1])
                cutoff_time = datetime.now() - timedelta(hours=hours)
            elif time_range.endswith('d'):
                days = int(time_range[:-1])
                cutoff_time = datetime.now() - timedelta(days=days)
            elif time_range.endswith('m'):
                # Últimos N minutos
                minutes = int(time_range[:-1])
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
            else:
                # Por defecto, últimas 24 horas
                cutoff_time = datetime.now() - timedelta(days=1)

            filtered_data = []
            for entry in data:
                timestamp_str = entry.get('timestamp', '')
                if timestamp_str:
                    try:
                        entry_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if entry_time >= cutoff_time:
                            filtered_data.append(entry)
                    except ValueError:
                        # Si no se puede parsear timestamp, incluir la entrada
                        filtered_data.append(entry)

            return filtered_data

        except (ValueError, TypeError):
            self.logger.warning(f"Error parseando time_range: {time_range}")
            return data

    def _contains_text(self, entry: Dict[str, Any], search_text: str) -> bool:
        """Verifica si una entrada contiene el texto de búsqueda"""
        # Buscar en campos de texto comunes
        text_fields = [
            entry.get('action', ''),
            entry.get('entry_category', ''),
            entry.get('source_component', ''),
            entry.get('component', ''),
            entry.get('agent', '')
        ]

        # Buscar en data_payload si existe
        data_payload = entry.get('data_payload', {})
        if isinstance(data_payload, dict):
            for key, value in data_payload.items():
                if isinstance(value, str):
                    text_fields.append(value)
                elif isinstance(value, list):
                    text_fields.extend([str(item) for item in value])

        # Buscar en details si existe
        details = entry.get('details', {})
        if isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, str):
                    text_fields.append(value)

        # Verificar si el texto de búsqueda está en alguno de los campos
        search_text_lower = search_text.lower()
        return any(search_text_lower in field.lower() for field in text_fields)

    def _generate_metadata(self, all_data: List[Dict[str, Any]], filtered_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera metadatos sobre la consulta"""
        # Estadísticas generales
        total_entries = len(all_data)
        filtered_entries = len(filtered_data)

        # Categorías presentes
        categories = {}
        components = {}
        versions = {}

        for entry in all_data:
            cat = entry.get('entry_category', 'unknown')
            comp = entry.get('source_component', entry.get('component', 'unknown'))
            ver = entry.get('version_id', 'unknown')

            categories[cat] = categories.get(cat, 0) + 1
            components[comp] = components.get(comp, 0) + 1
            versions[ver] = versions.get(ver, 0) + 1

        # Rango de tiempo
        timestamps = [entry.get('timestamp') for entry in all_data if entry.get('timestamp')]
        if timestamps:
            timestamps.sort()
            time_range = {
                "oldest": timestamps[0],
                "newest": timestamps[-1]
            }
        else:
            time_range = None

        return {
            "total_entries": total_entries,
            "filtered_entries": filtered_entries,
            "categories": categories,
            "components": components,
            "versions": versions,
            "time_range": time_range,
            "memory_integrity": self._check_memory_integrity(all_data)
        }

    def _check_memory_integrity(self, memory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verifica la integridad de la cadena de hashes"""
        integrity_status = {
            "valid_entries": 0,
            "invalid_entries": 0,
            "chain_valid": True,
            "details": []
        }

        previous_hash = ""

        for i, entry in enumerate(memory_data):
            entry_hash = entry.get('entry_hash', '')
            stored_prev_hash = entry.get('previous_entry_hash', '')

            # Verificar hash de la entrada actual
            entry_content = json.dumps({
                k: v for k, v in entry.items()
                if k != 'entry_hash'
            }, sort_keys=True, ensure_ascii=False)

            calculated_hash = self._calculate_hash(entry_content)

            if calculated_hash == entry_hash:
                integrity_status["valid_entries"] += 1
            else:
                integrity_status["invalid_entries"] += 1
                integrity_status["chain_valid"] = False
                integrity_status["details"].append(f"Entry {i}: hash mismatch")

            # Verificar cadena con entrada anterior
            if stored_prev_hash != previous_hash:
                integrity_status["chain_valid"] = False
                integrity_status["details"].append(f"Entry {i}: chain break")

            previous_hash = entry_hash

        return integrity_status

    def _calculate_hash(self, content: str) -> str:
        """Calcula hash SHA-256 de contenido"""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def get_context_for_aiphalab(self, query: str = "", time_range: str = "24h") -> str:
        """
        Genera contexto completo para AiphaLab basado en consulta

        Args:
            query: Consulta específica (opcional)
            time_range: Rango de tiempo para filtrar

        Returns:
            String con contexto formateado para AiphaLab
        """
        # Preparar parámetros de consulta
        query_params = {
            'time_range': time_range,
            'limit': 20,
            'include_metadata': True
        }

        if query:
            query_params['search_text'] = query

        # Realizar consulta
        result = self.query_shadow_memory(query_params)

        if result['status'] != 'success':
            return f"ERROR: {result['message']}"

        # Formatear contexto para AiphaLab
        context = self._format_context_for_aiphalab(result, query)

        return context

    def _format_context_for_aiphalab(self, result: Dict[str, Any], original_query: str = "") -> str:
        """Formatea los resultados de consulta para AiphaLab"""
        context_parts = []

        # Header
        context_parts.append("# 🔍 CONTEXTO SHADOW MEMORY")
        context_parts.append(f"**Consulta:** {original_query or 'Estado general del proyecto'}")
        context_parts.append(f"**Timestamp:** {result['query_timestamp']}")
        context_parts.append(f"**Entradas encontradas:** {result['filtered_entries']} de {result['total_entries']}")
        context_parts.append("")

        # Metadatos si disponibles
        if 'metadata' in result:
            meta = result['metadata']
            context_parts.append("## 📊 METADATOS")
            context_parts.append(f"- **Versiones:** {', '.join(meta.get('versions', {}).keys())}")
            context_parts.append(f"- **Componentes:** {', '.join(meta.get('components', {}).keys())}")
            context_parts.append(f"- **Categorías:** {', '.join(meta.get('categories', {}).keys())}")

            if meta.get('time_range'):
                tr = meta['time_range']
                context_parts.append(f"- **Rango temporal:** {tr['oldest']} → {tr['newest']}")

            integrity = meta.get('memory_integrity', {})
            context_parts.append(f"- **Integridad:** {'✅ VÁLIDA' if integrity.get('chain_valid') else '❌ COMPROMETIDA'}")
            context_parts.append("")

        # Entradas filtradas
        context_parts.append("## 📝 ENTRADAS RECIENTES")
        context_parts.append("")

        for i, entry in enumerate(result['data'][:10], 1):  # Limitar a 10 entradas
            context_parts.append(f"### {i}. {entry.get('action', 'Sin título')}")
            context_parts.append(f"**Timestamp:** {entry.get('timestamp', 'N/A')}")
            context_parts.append(f"**Componente:** {entry.get('source_component', entry.get('component', 'N/A'))}")
            context_parts.append(f"**Categoría:** {entry.get('entry_category', 'N/A')}")

            # Agregar detalles específicos según el tipo de entrada
            data_payload = entry.get('data_payload', {})
            if data_payload:
                if 'event_type' in data_payload:
                    # Evento Git
                    event_type = data_payload['event_type']
                    commit_hash = data_payload.get('commit_hash', 'N/A')[:8]
                    context_parts.append(f"**Evento Git:** {event_type} ({commit_hash})")

                    if event_type == 'commit':
                        msg = data_payload.get('commit_message', 'Sin mensaje')
                        files = data_payload.get('files_changed', [])
                        context_parts.append(f"**Mensaje:** {msg}")
                        context_parts.append(f"**Archivos:** {', '.join(files[:3])}{'...' if len(files) > 3 else ''}")

                    elif event_type == 'push':
                        url = data_payload.get('github_url', 'N/A')
                        context_parts.append(f"**URL GitHub:** {url}")

                elif 'analysis_type' in data_payload:
                    # Análisis de código
                    analysis_type = data_payload['analysis_type']
                    files_analyzed = data_payload.get('files_analyzed', 0)
                    context_parts.append(f"**Tipo de análisis:** {analysis_type}")
                    context_parts.append(f"**Archivos analizados:** {files_analyzed}")

            context_parts.append("")

        # Footer con instrucciones
        context_parts.append("---")
        context_parts.append("💡 **Para más detalles específicos, puedes preguntar:**")
        context_parts.append("- 'Muéstrame los últimos commits'")
        context_parts.append("- '¿Qué cambios se hicieron en la semana pasada?'")
        context_parts.append("- '¿Cuál es el estado actual del proyecto?'")
        context_parts.append("- '¿Qué milestones se han completado?'")

        return "\n".join(context_parts)


def create_cli_interface():
    """Crea interfaz de línea de comandos para el bridge"""
    parser = argparse.ArgumentParser(
        description='AiphaLab Bridge - Puente hacia Shadow Memory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

# Consulta general de las últimas 24 horas
python3 shadow/aiphalab_bridge.py --query

# Buscar eventos específicos
python3 shadow/aiphalab_bridge.py --category GIT_EVENT --time-range 7d

# Generar contexto para AiphaLab
python3 shadow/aiphalab_bridge.py --aiphalab-context --query "commits recientes"

# Consulta avanzada con filtros
python3 shadow/aiphalab_bridge.py --component Git_Hook --limit 5 --include-metadata
        """
    )

    parser.add_argument(
        '--memory-path',
        default='./aipha_memory_storage/action_history',
        help='Ruta al directorio de memoria Shadow'
    )

    # Modos de operación
    parser.add_argument(
        '--query',
        action='store_true',
        help='Realizar consulta general'
    )

    parser.add_argument(
        '--aiphalab-context',
        action='store_true',
        help='Generar contexto formateado para AiphaLab'
    )

    # Parámetros de consulta
    parser.add_argument(
        '--category',
        help='Filtrar por categoría (ej: GIT_EVENT, CODE_REFACTOR)'
    )

    parser.add_argument(
        '--component',
        help='Filtrar por componente (ej: Git_Hook, Developer)'
    )

    parser.add_argument(
        '--time-range',
        default='24h',
        help='Rango de tiempo (ej: 1h, 7d, 30m)'
    )

    parser.add_argument(
        '--search-text',
        help='Buscar texto específico en las entradas'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Limitar número de resultados'
    )

    parser.add_argument(
        '--include-metadata',
        action='store_true',
        help='Incluir metadatos en la respuesta'
    )

    parser.add_argument(
        '--output-file',
        help='Guardar resultado en archivo'
    )

    return parser


def main():
    """Función principal"""
    parser = create_cli_interface()
    args = parser.parse_args()

    # Crear bridge
    bridge = AiphaLabBridge(args.memory_path)

    try:
        if args.aiphalab_context:
            # Generar contexto para AiphaLab
            context = bridge.get_context_for_aiphalab(
                query=args.search_text or "",
                time_range=args.time_range
            )

            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(context)
                print(f"✅ Contexto guardado en: {args.output_file}")
            else:
                print(context)

        elif args.query or any([args.category, args.component, args.search_text]):
            # Realizar consulta con filtros
            query_params = {
                'time_range': args.time_range,
                'limit': args.limit,
                'include_metadata': args.include_metadata
            }

            if args.category:
                query_params['category'] = args.category
            if args.component:
                query_params['component'] = args.component
            if args.search_text:
                query_params['search_text'] = args.search_text

            result = bridge.query_shadow_memory(query_params)

            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"✅ Resultados guardados en: {args.output_file}")
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))

        else:
            # Mostrar ayuda
            parser.print_help()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()