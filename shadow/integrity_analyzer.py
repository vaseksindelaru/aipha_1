#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrity_analyzer.py - Analizador de integridad profunda para Shadow
Realiza análisis de checksum, verificación de archivos y validación de integridad
"""

import os
import sys
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Añadir path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class IntegrityAnalyzer:
    """
    Analizador de integridad profunda que verifica checksums, estructura y contenido de archivos
    """

    def __init__(self, repo_path: str = "../Aipha_0.0.1", shadow_memory_path: str = "../aipha_memory_storage/action_history"):
        """Inicializar analizador de integridad"""
        self.repo_path = os.path.abspath(repo_path)
        self.shadow_memory_path = os.path.abspath(shadow_memory_path)
        self.logger = logging.getLogger(__name__)

        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger.info("🔍 Integrity Analyzer inicializado")

    def perform_deep_integrity_analysis(self) -> Dict[str, Any]:
        """
        Realiza análisis de integridad profunda del repositorio

        Returns:
            Dict con resultados completos del análisis
        """
        timestamp = datetime.now().isoformat()

        analysis = {
            'timestamp': timestamp,
            'repository_path': self.repo_path,
            'analysis_type': 'DEEP_INTEGRITY_ANALYSIS',
            'file_checksums': {},
            'structure_validation': {},
            'content_analysis': {},
            'integrity_score': 0,
            'issues_found': [],
            'recommendations': []
        }

        try:
            # 1. Análisis de checksums
            analysis['file_checksums'] = self._calculate_file_checksums()

            # 2. Validación de estructura
            analysis['structure_validation'] = self._validate_repository_structure()

            # 3. Análisis de contenido
            analysis['content_analysis'] = self._analyze_file_contents()

            # 4. Calcular score de integridad
            analysis['integrity_score'] = self._calculate_integrity_score(analysis)

            # 5. Identificar issues
            analysis['issues_found'] = self._identify_integrity_issues(analysis)

            # 6. Generar recomendaciones
            analysis['recommendations'] = self._generate_recommendations(analysis)

            # 7. Registrar en memoria Shadow
            self._register_integrity_analysis(analysis)

            self.logger.info(f"✅ Análisis de integridad completado: Score {analysis['integrity_score']}/100")

        except Exception as e:
            self.logger.error(f"❌ Error en análisis de integridad: {e}")
            analysis['error'] = str(e)

        return analysis

    def _calculate_file_checksums(self) -> Dict[str, Any]:
        """Calcula checksums MD5 y SHA256 de todos los archivos"""
        checksums = {}

        if not os.path.exists(self.repo_path):
            return {'error': f'Repository path does not exist: {self.repo_path}'}

        for root, dirs, files in os.walk(self.repo_path):
            # Excluir directorios .git y __pycache__
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'node_modules']]

            for file in files:
                # Solo archivos de código fuente, excluir archivos compilados y ocultos
                if (file.endswith(('.py', '.json', '.md', '.txt', '.yaml', '.yml')) and
                    not file.endswith(('.pyc', '.pyo')) and
                    not file.startswith('.')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.repo_path)

                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()

                        checksums[relative_path] = {
                            'md5': hashlib.md5(content).hexdigest(),
                            'sha256': hashlib.sha256(content).hexdigest(),
                            'size': len(content),
                            'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        }

                    except Exception as e:
                        checksums[relative_path] = {'error': str(e)}

        return checksums

    def _validate_repository_structure(self) -> Dict[str, Any]:
        """Valida la estructura del repositorio"""
        validation = {
            'expected_files': [
                'README.md',
                'config.json',
                'config_loader.py',
                'main.py',
                'potential_capture_engine.py',
                'shadow.py',
                'strategy.py'
            ],
            'found_files': [],
            'missing_files': [],
            'unexpected_files': [],
            'structure_score': 0
        }

        if not os.path.exists(self.repo_path):
            validation['error'] = f'Repository path does not exist: {self.repo_path}'
            return validation

        # Obtener archivos reales
        real_files = []
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache']]
            for file in files:
                if (file.endswith(('.py', '.json', '.md')) and 
                    not file.endswith(('.pyc', '.pyo')) and 
                    not file.startswith('.')):
                    real_files.append(file)

        validation['found_files'] = real_files

        # Verificar archivos esperados
        for expected in validation['expected_files']:
            if expected not in real_files:
                validation['missing_files'].append(expected)

        # Verificar archivos inesperados
        for found in real_files:
            if found not in validation['expected_files']:
                validation['unexpected_files'].append(found)

        # Calcular score de estructura
        total_expected = len(validation['expected_files'])
        found_expected = total_expected - len(validation['missing_files'])
        validation['structure_score'] = int((found_expected / total_expected) * 100) if total_expected > 0 else 0

        return validation

    def _analyze_file_contents(self) -> Dict[str, Any]:
        """Analiza el contenido de los archivos"""
        content_analysis = {
            'python_files': {},
            'json_files': {},
            'markdown_files': {},
            'syntax_errors': [],
            'import_analysis': {},
            'code_quality_metrics': {}
        }

        if not os.path.exists(self.repo_path):
            return content_analysis

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache']]

            for file in files:
                # Solo archivos de código fuente, excluir archivos compilados y ocultos
                if (file.endswith(('.py', '.json', '.md')) and
                    not file.endswith(('.pyc', '.pyo')) and
                    not file.startswith('.')):
                    file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if file.endswith('.py'):
                        content_analysis['python_files'][relative_path] = self._analyze_python_file(content, file_path)
                    elif file.endswith('.json'):
                        content_analysis['json_files'][relative_path] = self._analyze_json_file(content, file_path)
                    elif file.endswith('.md'):
                        content_analysis['markdown_files'][relative_path] = self._analyze_markdown_file(content, file_path)

                except Exception as e:
                    content_analysis['syntax_errors'].append({
                        'file': relative_path,
                        'error': str(e)
                    })

        return content_analysis

    def _analyze_python_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analiza un archivo Python"""
        analysis = {
            'lines_of_code': len(content.split('\n')),
            'imports': [],
            'functions': [],
            'classes': [],
            'syntax_valid': True,
            'complexity_score': 0
        }

        # Extraer imports
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                analysis['imports'].append(line)

        # Validar sintaxis
        try:
            compile(content, file_path, 'exec')
        except SyntaxError as e:
            analysis['syntax_valid'] = False
            analysis['syntax_error'] = str(e)

        # Contar funciones y clases (simple)
        analysis['functions'] = len([line for line in lines if line.strip().startswith('def ')])
        analysis['classes'] = len([line for line in lines if line.strip().startswith('class ')])

        # Calcular complejidad simple
        analysis['complexity_score'] = analysis['functions'] + (analysis['classes'] * 2)

        return analysis

    def _analyze_json_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analiza un archivo JSON"""
        analysis = {
            'valid_json': True,
            'size': len(content),
            'keys_count': 0,
            'structure': {}
        }

        try:
            data = json.loads(content)
            analysis['keys_count'] = len(data) if isinstance(data, dict) else 0
            analysis['structure'] = self._analyze_json_structure(data)
        except json.JSONDecodeError as e:
            analysis['valid_json'] = False
            analysis['error'] = str(e)

        return analysis

    def _analyze_json_structure(self, data: Any, depth: int = 0) -> Dict[str, Any]:
        """Analiza la estructura de datos JSON"""
        if isinstance(data, dict):
            return {key: self._analyze_json_structure(value, depth + 1) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._analyze_json_structure(item, depth + 1) for item in data[:3]]  # Solo primeros 3 elementos
        else:
            return type(data).__name__

    def _analyze_markdown_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analiza un archivo Markdown"""
        analysis = {
            'lines_count': len(content.split('\n')),
            'headers': [],
            'code_blocks': 0,
            'links': 0
        }

        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('#'):
                analysis['headers'].append(line.strip())
            if '```' in line:
                analysis['code_blocks'] += 1
            if '[' in line and '](' in line:
                analysis['links'] += 1

        return analysis

    def _calculate_integrity_score(self, analysis: Dict[str, Any]) -> int:
        """Calcula el score general de integridad (0-100)"""
        score = 100

        # Penalizar archivos faltantes
        structure = analysis.get('structure_validation', {})
        missing_files = len(structure.get('missing_files', []))
        score -= missing_files * 15

        # Penalizar errores de sintaxis
        content = analysis.get('content_analysis', {})
        syntax_errors = len(content.get('syntax_errors', []))
        score -= syntax_errors * 20

        # Penalizar archivos inesperados
        unexpected_files = len(structure.get('unexpected_files', []))
        score -= unexpected_files * 5

        # Bonus por estructura correcta
        if structure.get('structure_score', 0) == 100:
            score += 10

        return max(0, min(100, score))

    def _identify_integrity_issues(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica issues de integridad"""
        issues = []

        structure = analysis.get('structure_validation', {})
        content = analysis.get('content_analysis', {})

        # Archivos faltantes
        for missing in structure.get('missing_files', []):
            issues.append({
                'type': 'MISSING_FILE',
                'severity': 'HIGH',
                'description': f'Archivo esperado faltante: {missing}',
                'recommendation': 'Verificar si el archivo fue eliminado o movido'
            })

        # Archivos inesperados
        for unexpected in structure.get('unexpected_files', []):
            issues.append({
                'type': 'UNEXPECTED_FILE',
                'severity': 'MEDIUM',
                'description': f'Archivo inesperado encontrado: {unexpected}',
                'recommendation': 'Verificar si este archivo debe estar en el repositorio'
            })

        # Errores de sintaxis
        for error in content.get('syntax_errors', []):
            issues.append({
                'type': 'SYNTAX_ERROR',
                'severity': 'HIGH',
                'description': f'Error de sintaxis en {error["file"]}: {error["error"]}',
                'recommendation': 'Corregir el error de sintaxis en el archivo'
            })

        return issues

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []

        score = analysis.get('integrity_score', 0)
        issues = analysis.get('issues_found', [])

        if score >= 90:
            recommendations.append("✅ Integridad excelente - No se requieren acciones inmediatas")
        elif score >= 70:
            recommendations.append("⚠️ Integridad buena - Revisar issues menores")
        else:
            recommendations.append("🚨 Integridad comprometida - Revisar issues críticos inmediatamente")

        # Recomendaciones específicas
        if issues:
            recommendations.append(f"Resolver {len(issues)} issues identificados")

        structure = analysis.get('structure_validation', {})
        if structure.get('missing_files'):
            recommendations.append("Recuperar archivos faltantes del repositorio")

        content = analysis.get('content_analysis', {})
        if content.get('syntax_errors'):
            recommendations.append("Corregir errores de sintaxis en archivos Python")

        return recommendations

    def _register_integrity_analysis(self, analysis: Dict[str, Any]):
        """Registra el análisis de integridad en memoria Shadow"""
        shadow_entry = {
            "timestamp": analysis['timestamp'],
            "action": f"Shadow Integrity Analysis: Deep repository integrity check - Score {analysis['integrity_score']}/100",
            "agent": "IntegrityAnalyzer",
            "component": "integrity_analysis",
            "status": "success",
            "details": {
                "analysis_type": "DEEP_INTEGRITY_ANALYSIS",
                "integrity_score": analysis['integrity_score'],
                "issues_found": len(analysis['issues_found']),
                "files_analyzed": len(analysis['file_checksums']),
                "structure_score": analysis['structure_validation'].get('structure_score', 0),
                "repository_path": analysis['repository_path'],
                "recommendations": analysis['recommendations']
            }
        }

        # Guardar en memoria Shadow
        memory_file = os.path.join(self.shadow_memory_path, 'current_history.json')

        try:
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []

            history.append(shadow_entry)

            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            self.logger.info(f"✅ Análisis de integridad registrado en Shadow memory")

        except Exception as e:
            self.logger.error(f"❌ Error registrando análisis en Shadow memory: {e}")

    def compare_checksums(self, previous_checksums: Dict[str, Any]) -> Dict[str, Any]:
        """Compara checksums actuales con anteriores para detectar cambios"""
        current_checksums = self._calculate_file_checksums()

        comparison = {
            'timestamp': datetime.now().isoformat(),
            'files_changed': [],
            'files_added': [],
            'files_removed': [],
            'files_unchanged': []
        }

        # Archivos actuales
        current_files = set(current_checksums.keys())
        previous_files = set(previous_checksums.keys())

        # Archivos añadidos
        comparison['files_added'] = list(current_files - previous_files)

        # Archivos eliminados
        comparison['files_removed'] = list(previous_files - current_files)

        # Archivos comunes - verificar cambios
        common_files = current_files & previous_files
        for file in common_files:
            current_md5 = current_checksums[file].get('md5')
            previous_md5 = previous_checksums[file].get('md5')

            if current_md5 != previous_md5:
                comparison['files_changed'].append({
                    'file': file,
                    'previous_md5': previous_md5,
                    'current_md5': current_md5
                })
            else:
                comparison['files_unchanged'].append(file)

        return comparison


def main():
    """Función principal - Demo del analizador de integridad"""
    print("🔍 ANALIZADOR DE INTEGRIDAD PROFUNDA - SHADOW")
    print("=" * 60)

    analyzer = IntegrityAnalyzer()

    print("\n🚀 REALIZANDO ANÁLISIS DE INTEGRIDAD PROFUNDA...")
    print("=" * 60)

    analysis = analyzer.perform_deep_integrity_analysis()

    print(f"\n📊 RESULTADOS DEL ANÁLISIS:")
    print("=" * 60)
    print(f"⏰ Timestamp: {analysis['timestamp']}")
    print(f"📁 Repositorio: {analysis['repository_path']}")
    print(f"🎯 Score de Integridad: {analysis['integrity_score']}/100")

    print(f"\n📋 VALIDACIÓN DE ESTRUCTURA:")
    structure = analysis['structure_validation']
    print(f"   ✅ Archivos encontrados: {len(structure['found_files'])}")
    print(f"   ❌ Archivos faltantes: {len(structure['missing_files'])}")
    print(f"   ⚠️ Archivos inesperados: {len(structure['unexpected_files'])}")
    print(f"   📊 Score de estructura: {structure['structure_score']}/100")

    if structure['missing_files']:
        print(f"   🚨 FALTAN: {', '.join(structure['missing_files'])}")

    print(f"\n🔍 ANÁLISIS DE CONTENIDO:")
    content = analysis['content_analysis']
    print(f"   📄 Archivos Python: {len(content['python_files'])}")
    print(f"   📋 Archivos JSON: {len(content['json_files'])}")
    print(f"   📝 Archivos Markdown: {len(content['markdown_files'])}")
    print(f"   ❌ Errores de sintaxis: {len(content['syntax_errors'])}")

    print(f"\n⚠️ ISSUES IDENTIFICADOS: {len(analysis['issues_found'])}")
    for issue in analysis['issues_found'][:5]:  # Mostrar primeros 5
        print(f"   {issue['severity']}: {issue['description']}")

    print(f"\n💡 RECOMENDACIONES:")
    for rec in analysis['recommendations']:
        print(f"   • {rec}")

    print(f"\n✅ ANÁLISIS COMPLETADO - SCORE: {analysis['integrity_score']}/100")


if __name__ == "__main__":
    main()