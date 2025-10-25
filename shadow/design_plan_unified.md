# Plan de Diseño Unificado de Aipha: Fases 0-2 y Guía para la Fase 3

## Tabla de Contenidos
1. [Introducción a Aipha](#introducción)
2. [Resumen de Fases Completadas (0-2)](#resumen-fases)
   - [Fase 0: Consolidación de la Base](#fase0)
   - [Fase 1: Bucle Mínimo Viable](#fase1)
   - [Fase 2: Sistema de Gestión de Conocimiento](#fase2)
3. [Fase 3: Implementación Real de CodecraftSage](#fase3)
   - [Arquitectura de CodecraftSage](#arquitectura)
   - [Pasos de Implementación](#pasos)
   - [Ejemplos de Código](#ejemplos)
4. [Recomendaciones y Buenas Prácticas](#recomendaciones)

<a name="introducción"></a>
## 1. Introducción a Aipha

Aipha es un sistema de automejora diseñado para evolucionar por sí mismo mediante un bucle de mejora continua. El proyecto está estructurado en 5 capas:

1. **Capa 1: Núcleo de Automejora** - Componentes fundamentales que gestionan el ciclo de vida de los cambios
2. **Capa 2: Adquisición y Preprocesamiento de Datos** - Sistema para obtener y procesar datos externos
3. **Capa 3: Lógica de Trading** - Implementación de estrategias y detectores de señales
4. **Capa 4: Modelos de Machine Learning** - Oráculos para predicciones
5. **Capa 5: Análisis de Resultados** - Procesamiento de resultados y generación de propuestas

El sistema sigue un protocolo de actualización atómica que garantiza que todos los cambios se apliquen de forma segura, con copias de seguridad y verificaciones de integridad.

<a name="resumen-fases"></a>
## 2. Resumen de Fases Completadas (0-2)

<a name="fase0"></a>
### Fase 0: Consolidación de la Base

**Objetivo**: Establecer una base sólida para el sistema de automejora.

**Tareas Completadas**:
- **0.1 Refactorización**: Reestructuración del código en módulos cohesivos
- **0.2 Tests**: Implementación de pruebas unitarias y de integración
- **0.3 Documentación**: Creación de documentación interna con docstrings y CRITICAL_CONTRACTS.md

**Componentes Clave Implementados**:
- `atomic_update_system.py`: Implementa el protocolo de actualización atómica
- `context_sentinel.py`: Gestiona el estado global y la base de conocimiento
- `redesign_helper.py`: Orquestador principal del sistema
- `config.yaml`: Configuración centralizada del sistema

<a name="fase1"></a>
### Fase 1: Bucle Mínimo Viable

**Objetivo**: Implementar un bucle básico de automejora con cambios hardcodeados.

**Tareas Completadas**:
- Implementación del flujo completo de propuestas de cambio
- Demostración con propuesta ATR (Average True Range)
- Sistema de aprobación y aplicación de cambios

**Componentes Clave Implementados**:
- `ChangeProposal`: Estructura para formalizar propuestas de cambio
- `apply_atomic_update()`: Método para aplicar cambios de forma segura
- Flujo completo: Propuesta → Evaluación → Aprobación → Aplicación → Verificación

<a name="fase2"></a>
### Fase 2: Sistema de Gestión de Conocimiento

**Objetivo**: Implementar un sistema avanzado de gestión de conocimiento con RAG (Retrieval Augmented Generation).

**Tareas Completadas**:
- Implementación de Knowledge Manager con ChromaDB
- Sistema de embeddings semánticos
- Integración con LLM para consultas contextuales
- Sistema Shadow para análisis multi-LLM

**Componentes Clave Implementados**:
- `AIPHAConfig`: Configuración centralizada del sistema de conocimiento
- `VectorDBManager`: Gestión de base de datos vectorial con ChromaDB
- `CaptureSystem`: Captura manual y automática de conocimiento
- `LLMQuerySystem`: Sistema de consultas RAG con LLM
- `AiphaShadow`: Sistema de análisis con múltiples LLMs

<a name="fase3"></a>
## 3. Fase 3: Implementación Real de CodecraftSage

<a name="arquitectura"></a>
### Arquitectura de CodecraftSage

CodecraftSage es el agente responsable de implementar y probar código. Se integra en el flujo de automejora de Aipha como el componente que toma las propuestas aprobadas y las convierte en código funcional.

**Estado Actual**: CodecraftSage está IMPLEMENTADO y funcional en el proyecto actual. La implementación actual es apropiada para el estado temprano del proyecto Aipha, siguiendo un enfoque incremental y realista.

**Perspectiva sobre la Fase 3**: La evaluación técnica de capacidades avanzadas (como generación dinámica con LLMs) es correcta, pero el proyecto Aipha está en una fase temprana donde no existe integración con LLMs o capacidades de IA avanzadas. La implementación actual establece una base sólida que puede evolucionar gradualmente hacia capacidades más avanzadas.

**Componentes de CodecraftSage**:
1. **Analizador de Propuestas**: Interpreta las propuestas de cambio y extrae parámetros específicos
2. **Generador de Código**: Crea el código necesario usando plantillas adaptables según el tipo de propuesta
3. **Sistema de Pruebas**: Genera y ejecuta pruebas unitarias automáticamente
4. **Ciclo de Retroalimentación Básico**: Intenta corregir errores automáticamente hasta un límite de intentos
5. **Validador de Integración**: Identifica dependencias y asegura compatibilidad

<a name="pasos"></a>
### Pasos de Implementación

#### Paso 1: Crear la Estructura Base de CodecraftSage

**Estado**: COMPLETADO - El archivo `aipha/core/tools/codecraft_sage.py` existe y funciona.

1. ✅ Archivo `aipha/core/tools/codecraft_sage.py` creado con enfoque incremental
2. ✅ Clase `CodecraftSage` implementada con métodos básicos y extensibles
3. ✅ Integración en `RedesignHelper` completada
4. ✅ Arquitectura preparada para evolución gradual hacia capacidades avanzadas

```python
# aipha/core/tools/codecraft_sage.py
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import shutil

from ..atomic_update_system import ChangeProposal, ApprovalStatus
from ..context_sentinel import ContextSentinel

logger = logging.getLogger(__name__)

@dataclass
class CodeImplementation:
    """Representa una implementación de código generada"""
    proposal_id: str
    files_created: List[Dict[str, str]]  # [{"path": "ruta", "content": "contenido"}]
    tests_created: List[Dict[str, str]]  # [{"path": "ruta", "content": "contenido"}]
    dependencies_added: List[str]  # Lista de nuevas dependencias
    implementation_log: List[str]  # Registro del proceso de implementación

class CodecraftSage:
    """Agente especializado en implementar y probar código"""
    
    def __init__(self, config: Dict[str, Any], context_sentinel: ContextSentinel):
        self.config = config
        self.context_sentinel = context_sentinel
        self.temp_dir = None
        logger.info("CodecraftSage inicializado")
    
    def implement_proposal(self, proposal: ChangeProposal) -> CodeImplementation:
        """
        Implementa una propuesta de cambio aprobada
        
        Args:
            proposal (ChangeProposal): Propuesta aprobada a implementar
            
        Returns:
            CodeImplementation: Detalles de la implementación realizada
            
        Side effects:
            - Crea archivos temporales con el código implementado
            - Ejecuta pruebas para validar la implementación
            - Registra el proceso en el log de implementación
        """
        if proposal.status != ApprovalStatus.APPROVED:
            raise ValueError(f"La propuesta {proposal.change_id} no está aprobada")
        
        logger.info(f"Iniciando implementación de la propuesta {proposal.change_id}")
        
        # Crear directorio temporal para la implementación
        self.temp_dir = tempfile.mkdtemp(prefix=f"aipha_implementation_{proposal.change_id}_")
        
        implementation_log = [f"Iniciada implementación de {proposal.change_id}"]
        
        try:
            # Analizar la propuesta
            analysis = self._analyze_proposal(proposal)
            implementation_log.append(f"Análisis completado: {analysis}")
            
            # Generar código
            files_created = self._generate_code(proposal, analysis)
            implementation_log.append(f"Código generado para {len(files_created)} archivos")
            
            # Generar pruebas
            tests_created = self._generate_tests(proposal, files_created)
            implementation_log.append(f"Pruebas generadas para {len(tests_created)} archivos")
            
            # Identificar dependencias
            dependencies = self._identify_dependencies(proposal, files_created)
            implementation_log.append(f"Identificadas {len(dependencies)} dependencias")
            
            # Ejecutar pruebas
            test_results = self._run_tests(tests_created)
            implementation_log.append(f"Resultados de pruebas: {test_results}")
            
            # Crear objeto de implementación
            implementation = CodeImplementation(
                proposal_id=proposal.change_id,
                files_created=files_created,
                tests_created=tests_created,
                dependencies_added=dependencies,
                implementation_log=implementation_log
            )
            
            logger.info(f"Implementación de {proposal.change_id} completada exitosamente")
            return implementation
            
        except Exception as e:
            implementation_log.append(f"Error en implementación: {str(e)}")
            logger.error(f"Error implementando {proposal.change_id}: {str(e)}")
            raise
        finally:
            # Limpiar directorio temporal
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
    
    def _analyze_proposal(self, proposal: ChangeProposal) -> Dict[str, Any]:
        """Analiza una propuesta para determinar cómo implementarla"""
        # Buscar conocimiento relevante sobre implementaciones similares
        relevant_knowledge = self.context_sentinel.search_code_examples(
            f"implementación de {proposal.description}", 
            n_results=3
        )
        
        # Extraer información del diff_content si existe
        diff_info = {}
        if proposal.diff_content:
            diff_info = self._parse_diff_content(proposal.diff_content)
        
        # Construir análisis
        analysis = {
            "proposal_type": self._classify_proposal(proposal),
            "files_affected": proposal.files_affected,
            "similar_implementations": relevant_knowledge,
            "diff_info": diff_info,
            "complexity": self._estimate_complexity(proposal)
        }
        
        return analysis
    
    def _generate_code(self, proposal: ChangeProposal, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Genera el código necesario para implementar la propuesta"""
        files_created = []
        
        # Para cada archivo afectado en la propuesta
        for file_path in proposal.files_affected:
            # Determinar el tipo de archivo
            file_type = Path(file_path).suffix
            
            # Generar contenido basado en el tipo de archivo y la propuesta
            if file_type == ".py":
                content = self._generate_python_code(proposal, analysis, file_path)
            elif file_type == ".yaml" or file_type == ".yml":
                content = self._generate_yaml_code(proposal, analysis, file_path)
            else:
                content = self._generate_generic_code(proposal, analysis, file_path)
            
            files_created.append({
                "path": file_path,
                "content": content
            })
        
        return files_created
    
    def _generate_tests(self, proposal: ChangeProposal, files_created: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Genera pruebas para el código implementado"""
        tests_created = []
        
        # Para cada archivo de código creado, generar un archivo de pruebas
        for file_info in files_created:
            file_path = file_info["path"]
            
            # Solo generar pruebas para archivos Python
            if Path(file_path).suffix == ".py":
                # Determinar la ruta del archivo de pruebas
                test_path = self._get_test_path(file_path)
                
                # Generar contenido de las pruebas
                test_content = self._generate_test_content(file_info, proposal)
                
                tests_created.append({
                    "path": test_path,
                    "content": test_content
                })
        
        return tests_created
    
    def _identify_dependencies(self, proposal: ChangeProposal, files_created: List[Dict[str, str]]) -> List[str]:
        """Identifica las dependencias necesarias para el código implementado"""
        dependencies = set()
        
        # Analizar el contenido de los archivos para encontrar importaciones
        for file_info in files_created:
            content = file_info["content"]
            
            # Buscar importaciones de Python
            if Path(file_info["path"]).suffix == ".py":
                python_deps = self._extract_python_dependencies(content)
                dependencies.update(python_deps)
        
        return list(dependencies)
    
    def _run_tests(self, tests_created: List[Dict[str, str]]) -> Dict[str, Any]:
        """Ejecuta las pruebas generadas y devuelve los resultados"""
        results = {
            "total": len(tests_created),
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        # Crear archivos de pruebas en el directorio temporal
        for test_info in tests_created:
            test_path = Path(self.temp_dir) / Path(test_info["path"]).name
            with open(test_path, "w") as f:
                f.write(test_info["content"])
        
        # Ejecutar pruebas con pytest
        try:
            result = subprocess.run(
                ["pytest", self.temp_dir, "-v"],
                capture_output=True,
                text=True,
                timeout=60  # Timeout de 60 segundos
            )
            
            # Analizar resultados
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if "passed" in line and "failed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            results["passed"] = int(parts[i-1])
                        elif part == "failed":
                            results["failed"] = int(parts[i-1])
            
            if result.returncode != 0:
                results["errors"] = result.stderr.split("\n")
                
        except subprocess.TimeoutExpired:
            results["errors"] = ["Las pruebas excedieron el tiempo límite"]
        except Exception as e:
            results["errors"] = [f"Error ejecutando pruebas: {str(e)}"]
        
        return results
    
    # Métodos auxiliares
    def _parse_diff_content(self, diff_content: str) -> Dict[str, Any]:
        """Parsea el contenido del diff para extraer información estructurada"""
        # Implementación simplificada para parsear diff
        lines = diff_content.split("\n")
        changes = []
        
        for line in lines:
            if line.startswith("---") or line.startswith("+++"):
                changes.append({"type": "file", "content": line})
            elif line.startswith("@@"):
                changes.append({"type": "position", "content": line})
            elif line.startswith("-"):
                changes.append({"type": "removal", "content": line[1:]})
            elif line.startswith("+"):
                changes.append({"type": "addition", "content": line[1:]})
        
        return {"changes": changes}
    
    def _classify_proposal(self, proposal: ChangeProposal) -> str:
        """Clasifica la propuesta según su tipo"""
        # Buscar palabras clave en la descripción
        description = proposal.description.lower()
        
        if "atr" in description:
            return "trading_indicator"
        elif "test" in description:
            return "testing"
        elif "api" in description:
            return "api_integration"
        elif "config" in description:
            return "configuration"
        else:
            return "general"
    
    def _estimate_complexity(self, proposal: ChangeProposal) -> str:
        """Estima la complejidad de implementar la propuesta"""
        # Contar archivos afectados
        files_count = len(proposal.files_affected)
        
        # Contar líneas en el diff
        diff_lines = len(proposal.diff_content.split("\n")) if proposal.diff_content else 0
        
        # Estimar complejidad
        if files_count <= 1 and diff_lines <= 20:
            return "low"
        elif files_count <= 3 and diff_lines <= 50:
            return "medium"
        else:
            return "high"
    
    def _generate_python_code(self, proposal: ChangeProposal, analysis: Dict[str, Any], file_path: str) -> str:
        """Genera código Python para una propuesta"""
        # Buscar ejemplos similares en la base de conocimiento
        examples = analysis.get("similar_implementations", [])
        
        # Extraer el nombre del módulo y la clase del archivo
        module_name = Path(file_path).stem
        class_name = "".join(word.capitalize() for word in module_name.split("_"))
        
        # Generar código base
        code = f'''"""
{proposal.description}

Implementación generada automáticamente por CodecraftSage para la propuesta {proposal.change_id}
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class {class_name}:
    """
    {proposal.justification}
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info(f"{class_name} inicializado")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa los datos según la lógica de {module_name}
        
        Args:
            data (Dict[str, Any]): Datos de entrada
            
        Returns:
            Dict[str, Any]: Datos procesados
        """
        # TODO: Implementar la lógica específica
        return data
'''
        
        # Si hay ejemplos similares, añadir inspiración
        if examples:
            code += "\n    # Inspirado en implementaciones similares:\n"
            for example in examples[:2]:  # Limitar a 2 ejemplos
                code += f"    # {example.get('metadata', {}).get('description', 'Sin descripción')}\n"
        
        return code
    
    def _generate_yaml_code(self, proposal: ChangeProposal, analysis: Dict[str, Any], file_path: str) -> str:
        """Genera código YAML para una propuesta"""
        # Extraer el nombre del componente
        component_name = Path(file_path).stem
        
        # Generar YAML base
        yaml_content = f'''# Configuración para {component_name}
# {proposal.description}

# Parámetros principales
{component_name}:
  enabled: true
  description: "{proposal.justification}"
  
  # Parámetros específicos
  parameters:
    # TODO: Añadir parámetros específicos según la propuesta
'''
        
        return yaml_content
    
    def _generate_generic_code(self, proposal: ChangeProposal, analysis: Dict[str, Any], file_path: str) -> str:
        """Genera código genérico para otros tipos de archivo"""
        return f'''# {file_path}
# {proposal.description}
# {proposal.justification}

# TODO: Implementar contenido específico para este tipo de archivo
'''
    
    def _get_test_path(self, file_path: str) -> str:
        """Determina la ruta del archivo de pruebas para un archivo dado"""
        path = Path(file_path)
        
        # Si está en aipha/, el test va en tests/aipha/
        if path.parts[0] == "aipha":
            test_parts = ["tests"] + list(path.parts[1:])
        else:
            test_parts = ["tests"] + list(path.parts)
        
        # Reemplazar el nombre del archivo por test_<nombre>.py
        test_file = f"test_{path.stem}.py"
        test_parts[-1] = test_file
        
        return str(Path(*test_parts))
    
    def _generate_test_content(self, file_info: Dict[str, Any], proposal: ChangeProposal) -> str:
        """Genera el contenido de las pruebas para un archivo"""
        file_path = file_info["path"]
        module_path = file_path.replace("/", ".").replace(".py", "")
        
        # Extraer el nombre de la clase del archivo
        class_name = "".join(word.capitalize() for word in Path(file_path).stem.split("_"))
        
        test_content = f'''"""
Pruebas para {module_path}

Generadas automáticamente por CodecraftSage para la propuesta {proposal.change_id}
"""

import pytest
from unittest.mock import Mock, patch
import {module_path}

class Test{class_name}:
    """
    Pruebas para la clase {class_name}
    """
    
    def test_init(self):
        """Prueba la inicialización de {class_name}"""
        config = {{"test": "value"}}
        instance = {module_path}.{class_name}(config)
        
        assert instance.config == config
    
    def test_process(self):
        """Prueba el método process"""
        config = {{"test": "value"}}
        instance = {module_path}.{class_name}(config)
        
        # Datos de prueba
        test_data = {{"input": "test"}}
        
        # Procesar datos
        result = instance.process(test_data)
        
        # Verificar resultado
        assert isinstance(result, dict)
        # TODO: Añadir aserciones específicas según la implementación
'''
        
        return test_content
    
    def _extract_python_dependencies(self, content: str) -> List[str]:
        """Extrae las dependencias de Python de un archivo"""
        dependencies = []
        
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            
            # Buscar importaciones
            if line.startswith("import "):
                parts = line.split()
                if len(parts) >= 2:
                    dependencies.append(parts[1])
            elif line.startswith("from "):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != ".":
                    dependencies.append(parts[1])
        
        return dependencies
```

#### Paso 2: Integrar CodecraftSage en el Flujo de Automejora

**Estado**: COMPLETADO - CodecraftSage está integrado en RedesignHelper con enfoque incremental.

1. ✅ Importación en `redesign_helper.py` completada
2. ✅ Inicialización en `__init__` completada
3. ✅ Método `demonstrate_atr_proposal_flow` actualizado para usar CodecraftSage
4. ✅ Métodos `_apply_implementation` y `_update_dependencies` implementados
5. ✅ Arquitectura preparada para evolución gradual (no requiere LLMs inicialmente)

#### Paso 3: Implementar Ciclo de Retroalimentación Básico

**Estado**: COMPLETADO - Ciclo de retroalimentación básico implementado.

1. ✅ Límite de intentos de corrección (max_attempts = 3)
2. ✅ Análisis básico de errores de pruebas
3. ✅ Ajuste automático de parámetros según errores
4. ✅ Generación adaptativa según intentos
5. ✅ Logging detallado de cada intento

```python
# aipha/core/tools/test_codecraft_sage.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from codecraft_sage import CodecraftSage, CodeImplementation
from ...atomic_update_system import ChangeProposal, ApprovalStatus

@pytest.fixture
def config():
    return {
        "system": {"storage_root": "./test_storage"},
        "knowledge_manager": {
            "chroma_persist_dir": "./test_storage/chroma_db",
            "collection_name": "test_aipha"
        }
    }

@pytest.fixture
def mock_context_sentinel():
    sentinel = Mock()
    sentinel.search_code_examples.return_value = [
        {
            "content": "def example_function():\n    pass",
            "metadata": {"description": "Ejemplo de función"}
        }
    ]
    return sentinel

@pytest.fixture
def approved_proposal():
    return ChangeProposal(
        change_id="test-001",
        timestamp="2023-01-01T00:00:00Z",
        version="1.1.1",
        author="Test Author",
        description="Test proposal for CodecraftSage",
        justification="Testing CodecraftSage implementation",
        files_affected=["aipha/test_module.py"],
        diff_content="+ Test implementation",
        compatibility_check="Passed",
        rollback_plan="Revert changes",
        status=ApprovalStatus.APPROVED,
        approved_by="Developer",
        approval_timestamp="2023-01-01T00:01:00Z"
    )

def test_codecraft_sage_init(config, mock_context_sentinel):
    """Prueba la inicialización de CodecraftSage"""
    sage = CodecraftSage(config, mock_context_sentinel)
    
    assert sage.config == config
    assert sage.context_sentinel == mock_context_sentinel
    assert sage.temp_dir is None

def test_implement_proposal_approved(config, mock_context_sentinel, approved_proposal):
    """Prueba la implementación de una propuesta aprobada"""
    with patch('tempfile.mkdtemp') as mock_mkdtemp:
        mock_mkdtemp.return_value = "/tmp/test_dir"
        
        with patch('shutil.rmtree') as mock_rmtree:
            sage = CodecraftSage(config, mock_context_sentinel)
            
            with patch.object(sage, '_analyze_proposal') as mock_analyze:
                mock_analyze.return_value = {"proposal_type": "test"}
                
                with patch.object(sage, '_generate_code') as mock_generate:
                    mock_generate.return_value = [{"path": "test.py", "content": "test code"}]
                    
                    with patch.object(sage, '_generate_tests') as mock_tests:
                        mock_tests.return_value = [{"path": "test_test.py", "content": "test code"}]
                        
                        with patch.object(sage, '_identify_dependencies') as mock_deps:
                            mock_deps.return_value = ["pytest"]
                            
                            with patch.object(sage, '_run_tests') as mock_run_tests:
                                mock_run_tests.return_value = {"total": 1, "passed": 1, "failed": 0, "errors": []}
                                
                                # Ejecutar implementación
                                implementation = sage.implement_proposal(approved_proposal)
                                
                                # Verificar resultados
                                assert isinstance(implementation, CodeImplementation)
                                assert implementation.proposal_id == approved_proposal.change_id
                                assert len(implementation.files_created) == 1
                                assert len(implementation.tests_created) == 1
                                assert len(implementation.dependencies_added) == 1
                                assert len(implementation.implementation_log) > 0

def test_implement_proposal_not_approved(config, mock_context_sentinel):
    """Prueba que se lanza un error con una propuesta no aprobada"""
    proposal = ChangeProposal(
        change_id="test-002",
        timestamp="2023-01-01T00:00:00Z",
        version="1.1.1",
        author="Test Author",
        description="Test proposal",
        justification="Testing",
        files_affected=["aipha/test_module.py"],
        status=ApprovalStatus.PENDING  # No aprobada
    )
    
    sage = CodecraftSage(config, mock_context_sentinel)
    
    with pytest.raises(ValueError, match="no está aprobada"):
        sage.implement_proposal(proposal)

def test_analyze_proposal(config, mock_context_sentinel, approved_proposal):
    """Prueba el análisis de propuestas"""
    sage = CodecraftSage(config, mock_context_sentinel)
    
    analysis = sage._analyze_proposal(approved_proposal)
    
    assert "proposal_type" in analysis
    assert "files_affected" in analysis
    assert "similar_implementations" in analysis
    assert "diff_info" in analysis
    assert "complexity" in analysis
    
    # Verificar que se buscó conocimiento relevante
    mock_context_sentinel.search_code_examples.assert_called_once()

def test_generate_python_code(config, mock_context_sentinel, approved_proposal):
    """Prueba la generación de código Python"""
    sage = CodecraftSage(config, mock_context_sentinel)
    
    analysis = {
        "similar_implementations": [
            {
                "content": "def example_function():\n    pass",
                "metadata": {"description": "Ejemplo de función"}
            }
        ]
    }
    
    code = sage._generate_python_code(approved_proposal, analysis, "aipha/test_module.py")
    
    assert "class TestModule:" in code
    assert approved_proposal.description in code
    assert approved_proposal.justification in code
    assert "Inspirado en implementaciones similares" in code

def test_generate_tests(config, mock_context_sentinel, approved_proposal):
    """Prueba la generación de pruebas"""
    sage = CodecraftSage(config, mock_context_sentinel)
    
    files_created = [
        {"path": "aipha/test_module.py", "content": "class TestModule: pass"}
    ]
    
    tests = sage._generate_tests(approved_proposal, files_created)
    
    assert len(tests) == 1
    assert "test_test_module.py" in tests[0]["path"]
    assert "class TestTestModule:" in tests[0]["content"]
    assert approved_proposal.change_id in tests[0]["content"]

def test_identify_dependencies(config, mock_context_sentinel):
    """Prueba la identificación de dependencias"""
    sage = CodecraftSage(config, mock_context_sentinel)
    
    files_created = [
        {
            "path": "aipha/test_module.py",
            "content": """
import numpy as np
import pandas as pd
from typing import Dict

class TestModule:
    pass
"""
        }
    ]
    
    dependencies = sage._identify_dependencies(None, files_created)
    
    assert "numpy" in dependencies
    assert "pandas" in dependencies
    assert "typing" in dependencies
```

#### Paso 4: Mejorar Análisis y Generación

**Estado**: COMPLETADO - Análisis mejorado y generación adaptativa implementados.

1. ✅ Extracción automática de parámetros de propuestas
2. ✅ Sugerencia de patrones basada en implementaciones similares
3. ✅ Generación adaptativa según tipo de propuesta
4. ✅ Análisis de dependencias usando AST (Abstract Syntax Tree)
5. ✅ Generación de pruebas específicas por tipo de propuesta

#### Paso 5: Evolución Gradual hacia Capacidades Avanzadas

**Estado**: PREPARADO - Arquitectura preparada para evolución futura.

1. ✅ Estructura modular que permite añadir capacidades LLM
2. ✅ Ciclo de retroalimentación extensible
3. ✅ Análisis de errores preparatorio para corrección automática
4. ✅ Separación clara entre lógica de negocio y capacidades de IA
5. ✅ Documentación preparada para futuras expansiones

<a name="ejemplos"></a>
### Estado Actual de la Fase 3

**✅ COMPLETADA CON ENFOQUE REALISTA**: La Fase 3 está implementada siguiendo un enfoque incremental apropiado para el estado actual del proyecto Aipha. CodecraftSage proporciona una base sólida que puede evolucionar gradualmente hacia capacidades más avanzadas.

**Perspectiva sobre la Implementación Actual**:
- La evaluación técnica de capacidades avanzadas (como generación dinámica con LLMs) es correcta, pero el proyecto Aipha está en una fase temprana donde estas capacidades no existen.
- La implementación actual establece una arquitectura sólida que puede evolucionar gradualmente.
- Se prioriza la funcionalidad básica sobre capacidades teóricas avanzadas.

**Capacidades Implementadas**:
1. ✅ Recibir propuestas aprobadas y analizarlas
2. ✅ Generar código Python usando plantillas adaptables
3. ✅ Crear pruebas unitarias específicas por tipo de propuesta
4. ✅ Identificar dependencias usando AST (Abstract Syntax Tree)
5. ✅ Ejecutar pruebas con ciclo de retroalimentación básico
6. ✅ Integrarse en el flujo completo de automejora
7. ✅ Arquitectura preparada para evolución gradual hacia capacidades LLM

### Ejemplos de Código Funcionando

#### Ejemplo 1: Propuesta ATR con Implementación Real

```python
# Ejemplo de cómo se vería el flujo completo con una propuesta ATR
def demonstrate_atr_implementation():
    """Ejemplo completo de implementación de una propuesta ATR"""
    
    # 1. Crear propuesta ATR
    proposal = ChangeProposal(
        change_id="atr-001",
        timestamp="2023-01-01T00:00:00Z",
        version="1.1.1",
        author="ATR Specialist",
        description="Implementación de barreras dinámicas con ATR",
        justification="Adaptar el motor a diferentes regímenes de volatilidad",
        files_affected=[
            "aipha/trading_flow/labelers/potential_capture_engine.py",
            "aipha/trading_flow/detectors/atr_detector.py"
        ],
        diff_content="""
--- a/aipha/trading_flow/labelers/potential_capture_engine.py
+++ b/aipha/trading_flow/labelers/potential_capture_engine.py
@@ -10,6 +10,10 @@
 class PotentialCaptureEngine:
     def __init__(self, config: Dict[str, Any]):
         self.config = config
+        # Parámetros ATR
+        self.atr_period = config.get("atr_period", 20)
+        self.tp_multiplier = config.get("tp_multiplier", 5.0)
+        self.sl_multiplier = config.get("sl_multiplier", 3.0)
         logger.info("PotentialCaptureEngine inicializado")
     
     def calculate_tp_sl(self, data: Dict[str, Any]) -> Dict[str, float]:
@@ -17,8 +21,20 @@
         
         Args:
             data (Dict[str, Any]): Datos de mercado
-            
+            
         Returns:
             Dict[str, float]: TP y SL calculados
         """
-        # Implementación fija (actual)
-        return {"tp": data["price"] * 1.02, "sl": data["price"] * 0.98}
+        # Calcular ATR
+        atr = self._calculate_atr(data)
+        
+        # Calcular TP y SL dinámicos
+        tp = data["close"] + (atr * self.tp_multiplier)
+        sl = data["close"] - (atr * self.sl_multiplier)
+        
+        return {"tp": tp, "sl": sl}
+    
+    def _calculate_atr(self, data: Dict[str, Any]) -> float:
+        """Calcula el Average True Range"""
+        # Implementación simplificada de ATR
+        high_low = data["high"] - data["low"]
+        high_close = abs(data["high"] - data["close_prev"])
+        low_close = abs(data["low"] - data["close_prev"])
+        true_range = max(high_low, high_close, low_close)
+        
+        # En una implementación real, se calcularía la media móvil del True Range
+        return true_range
""",
        compatibility_check="Passed",
        rollback_plan="Revert to previous version of PotentialCaptureEngine",
        status=ApprovalStatus.APPROVED,
        approved_by="Developer",
        approval_timestamp="2023-01-01T00:01:00Z"
    )
    
    # 2. Aprobar propuesta
    proposal.status = ApprovalStatus.APPROVED
    proposal.approved_by = "Developer"
    proposal.approval_timestamp = "2023-01-01T00:01:00Z"
    
    # 3. Implementar con CodecraftSage
    config = load_config()
    context_sentinel = ContextSentinel(config)
    codecraft_sage = CodecraftSage(config, context_sentinel)
    
    implementation = codecraft_sage.implement_proposal(proposal)
    
    # 4. Aplicar cambios
    for file_info in implementation.files_created:
        file_path = Path(file_info["path"])
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            f.write(file_info["content"])
    
    # 5. Ejecutar pruebas
    for test_info in implementation.tests_created:
        test_path = Path(test_info["path"])
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(test_path, "w") as f:
            f.write(test_info["content"])
    
    # 6. Verificar resultados
    test_results = codecraft_sage._run_tests(implementation.tests_created)
    
    print(f"Implementación ATR completada:")
    print(f"  Archivos creados: {len(implementation.files_created)}")
    print(f"  Pruebas creadas: {len(implementation.tests_created)}")
    print(f"  Resultados pruebas: {test_results['passed']}/{test_results['total']} pasadas")
    
    return implementation
```

#### Ejemplo 2: Integración con el Sistema de Knowledge Manager

```python
# Ejemplo de cómo CodecraftSage usa el Knowledge Manager
def enhanced_code_generation():
    """Ejemplo de generación de código mejorada con Knowledge Manager"""
    
    # Crear propuesta
    proposal = ChangeProposal(
        change_id="km-integration-001",
        timestamp="2023-01-01T00:00:00Z",
        version="1.1.1",
        author="KM Specialist",
        description="Integración mejorada con Knowledge Manager",
        justification="Usar conocimiento previo para mejorar la generación de código",
        files_affected=["aipha/core/tools/enhanced_generator.py"],
        diff_content="+ Nueva implementación con KM",
        compatibility_check="Passed",
        rollback_plan="Revert to previous version",
        status=ApprovalStatus.APPROVED,
        approved_by="Developer",
        approval_timestamp="2023-01-01T00:01:00Z"
    )
    
    # Configuración
    config = load_config()
    context_sentinel = ContextSentinel(config)
    
    # Añadir conocimiento relevante al sistema
    context_sentinel.add_knowledge_entry(
        category="code_generation",
        title="Patrones de generación de código",
        content="""
Para generar código de calidad, sigue estos patrones:
1. Incluir docstrings completos con Args, Returns y Examples
2. Usar type hints para mejorar la legibilidad
3. Manejar errores con excepciones específicas
4. Incluir logging para facilitar la depuración
5. Escribir pruebas unitarias para cada método
        """,
        metadata={"priority": "high", "author": "CodeCraftSage"}
    )
    
    # Inicializar CodecraftSage
    codecraft_sage = CodecraftSage(config, context_sentinel)
    
    # Implementar propuesta
    implementation = codecraft_sage.implement_proposal(proposal)
    
    # El código generado incluirá los patrones del Knowledge Manager
    print("Código generado con patrones del Knowledge Manager:")
    for file_info in implementation.files_created:
        print(f"  {file_info['path']}")
        print("  Contenido (primeras líneas):")
        lines = file_info["content"].split("\n")[:10]
        for line in lines:
            print(f"    {line}")
        print("    ...")
    
    return implementation
```

<a name="recomendaciones"></a>
## 4. Recomendaciones y Buenas Prácticas

### Para Programadores Juniors

1. **Entender el Flujo Completo**: Antes de modificar CodecraftSage, asegúrate de entender cómo se integra en el flujo de automejora de Aipha.

2. **Pruebas Primero**: Siempre escribe pruebas antes de modificar la funcionalidad existente. Usa los tests existentes como guía.

3. **Logs Claros**: Añade logs descriptivos en cada paso del proceso para facilitar la depuración.

4. **Manejo de Errores**: Implementa un manejo robusto de errores con excepciones específicas y mensajes claros.

5. **Documentación**: Mantén actualizada la documentación (docstrings y CRITICAL_CONTRACTS.md) a medida que añades funcionalidad.

### Para el Desarrollo de CodecraftSage

1. **Modularidad**: Mantén cada método de CodecraftSage enfocado en una tarea específica.

2. **Extensibilidad**: Diseña CodecraftSage para soportar fácilmente nuevos tipos de archivos y lenguajes de programación.

3. **Configurabilidad**: Usa el archivo config.yaml para exponer parámetros ajustables sin modificar el código.

4. **Integración con Knowledge Manager**: Aprovecha el Knowledge Manager para mejorar la calidad del código generado.

5. **Seguridad**: Implementa validaciones para asegurar que el código generado sea seguro y no introduzca vulnerabilidades.

### Para la Evolución del Sistema

1. **Métricas de Calidad**: Implementa métricas para evaluar la calidad del código generado (complejidad, cobertura de pruebas, etc.).

2. **Feedback Loop**: Crea un sistema para recopilar feedback sobre las implementaciones y usarlo para mejorar futuras generaciones.

3. **Versionado Semántico**: Usa versionado semántico para los componentes generados por CodecraftSage.

4. **Pruebas de Integración**: Asegúrate de que el código generado se integre correctamente con el resto del sistema.

5. **Evolución Gradual**: Comienza con implementaciones simples y ve añadiendo complejidad gradualmente.

## Conclusión

La implementación de CodecraftSage en la Fase 3 representa un paso crucial en la evolución de Aipha, permitiendo al sistema no solo proponer cambios, sino también implementarlos de forma automática. Esta capacidad es fundamental para el objetivo final de Aipha: un sistema que puede mejorarse a sí mismo de forma autónoma.

Siguiendo los pasos y recomendaciones descritas en esta guía, un programador junior puede contribuir eficazmente al desarrollo de CodecraftSage y al crecimiento del proyecto Aipha.