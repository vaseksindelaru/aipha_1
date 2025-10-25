# shadow/ingest_design_plan.py
import os
import yaml
from pathlib import Path
from datetime import datetime
import uuid
from typing import Dict, List, Any
import logging

from sentence_transformers import SentenceTransformer
import chromadb

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DesignPlanIngestor:
    """Clase para ingestar el Plan de Diseño Unificado en ChromaDB"""

    def __init__(self, config_path: str = None):
        """Inicializa el ingestor con la configuración del sistema"""
        if config_path is None:
            # Buscar config.yaml en el directorio padre
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(parent_dir, 'config.yaml')

        self.config = self._load_config(config_path)
        self.setup_chromadb()
        self.setup_embeddings()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga la configuración desde el archivo YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_chromadb(self):
        """Configura la conexión a ChromaDB"""
        chroma_config = self.config.get('knowledge_manager', {})
        persist_dir = chroma_config.get('chroma_persist_dir', './shadow_storage/chroma_db')
        collection_name = chroma_config.get('collection_name', 'aipha_development') + "_shadow"

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB inicializado con colección: {collection_name}")

    def setup_embeddings(self):
        """Configura el modelo de embeddings"""
        chroma_config = self.config.get('knowledge_manager', {})
        model_name = chroma_config.get('embedding_model', 'all-MiniLM-L6-v2')

        self.embedder = SentenceTransformer(model_name)
        logger.info(f"Modelo de embeddings cargado: {model_name}")

    def ingest_design_plan(self, plan_content: str):
        """Ingresa el Plan de Diseño Unificado en ChromaDB"""
        logger.info("Iniciando ingesta del Plan de Diseño Unificado")

        # Dividir el contenido en secciones
        sections = self._parse_sections(plan_content)

        # Procesar cada sección
        documents = []
        metadatas = []
        ids = []

        for section in sections:
            # Crear un ID único para cada sección
            section_id = str(uuid.uuid4())

            # Preparar metadatos
            metadata = {
                "document_type": "design_plan",
                "section_title": section["title"],
                "section_type": section["type"],
                "phase": section.get("phase", "general"),
                "priority": section.get("priority", "medium"),
                "ingestion_timestamp": datetime.now().isoformat(),
                "content_length": len(section["content"])
            }

            # Añadir a las listas
            documents.append(section["content"])
            metadatas.append(metadata)
            ids.append(section_id)

            logger.info(f"Sección procesada: {section['title']} ({len(section['content'])} caracteres)")

        # Almacenar en ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Plan de Diseño ingresado exitosamente: {len(documents)} secciones")

        # Verificar la ingesta
        count = self.collection.count()
        logger.info(f"Total de documentos en ChromaDB: {count}")

        return len(documents)

    def _parse_sections(self, content: str) -> List[Dict[str, Any]]:
        """Parsea el contenido del plan en secciones lógicas"""
        sections = []

        # Dividir por encabezados Markdown
        lines = content.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            # Detectar encabezados
            if line.startswith('#'):
                # Guardar la sección anterior si existe
                if current_section:
                    current_section["content"] = '\n'.join(current_content).strip()
                    sections.append(current_section)

                # Crear nueva sección
                level = len(line) - len(line.lstrip('#'))
                title = line.strip('#').strip()

                current_section = {
                    "title": title,
                    "type": self._classify_section(title, level),
                    "content": "",
                    "level": level
                }

                # Extraer fase si está en el título
                if "Fase" in title:
                    phase_match = title.split("Fase")[1].split(":")[0].strip()
                    current_section["phase"] = phase_match.lower()

                # Determinar prioridad basada en el título
                current_section["priority"] = self._determine_priority(title)

                current_content = []
            else:
                # Añadir línea al contenido actual
                if line.strip():  # Ignorar líneas vacías
                    current_content.append(line)

        # Guardar la última sección
        if current_section:
            current_section["content"] = '\n'.join(current_content).strip()
            sections.append(current_section)

        return sections

    def _classify_section(self, title: str, level: int) -> str:
        """Clasifica el tipo de sección basado en su título y nivel"""
        title_lower = title.lower()

        if level == 1:
            return "main_section"
        elif "introducción" in title_lower:
            return "introduction"
        elif "resumen" in title_lower or "fase" in title_lower:
            return "summary"
        elif "arquitectura" in title_lower:
            return "architecture"
        elif "pasos" in title_lower or "implementación" in title_lower:
            return "implementation"
        elif "ejemplos" in title_lower:
            return "examples"
        elif "recomendaciones" in title_lower:
            return "recommendations"
        elif "conclusión" in title_lower:
            return "conclusion"
        else:
            return "general"

    def _determine_priority(self, title: str) -> str:
        """Determina la prioridad de una sección basada en su título"""
        title_lower = title.lower()

        if "introducción" in title_lower or "resumen" in title_lower:
            return "high"
        elif "fase" in title_lower or "arquitectura" in title_lower:
            return "high"
        elif "implementación" in title_lower or "pasos" in title_lower:
            return "high"
        elif "ejemplos" in title_lower:
            return "medium"
        elif "recomendaciones" in title_lower:
            return "medium"
        else:
            return "low"

    def query_design_plan(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Consulta el Plan de Diseño ingresado"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"document_type": "design_plan"}
        )

        # Formatear resultados
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            }
            formatted_results.append(result)

        return formatted_results

def main():
    """Función principal para ingestar el Plan de Diseño"""
    # Ruta al archivo del Plan de Diseño
    plan_path = Path(__file__).parent / "design_plan_unified.md"

    if not plan_path.exists():
        logger.error(f"Archivo no encontrado: {plan_path}")
        logger.info("Por favor, guarda el Plan de Diseño Unificado como shadow/design_plan_unified.md")
        return

    # Leer el contenido del plan
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan_content = f.read()

    # Crear ingestor
    ingestor = DesignPlanIngestor()

    # Ingestar el plan
    sections_count = ingestor.ingest_design_plan(plan_content)

    # Demostrar consulta
    print("\n=== Demostración de Consultas ===")

    queries = [
        "¿Qué es Aipha?",
        "Fase 3 CodecraftSage",
        "arquitectura del sistema",
        "recomendaciones para juniors",
        "implementación de propuestas"
    ]

    for query in queries:
        print(f"\nConsulta: {query}")
        results = ingestor.query_design_plan(query, n_results=2)

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['metadata']['section_title']}")
            print(f"     Tipo: {result['metadata']['section_type']}")
            print(f"     Fase: {result['metadata'].get('phase', 'N/A')}")
            print(f"     Prioridad: {result['metadata']['priority']}")
            print(f"     Relevancia: {1 - result['distance']:.2f}" if result['distance'] else "     Relevancia: N/A")
            print(f"     Extracto: {result['content'][:100]}...")

if __name__ == "__main__":
    main()