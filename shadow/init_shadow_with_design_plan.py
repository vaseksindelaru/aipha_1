# shadow/init_shadow_with_design_plan.py
"""
Script para inicializar AiphaShadow con el Plan de Diseño Unificado
"""
import logging
from pathlib import Path
from ingest_design_plan import DesignPlanIngestor
from aipha_shadow import AiphaShadow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Inicializa AiphaShadow con el Plan de Diseño"""
    logger.info("=== Inicializando AiphaShadow con Plan de Diseño ===")

    # 1. Ingestar el Plan de Diseño
    logger.info("Paso 1: Ingestando Plan de Diseño...")
    plan_path = Path(__file__).parent / "design_plan_unified.md"

    if not plan_path.exists():
        logger.error(f"Archivo no encontrado: {plan_path}")
        return

    with open(plan_path, 'r', encoding='utf-8') as f:
        plan_content = f.read()

    ingestor = DesignPlanIngestor()
    sections_count = ingestor.ingest_design_plan(plan_content)
    logger.info(f"Plan de Diseño ingresado: {sections_count} secciones")

    # 2. Inicializar AiphaShadow
    logger.info("Paso 2: Inicializando AiphaShadow...")
    shadow = AiphaShadow()

    # 3. Demostrar consultas
    logger.info("Paso 3: Demostrando consultas...")

    demo_queries = [
        ("¿Qué es Aipha?", "design"),
        ("¿Cómo implemento CodecraftSage?", "design"),
        ("¿Cuáles son las fases completadas?", "comprehensive"),
        ("¿Qué recomendaciones hay para juniors?", "design"),
        ("ATR implementation", "general")
    ]

    for query, mode in demo_queries:
        print(f"\n{'='*60}")
        print(f"Consulta: {query} (modo: {mode})")
        print('='*60)

        if mode == "design":
            result = shadow.query_design_plan(query)
        elif mode == "comprehensive":
            result = shadow.query_comprehensive(query)
        else:
            result = shadow.query(query)

        print(result)

    logger.info("=== Inicialización completada ===")

if __name__ == "__main__":
    main()