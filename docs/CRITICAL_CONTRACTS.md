# CRITICAL CONTRACTS for Aipha_1.1 - High-Level Protocols and Principles

## 1. Atomic Update Protocol (5 Immutable Steps)
# ... (contenido original del primer doc)

## 2. ChangeProposal Format and Validation
# ... (contenido original)

## 3. Safety-First Principles
# ... (contenido original)

## 4. Evaluation Criteria for Proposals
# ... (contenido original)

## 5. RAG Protocol for Knowledge Queries (Retrieval-Augmented Generation)
- **Retrieval**: Use semantic search with SentenceTransformer embeddings (cosine similarity) on ChromaDB vector database for >80% precision vs keyword matching.
- **Augmentation**: Combine top N semantically relevant results (default 5, configurable) as structured context with source attribution in prompt.
- **Generation**: Query LLM (OpenAI GPT-3.5-turbo) with enhanced RAG prompt: system prompt defining assistant role + user prompt containing retrieved context + original query.
- **Filtering**: Apply metadata type/category filters in vector search to restrict results to specific domains (e.g., "architecture", "trading").
- **Safety**: Limit N results to prevent token overflow; validate API_KEY from environment variables only; implement comprehensive error handling with fallbacks.

## 6. Embeddings Configuration and Management
- **Model**: all-MiniLM-L6-v2 (384 dimensions) via SentenceTransformer for balanced performance/accuracy.
- **Usage**: Automatic embedding generation for document addition; semantic similarity search using cosine distance; metadata stored separately for filtering.
- **Persistence**: ChromaDB handles embedding storage and indexing; collection recreated on initialization if missing.
- **Integrity**: Comprehensive verification in `verify_knowledge_base_integrity()` including collection accessibility, search functionality, and document count validation.

## 7. LLM Safety and Operational Guidelines
- **Provider/Model**: OpenAI GPT-3.5-turbo (configurable via AIPHAConfig); supports openai>=1.0 API with chat.completions.create.
- **API Security**: API_KEY loaded exclusively from environment variables (never hardcoded); validated on system initialization.
- **Prompt Engineering**: Structured RAG prompts with system context ("You are an AI assistant with access to Aipha's knowledge base") and user context containing retrieved knowledge.
- **Rate Limiting**: Not implemented; monitor API usage and implement if needed for production deployment.
- **Privacy**: No sensitive financial/trading data in LLM queries; log queries/responses in anonymized format for debugging.
- **Error Handling**: Comprehensive try/catch blocks with graceful degradation; API failures return informative error messages instead of crashes.
- **Content Safety**: LLM responses validated for relevance; fallback to "No relevant context found" if retrieval fails.

## 8. Proposal Evaluation Protocol (Rule-Based + RAG)
- **Evaluation Criteria Categories**: Feasibility (0.3 weight), Impact (0.4 weight), Risk (0.3 weight, inverted).
- **Knowledge Retrieval**: Query evaluation_criteria from vector DB for contextual assessment.
- **LLM Integration**: Use RAG prompt with retrieved criteria for contextual evaluation; fallback to rule-based if LLM unavailable.
- **Score Calculation**: Weighted average with configurable thresholds (default 0.7 for approval).
- **Response Parsing**: Regex-based extraction of numerical scores from LLM responses.
- **Fallback Strategy**: Rule-based evaluation when LLM fails (API key issues, network problems).
- **Logging**: Comprehensive evaluation logging with scores, criteria used, and approval status.
- **Error Handling**: Conservative scoring (0.5) on evaluation failures with detailed error reporting.

## ChangeProposer Agent Role

### Misión Principal
El agente **ChangeProposer** es el iniciador del ciclo de automejora en Aipha_1.1. Su propósito fundamental es analizar directivas de alto nivel y transformarlas en propuestas de cambio concretas, verificables y seguras que puedan ser evaluadas e implementadas por otros agentes del sistema. Actúa como el puente entre las necesidades estratégicas del sistema y las modificaciones técnicas específicas, asegurando que todas las propuestas sigan estrictamente el Protocolo de Actualización Atómica (ATR) y los principios de seguridad del sistema.

### Entradas (Inputs)
- **Directivas de Alto Nivel**: Instrucciones estratégicas expresadas en lenguaje natural (ej: "Implementar barreras dinámicas con ATR", "Optimizar el rendimiento del motor de trading").
- **Estado del Sistema**: Información actual del sistema obtenida de `CriticalMemoryRules`, incluyendo versión actual, historial de cambios y estado de integridad.
- **Conocimiento Contextual**: Datos relevantes del `ContextSentinel`, incluyendo patrones de código, criterios de evaluación previos y conocimiento base sobre arquitectura y principios.
- **Configuración del Sistema**: Parámetros de configuración que afectan la generación de propuestas, como límites de complejidad y tipos de cambios permitidos.

### Proceso (Process)
1. **Análisis de la Directiva**: Descomponer la directiva en componentes técnicos identificables, validando que sea compatible con la arquitectura actual.
2. **Consulta de Conocimiento**: Buscar en la base de conocimiento patrones similares, restricciones de seguridad y mejores prácticas relevantes.
3. **Evaluación de Viabilidad**: Verificar que la propuesta no viole principios de seguridad ni exceda límites de complejidad definidos.
4. **Generación de Justificación**: Construir una argumentación técnica detallada basada en evidencia del conocimiento base y análisis de impacto.
5. **Formulación del Diff**: Crear un diff unificado preciso que muestre exactamente qué archivos se modifican y cómo, siguiendo el formato estándar de patches.
6. **Validación de Seguridad**: Asegurar que la propuesta incluye salvaguardas para rollback y no compromete la integridad del sistema.

### Salidas (Outputs)
Produce una instancia completa de `ChangeProposal` que incluye:
- **Descripción**: Resumen claro y conciso del cambio propuesto.
- **Justificación**: Argumentación técnica detallada con referencias al conocimiento base.
- **Archivos Afectados**: Lista exacta de archivos que serán modificados.
- **Contenido del Diff**: Patch unificado que especifica exactamente qué líneas cambiar, añadir o eliminar.
- **Metadatos**: Información adicional como prioridad, impacto estimado y autor del agente.

### Interacciones Clave
- **ContextSentinel**: Consulta conocimiento base para contextualizar propuestas y validar compatibilidad.
- **RedesignHelper**: Entrega la propuesta completada para evaluación posterior en el ciclo de automejora.
- **CriticalMemoryRules**: Utiliza para crear la propuesta formal y validar contra reglas de integridad del sistema.
- **Sistema de Logging**: Registra todas las decisiones y razonamientos para trazabilidad y auditoría.