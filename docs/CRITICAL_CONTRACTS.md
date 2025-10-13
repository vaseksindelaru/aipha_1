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