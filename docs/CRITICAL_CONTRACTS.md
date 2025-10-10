# CRITICAL CONTRACTS for Aipha_1.1 - High-Level Protocols and Principles

## 1. Atomic Update Protocol (5 Immutable Steps)
# ... (contenido original del primer doc)

## 2. ChangeProposal Format and Validation
# ... (contenido original)

## 3. Safety-First Principles
# ... (contenido original)

## 4. Evaluation Criteria for Proposals
# ... (contenido original)

## 5. RAG Protocol for Knowledge Queries
- Retrieval: Use semantic search with embeddings (SentenceTransformer) on ChromaDB.
- Augmentation: Combine top N results (default 5) as context in prompt.
- Generation: Query LLM (OpenAI) with context + user query.
- Filter: Apply type/category filter in search.
- Safety: Limit N results to avoid token overflow; validate API_KEY env var.

## 6. Embeddings Configuration
- Model: all-MiniLM-L6-v2 (384 dim).
- Usage: Embed content for add/search; metadata stored separately.
- Integrity: Verify collection count > 0 in verify_knowledge_base_integrity.

## 7. LLM Safety Guidelines
- Provider/Model: OpenAI gpt-3.5-turbo (configurable).
- API Key: From env var only (never hardcode).
- Prompt Engineering: System prompt as "useful assistant"; user prompt with context.
- Rate Limiting: Not implemented yet; monitor API calls.
- Privacy: No sensitive data in queries; log anonymized.
- Error Handling: Catch API errors, fallback to empty response.