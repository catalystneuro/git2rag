from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

import instructor
from pydantic import BaseModel
from litellm import embedding, completion
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

# Prompts for LLM interactions
FILTER_RESULTS_PROMPT = """Given a search query, its context, and a list of search results, determine which results are truly relevant to answering the query within its context.
Analyze each result carefully and return the indices of only the most relevant items.

Query: {query}
Context: {context}

Results:
{formatted_results}

Return the list of indices (0-based) for results that are most relevant to the query and context."""

QUERY_EXPANSION_PROMPT = """Given a search query and its context in the user input, generate 2 alternative queries that could help find relevant information in a semantic search to a vector database.
These alternatives should be very short and informative, focusing on the most relevant terms of the original query.
---
Example:
Original Query: What are the most common symptoms of COVID-19 in adults?
Context: The user is writing an essay on public health measures to control the spread of epidemics.
Your response:
Most common symptoms of COVID-19 in adults
Symptoms of COVID-19 in adults
---
User Input:
Original Query: {query}
Context: {context}

Return only the 2 alternative queries, one per line."""

SUMMARY_PROMPT = """Synthesize a concise and informative response based on the following search results, original query, and context. Focus on relevant information and discard any irrelevant details.
Original Query: {query}
Context: {context}
Search Results:
{results}

Provide a clear and focused response that addresses the query within its context."""


class FilterResult(BaseModel):
    indices: List[int]


def generate_embeddings(
    texts: List[str],
    model: str = "text-embedding-ada-002",
) -> List[List[float]]:
    """Generate embeddings for a list of texts using litellm.

    Args:
        texts: List of texts to generate embeddings for
        model: Model to use for embeddings

    Returns:
        List of embedding vectors
    """
    response = embedding(model=model, input=texts)
    return [data["embedding"] for data in response.data]


def search_vectors(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    vector_name: str,
    limit: int = 5,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Search for similar vectors using Qdrant.

    Args:
        client: QdrantClient instance
        collection_name: Collection name
        query_vector: Query vector
        vector_name: Name of the vector to search
        limit: Maximum number of results
        score_threshold: Minimum similarity score threshold

    Returns:
        List of search results
    """
    results = client.search(
        collection_name=collection_name,
        query_vector=(vector_name, query_vector),
        limit=limit,
        score_threshold=score_threshold,
    )

    return [
        {
            "id": str(r.id),
            "score": r.score,
            "content": r.payload.get("content", ""),
            "file": r.payload.get("source_file", ""),
            "type": r.payload.get("chunk_type", ""),
            "context": r.payload.get("context"),
        }
        for r in results
    ]


@dataclass
class SearchResult:
    """Represents a single search result with its score."""

    id: str
    score: float
    content: str
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create SearchResult from dictionary response."""
        return cls(
            id=data["id"],
            score=data["score"],
            content=data["content"],
            metadata={"file": data["file"], "type": data["type"], "context": data["context"]},
        )


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]], k: int = 60
) -> List[SearchResult]:
    """Combine multiple result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of result lists to combine
        k: Constant to prevent items with very low ranks from having too much impact

    Returns:
        Combined and reranked list of results
    """
    # Track scores for each document
    scores: Dict[str, float] = defaultdict(float)

    # Process each result list
    for results in result_lists:
        for rank, result in enumerate(results):
            scores[result.id] += 1.0 / (k + rank)

    # Create combined results list
    combined_results = []
    seen_ids = set()

    # Sort by RRF score and create new result objects
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            # Find original result to get content and metadata
            for results in result_lists:
                for r in results:
                    if r.id == doc_id:
                        combined_results.append(
                            SearchResult(
                                id=doc_id, score=score, content=r.content, metadata=r.metadata
                            )
                        )
                        break
                if doc_id in seen_ids:
                    break

    return combined_results


def expand_query(query: str, context: str, model: str = "gpt-3.5-turbo") -> List[str]:
    """Generate alternative search queries using LLM.

    Args:
        query: Original search query
        context: Context for the query
        model: LLM model to use

    Returns:
        List of 2 alternative queries
    """
    prompt = QUERY_EXPANSION_PROMPT.format(query=query, context=context)

    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    # Split response into lines and clean up
    expanded_queries = [
        line.strip()
        for line in response.choices[0].message.content.strip().split("\n")
        if line.strip()
    ]

    # Ensure we have exactly 2 queries
    if len(expanded_queries) < 2:
        expanded_queries.extend([query] * (2 - len(expanded_queries)))
    elif len(expanded_queries) > 2:
        expanded_queries = expanded_queries[:2]

    return expanded_queries


def llm_filter_results(
    query: str,
    context: str,
    results: List[SearchResult],
    model: str = "gpt-3.5-turbo"
) -> List[SearchResult]:
    """Filter search results using LLM to keep only relevant items.

    Args:
        query: Original search query
        context: Context for the query
        results: List of search results to filter
        model: LLM model to use

    Returns:
        Filtered list of search results containing only relevant items
    """
    # Format results with indices for the prompt
    formatted_results = "\n".join(
        f"[{i}] {r.content}"
        for i, r in enumerate(results)
    )

    messages = [
        {
            "role": "user",
            "content": FILTER_RESULTS_PROMPT.format(
                query=query,
                context=context,
                formatted_results=formatted_results
            ),
        }
    ]

    client = instructor.from_litellm(completion)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=FilterResult,
    )

    # Return only the results at the specified indices
    return [results[i] for i in resp.indices]


def generate_summary(
    query: str, context: str, results: List[SearchResult], model: str = "gpt-3.5-turbo"
) -> str:
    """Generate a summary of search results using LLM.

    Args:
        query: Original search query
        context: Context for the query
        results: List of search results to summarize
        model: LLM model to use

    Returns:
        Generated summary
    """
    # Format results for the prompt
    formatted_results = "\n".join(f"- {r.content}" for r in results)

    prompt = SUMMARY_PROMPT.format(query=query, context=context, results=formatted_results)

    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


def search(
    query: str,
    context: str,
    qdrant_url: str,
    collection_name: str,
    keywords: Optional[List[str]] = None,
    qdrant_api_key: Optional[str] = None,
    timeout: float = 60.0,
    return_digest_summary: bool = True,
    return_references: bool = True,
    limit: int = 10,
    model: str = "gpt-3.5-turbo",
) -> Dict[str, Any]:
    """Perform advanced hybrid search with query expansion and optional summarization.

    Args:
        query: Search query text
        context: Context in which the query is relevant
        keywords: List of keywords for sparse search
        qdrant_manager: QdrantManager instance
        collection_name: Name of collection to search
        return_digest_summary: Whether to generate a summary of results
        return_references: Whether to include search results in response
        limit: Maximum number of results to return
        model: LLM model to use for query expansion and summarization

    Returns:
        Dictionary containing:
        - expanded_queries: List of generated alternative queries
        - search_results: List of search results (if return_references=True)
        - summary: Generated summary (if return_digest_summary=True)
    """
    # Step 1: Generate alternative queries
    print(f"Expanding query: {query}")
    expanded_queries = expand_query(query, context, model)
    print(f"Expanded queries: {expanded_queries}")

    # Initialize Qdrant client
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=timeout)

    # Get vector names from collection
    collection_info = client.get_collection(collection_name=collection_name)
    vector_names = list(collection_info.config.params.vectors.keys())
    print(f"Available vector names: {vector_names}")

    # Dictionary to track unique results by ID
    unique_results: Dict[str, SearchResult] = {}

    # Generate embedding for original query
    query_embedding = generate_embeddings([query])[0]

    # Search with original query across all vector names
    for vector_name in vector_names:
        results = search_vectors(
            client=client,
            collection_name=collection_name,
            query_vector=query_embedding,
            vector_name=vector_name,
            limit=limit,
        )
        for r in results:
            result = SearchResult.from_dict(r)
            if result.id not in unique_results or result.score > unique_results[result.id].score:
                unique_results[result.id] = result
        print(f"Semantic search results for original query with {vector_name}: {len(results)}")

    # Expanded queries search
    for exp_query in expanded_queries:
        # Generate embedding for expanded query
        exp_query_embedding = generate_embeddings([exp_query])[0]
        for vector_name in vector_names:
            results = search_vectors(
                client=client,
                collection_name=collection_name,
                query_vector=exp_query_embedding,
                vector_name=vector_name,
                limit=limit,
            )
            for r in results:
                result = SearchResult.from_dict(r)
                if (
                    result.id not in unique_results
                    or result.score > unique_results[result.id].score
                ):
                    unique_results[result.id] = result
            print(f"Semantic search results for expanded query with {vector_name}: {len(results)}")

    # Convert unique results to sorted list
    semantic_results = [
        sorted(unique_results.values(), key=lambda x: x.score, reverse=True)[:limit]
    ]

    # Keyword search if keywords provided
    if keywords:
        # Generate embeddings for keywords
        keyword_embeddings = generate_embeddings(keywords)

        # Dictionary to track unique keyword results
        keyword_unique_results: Dict[str, SearchResult] = {}

        for keyword, keyword_embedding in zip(keywords, keyword_embeddings):
            for vector_name in vector_names:
                results = search_vectors(
                    client=client,
                    collection_name=collection_name,
                    query_vector=keyword_embedding,
                    vector_name=vector_name,
                    limit=limit,
                )
                for r in results:
                    result = SearchResult.from_dict(r)
                    if (
                        result.id not in keyword_unique_results
                        or result.score > keyword_unique_results[result.id].score
                    ):
                        keyword_unique_results[result.id] = result
                print(f"Keyword search results for '{keyword}' with {vector_name}: {len(results)}")

        # Convert unique keyword results to sorted list
        keyword_results = sorted(
            keyword_unique_results.values(), key=lambda x: x.score, reverse=True
        )[:limit]

        # Only append keyword results if we found any
        if keyword_results:
            semantic_results.append(keyword_results)

    # Combine results using RRF
    if keywords:
        print(f"Combining search results using Reciprocal Rank Fusion")
        combined_results = reciprocal_rank_fusion(semantic_results)[:limit]
    else:
        combined_results = semantic_results[0]

    # Prepare response
    response = {"expanded_queries": expanded_queries}

    if return_references:
        response["search_results"] = [
            {"id": r.id, "score": r.score, "content": r.content, **r.metadata}
            for r in combined_results
        ]

    if return_digest_summary and combined_results:
        # First filter the results
        print(f"LLM is filtering the search results")
        filtered_results = llm_filter_results(
            query=query,
            context=context,
            results=combined_results,
            model=model,
        )

        # Then generate summary only if we have filtered results
        if filtered_results:
            response["search_results"] = [
                {"id": r.id, "score": r.score, "content": r.content, **r.metadata}
                for r in filtered_results
            ]
            print(f"Generating summary based on filtered results")
            response["summary"] = generate_summary(
                query=query,
                context=context,
                results=filtered_results,
                model=model,
            )

    return response
