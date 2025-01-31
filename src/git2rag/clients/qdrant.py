"""Qdrant client operations for vector storage and retrieval."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct, Filter


@dataclass
class SearchResult:
    """Search result from Qdrant."""
    score: float
    content: str
    file: str
    type: str
    context: Optional[str]
    lines: Tuple[Optional[int], Optional[int]]


class QdrantManager:
    """Manager for Qdrant operations."""

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        batch_size: int = 100
    ):
        """Initialize Qdrant manager.

        Args:
            url: URL of the Qdrant server
            api_key: API key for authentication
            timeout: Request timeout in seconds
            batch_size: Default batch size for operations
        """
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout
        )
        self.batch_size = batch_size

    def create_collection(
        self,
        name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        on_disk_payload: bool = True
    ) -> bool:
        """Create a new collection if it doesn't exist.

        Args:
            name: Collection name
            vector_size: Size of vectors
            distance: Distance function for similarity
            on_disk_payload: Store payload on disk instead of memory

        Returns:
            True if collection was created, False if it already existed
        """
        collections = self.client.get_collections().collections
        if any(c.name == name for c in collections):
            return False

        self.client.create_collection(
            collection_name=name,
            vectors_config={
                "raw": VectorParams(size=vector_size, distance=distance),
                "processed": VectorParams(size=vector_size, distance=distance)
            },
            on_disk_payload=on_disk_payload
        )
        return True

    def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if collection was deleted, False if it didn't exist
        """
        try:
            self.client.delete_collection(name)
            return True
        except Exception:
            return False

    def insert_points(
        self,
        collection_name: str,
        points: List[dict],  # Each point must have id, payload, and vectors
        batch_size: Optional[int] = None
    ) -> None:
        """Insert points into collection.

        Args:
            collection_name: Collection name
            points: Points to insert, each with format:
                   {
                       'id': point_id,
                       'payload': payload_dict,
                       'vectors': {
                           'raw': raw_vector,      # optional
                           'processed': proc_vector # optional
                       }
                   }
            batch_size: Optional custom batch size
        """
        batch_size = batch_size or self.batch_size
        total_points = len(points)

        for i in range(0, total_points, batch_size):
            batch = []
            for point in points[i:i + batch_size]:
                # Create point with named vectors
                point_data = {
                    "id": point['id'],
                    "payload": point['payload']
                }
                # Add vectors if they exist
                vectors_dict = {}
                if 'raw' in point['vectors']:
                    vectors_dict["raw"] = point['vectors']['raw']
                if 'processed' in point['vectors']:
                    vectors_dict["processed"] = point['vectors']['processed']
                point_data["vector"] = vectors_dict

                batch.append(point_data)

            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )

    def delete_points(
        self,
        collection_name: str,
        point_ids: List[Union[str, int]]
    ) -> None:
        """Delete points from collection.

        Args:
            collection_name: Collection name
            point_ids: IDs of points to delete
        """
        self.client.delete(
            collection_name=collection_name,
            points_selector=point_ids
        )

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        vector_name: str = "raw",  # "raw" or "processed"
        limit: int = 5,
        offset: int = 0,
        filter_conditions: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar vectors.

        Args:
            collection_name: Collection name
            query_vector: Query vector
            limit: Maximum number of results
            offset: Number of results to skip
            filter_conditions: Optional filtering conditions
            with_payload: Include payload in results
            score_threshold: Minimum similarity score threshold

        Returns:
            List of search results
        """
        # Build filter if conditions provided
        search_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, dict) and "path" in value:
                    # Handle regex path matching
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchText(text=value["path"])
                        )
                    )
                else:
                    # Handle exact value matching
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
            search_filter = models.Filter(must=must_conditions)

        # Perform search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector if vector_name == "raw" else None,
            query_vector_processed=query_vector if vector_name == "processed" else None,
            limit=limit,
            offset=offset,
            query_filter=search_filter,
            with_payload=with_payload,
            score_threshold=score_threshold
        )

        # Convert to SearchResult objects
        return [
            SearchResult(
                score=result.score,
                content=result.payload["content"],
                file=result.payload["source_file"],
                type=result.payload["chunk_type"],
                context=result.payload.get("context"),
                lines=(
                    result.payload.get("start_line"),
                    result.payload.get("end_line")
                )
            )
            for result in results
        ]

    def count_points(
        self,
        collection_name: str,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count points in collection.

        Args:
            collection_name: Collection name
            filter_conditions: Optional filtering conditions

        Returns:
            Number of points matching criteria
        """
        # Build filter if conditions provided
        count_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            count_filter = models.Filter(must=must_conditions)

        return self.client.count(
            collection_name=collection_name,
            count_filter=count_filter
        ).count

    def scroll_points(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[Union[str, int]] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> tuple[List[PointStruct], Optional[str]]:
        """Scroll through points in collection.

        Args:
            collection_name: Collection name
            limit: Maximum number of points per batch
            offset: Offset from previous scroll
            filter_conditions: Optional filtering conditions
            with_payload: Include payload in results
            with_vectors: Include vectors in results

        Returns:
            Tuple of (points, next_offset)
        """
        # Build filter if conditions provided
        scroll_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            scroll_filter = models.Filter(must=must_conditions)

        return self.client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            scroll_filter=scroll_filter,
            with_payload=with_payload,
            with_vectors=with_vectors
        )

    def update_vectors(
        self,
        collection_name: str,
        points: List[tuple[Union[str, int], List[float]]]
    ) -> None:
        """Update vectors for existing points.

        Args:
            collection_name: Collection name
            points: List of (point_id, new_vector) tuples
        """
        self.client.update_vectors(
            collection_name=collection_name,
            points=points
        )

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections and their information.

        Returns:
            List of collection information dictionaries
        """
        collections = self.client.get_collections().collections
        result = []

        for collection in collections:
            try:
                info = self.client.get_collection(collection.name)
                result.append({
                    "name": info.name,
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance,
                    "status": info.status,
                    "vectors_count": info.vectors_count
                })
            except Exception:
                # Skip collections that can't be accessed
                continue

        return result

    def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get collection information.

        Args:
            name: Collection name

        Returns:
            Collection info or None if not found
        """
        try:
            info = self.client.get_collection(name)
            return {
                "name": info.name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "status": info.status,
                "vectors_count": info.vectors_count
            }
        except Exception:
            return None
