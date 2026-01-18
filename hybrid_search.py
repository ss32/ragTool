"""
Hybrid Search Module
Implements Reciprocal Rank Fusion (RRF) for combining BM25 and vector search results.
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
    weights: Optional[List[float]] = None,
    id_key: str = 'doc_id'
) -> List[Dict]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF formula: score(d) = sum(weight_i / (k + rank_i(d)))
    where k is a constant (typically 60) and rank_i(d) is the rank of document d
    in result list i (1-indexed).

    Args:
        result_lists: List of result lists, each containing dicts with id_key
        k: RRF constant (default 60, research-optimal)
        weights: Optional weights for each result list (default: equal weights)
        id_key: Key to use for document identification

    Returns:
        Combined results sorted by RRF score, with added 'rrf_score' field
    """
    if not result_lists:
        return []

    # Default to equal weights
    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError("Number of weights must match number of result lists")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Calculate RRF scores
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_data: Dict[str, Dict] = {}  # Store document data by ID
    doc_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)  # Track ranks from each source

    for list_idx, (results, weight) in enumerate(zip(result_lists, weights)):
        source_name = f"source_{list_idx}"
        for rank, doc in enumerate(results, start=1):  # 1-indexed ranks
            doc_id = doc.get(id_key)
            if doc_id is None:
                continue

            # Calculate RRF contribution
            rrf_scores[doc_id] += weight / (k + rank)

            # Store document data (first occurrence wins)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc.copy()

            # Track which source found this document and at what rank
            doc_ranks[doc_id][source_name] = rank

    # Build final results with RRF scores
    fused_results = []
    for doc_id, rrf_score in rrf_scores.items():
        result = doc_data[doc_id].copy()
        result['rrf_score'] = rrf_score
        result['source_ranks'] = doc_ranks[doc_id]
        fused_results.append(result)

    # Sort by RRF score (descending)
    fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)

    return fused_results


class HybridSearcher:
    """Combines BM25 and vector search results using RRF."""

    def __init__(
        self,
        rrf_k: int = 60,
        default_vector_weight: float = 0.5
    ):
        """Initialize hybrid searcher.

        Args:
            rrf_k: RRF constant (default 60)
            default_vector_weight: Default weight for vector search (0.0-1.0)
                                   BM25 weight = 1.0 - vector_weight
        """
        self.rrf_k = rrf_k
        self.default_vector_weight = default_vector_weight

    def fuse_results(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        n_results: int = 10,
        vector_weight: Optional[float] = None
    ) -> List[Dict]:
        """Fuse BM25 and vector search results using RRF.

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            n_results: Number of results to return
            vector_weight: Weight for vector results (0.0-1.0, default: 0.5)
                          BM25 weight = 1.0 - vector_weight

        Returns:
            Fused results with RRF scores and provenance info
        """
        if vector_weight is None:
            vector_weight = self.default_vector_weight

        bm25_weight = 1.0 - vector_weight

        # Prepare result lists with consistent ID key
        # BM25 results use 'doc_id', vector results use position-based matching
        bm25_prepared = []
        for i, r in enumerate(bm25_results):
            doc = r.copy()
            if 'doc_id' not in doc:
                doc['doc_id'] = f"bm25_{i}"
            doc['bm25_rank'] = r.get('bm25_rank', i)
            bm25_prepared.append(doc)

        vector_prepared = []
        for i, r in enumerate(vector_results):
            doc = r.copy()
            # Use content hash as ID for matching if no doc_id present
            if 'doc_id' not in doc:
                content = doc.get('content', '')
                doc['doc_id'] = f"vec_{hash(content)}"
            doc['vector_rank'] = i
            vector_prepared.append(doc)

        # Perform RRF fusion
        fused = reciprocal_rank_fusion(
            [bm25_prepared, vector_prepared],
            k=self.rrf_k,
            weights=[bm25_weight, vector_weight],
            id_key='doc_id'
        )

        # Enrich results with provenance info
        for doc in fused:
            source_ranks = doc.get('source_ranks', {})
            doc['bm25_rank'] = source_ranks.get('source_0')  # None if not in BM25
            doc['vector_rank'] = source_ranks.get('source_1')  # None if not in vector

        return fused[:n_results]

    def fuse_by_content(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        n_results: int = 10,
        vector_weight: Optional[float] = None
    ) -> List[Dict]:
        """Fuse results by matching on content (for when doc IDs don't match).

        This is useful when BM25 and vector stores have different ID schemes.

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            n_results: Number of results to return
            vector_weight: Weight for vector results (0.0-1.0)

        Returns:
            Fused results with RRF scores
        """
        if vector_weight is None:
            vector_weight = self.default_vector_weight

        bm25_weight = 1.0 - vector_weight

        # Create content-based IDs for matching
        def content_id(content: str) -> str:
            """Create a stable ID from content."""
            # Use first 200 chars to create ID (handles chunked duplicates)
            return str(hash(content[:200] if content else ""))

        bm25_prepared = []
        for i, r in enumerate(bm25_results):
            doc = r.copy()
            content = doc.get('content', '')
            doc['doc_id'] = content_id(content)
            doc['bm25_rank'] = r.get('bm25_rank', i)
            bm25_prepared.append(doc)

        vector_prepared = []
        for i, r in enumerate(vector_results):
            doc = r.copy()
            content = doc.get('content', '')
            doc['doc_id'] = content_id(content)
            doc['vector_rank'] = i
            vector_prepared.append(doc)

        # Perform RRF fusion
        fused = reciprocal_rank_fusion(
            [bm25_prepared, vector_prepared],
            k=self.rrf_k,
            weights=[bm25_weight, vector_weight],
            id_key='doc_id'
        )

        # Enrich results with provenance info
        for doc in fused:
            source_ranks = doc.get('source_ranks', {})
            doc['bm25_rank'] = source_ranks.get('source_0')
            doc['vector_rank'] = source_ranks.get('source_1')

        return fused[:n_results]


if __name__ == "__main__":
    # Test RRF fusion
    bm25_results = [
        {'doc_id': 'A', 'content': 'Document A content', 'bm25_score': 10.5},
        {'doc_id': 'B', 'content': 'Document B content', 'bm25_score': 8.2},
        {'doc_id': 'C', 'content': 'Document C content', 'bm25_score': 5.1},
    ]

    vector_results = [
        {'doc_id': 'B', 'content': 'Document B content', 'distance': 0.15},
        {'doc_id': 'D', 'content': 'Document D content', 'distance': 0.22},
        {'doc_id': 'A', 'content': 'Document A content', 'distance': 0.35},
    ]

    searcher = HybridSearcher()
    fused = searcher.fuse_results(bm25_results, vector_results, n_results=5)

    print("Fused Results:")
    for i, doc in enumerate(fused):
        print(f"  [{i + 1}] ID: {doc['doc_id']}, RRF: {doc['rrf_score']:.4f}")
        print(f"       BM25 rank: {doc.get('bm25_rank')}, Vector rank: {doc.get('vector_rank')}")
