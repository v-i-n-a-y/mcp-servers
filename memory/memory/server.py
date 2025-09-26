# memory_service.py
import logging
import uuid
import asyncio
from typing import Dict, Any, Optional, List, Sequence, Tuple

import httpx
import chromadb

from mcp.server.fastmcp import FastMCP

# Keep the mcp decorator and runner from your environment
# from some_module import mcp  # <-- you already have this in your environment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configuration - change as needed
LMSTUDIO_EMBEDDING_URL = "http://localhost:1234/v1/embeddings"  # embedding endpoint (expects {"input": text})
CHROMA_DB_PATH = "./chroma_db"
DEFAULT_EMBED_TIMEOUT = 30  # seconds

# Initialize persistent Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

mcp = FastMCP("memory")
# ---------------------
# Utilities
# ---------------------

async def _fetch_embedding_async(texts: Sequence[str], timeout: int = DEFAULT_EMBED_TIMEOUT) -> List[List[float]]:
    """
    Fetch embeddings for a list of texts from LM Studio embedding API.
    Calls the endpoint once per text. Adjust if batching is supported.
    """
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                LMSTUDIO_EMBEDDING_URL,
                json={"input": t, "model": "text-embedding-embeddinggemma-300m"},
                timeout=timeout
            )
            for t in texts
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    embeddings: List[List[float]] = []
    for i, r in enumerate(responses):
        if isinstance(r, Exception):
            logger.error("Embedding request failed for text #%d: %s", i, r)
            raise r
        r.raise_for_status()
        payload = r.json()
        # LM Studio usually returns: {"data":[{"embedding":[...] }]}
        try:
            emb = payload["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected response for text #{i}: {payload}") from e
        embeddings.append(emb)

    return embeddings


def _edge_collection_name(collection_name: str) -> str:
    """Edge collection stores relationship edges for a node collection."""
    return f"{collection_name}__edges"


def _make_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------
# Core collection / node operations
# ---------------------
@mcp.tool()
async def memory_create_collection(collection_name: str) -> Dict[str, Any]:
    """
    Create a node collection and its corresponding edge collection (if not present).
    """
    try:
        chroma_client.get_or_create_collection(name=collection_name)
        chroma_client.get_or_create_collection(name=_edge_collection_name(collection_name))
        return {"success": True, "data": {"message": f"Collections '{collection_name}' and edges created."}}
    except Exception as e:
        logger.exception("Failed creating collection")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def memory_list_collections() -> Dict[str, Any]:
    """
    List all top-level collections (excludes auto edges suffix by filter).
    """
    try:
        cols = chroma_client.list_collections()
        # attempt to read .name but support either dict or object representation
        names = []
        for c in cols:
            if isinstance(c, dict):
                names.append(c.get("name"))
            else:
                # Collection object in some versions exposes .name or .id
                names.append(getattr(c, "name", getattr(c, "id", str(c))))
        # filter out edge collections optionally
        filtered = [n for n in names if not (isinstance(n, str) and n.endswith("__edges"))]
        return {"success": True, "data": {"collections": filtered}}
    except Exception as e:
        logger.exception("Failed listing collections")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def memory_delete_collection(collection_name: str) -> Dict[str, Any]:
    """
    Delete a collection and its edges collection.
    """
    try:
        chroma_client.delete_collection(name=collection_name)
        # delete edges too (if present)
        try:
            chroma_client.delete_collection(name=_edge_collection_name(collection_name))
        except Exception:
            # ignore if edges not present
            pass
        return {"success": True, "data": {"message": f"Collection '{collection_name}' (and edges) deleted."}}
    except Exception as e:
        logger.exception("Failed deleting collection")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def memory_add_node(
    collection_name: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    node_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add a single node to a collection. Returns the node id.
    """
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
        node_id = node_id or _make_uuid()
        emb = await _fetch_embedding_async([text])
        collection.add(
            documents=[text],
            embeddings=[emb[0]],
            metadatas=[metadata or {}],
            ids=[node_id]
        )
        return {"success": True, "data": {"id": node_id, "message": "Node added."}}
    except Exception as e:
        logger.exception("Failed adding node")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def memory_add_nodes_batch(
    collection_name: str,
    texts: Sequence[str],
    metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Add multiple nodes in a single call. This is faster than repeated single adds.
    - texts: list of document strings
    - metadatas: optional list of metadata dicts or None
    - ids: optional list of ids; if not provided, UUIDs will be generated
    """
    try:
        if not texts:
            return {"success": False, "error": "Empty texts list"}

        collection = chroma_client.get_or_create_collection(name=collection_name)
        embeddings = await _fetch_embedding_async(list(texts))

        final_ids = list(ids) if ids else [_make_uuid() for _ in texts]
        if len(final_ids) != len(texts):
            return {"success": False, "error": "Length of ids must match texts"}

        final_metadatas = []
        if metadatas:
            if len(metadatas) != len(texts):
                return {"success": False, "error": "Length of metadatas must match texts"}
            final_metadatas = [m or {} for m in metadatas]
        else:
            final_metadatas = [{} for _ in texts]

        collection.add(
            documents=list(texts),
            embeddings=embeddings,
            metadatas=final_metadatas,
            ids=final_ids
        )
        return {"success": True, "data": {"ids": final_ids, "message": "Batch insert complete."}}
    except Exception as e:
        logger.exception("Failed batch insert")
        return {"success": False, "error": str(e)}


# ---------------------
# Edge (relationship) operations
# ---------------------
@mcp.tool()
async def memory_create_relationship(
    collection_name: str,
    source_node_id: str,
    target_node_id: str,
    relationship_type: str,
    properties: Optional[Dict[str, Any]] = None,
    edge_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a directional relationship stored as an 'edge' document in the edges collection.
    Edge doc fields:
      - source: source node id
      - target: target node id
      - type: relationship type
      - properties: dict
    """
    try:
        edge_col_name = _edge_collection_name(collection_name)
        edges_col = chroma_client.get_or_create_collection(name=edge_col_name)
        edge_id = edge_id or _make_uuid()
        edge_doc = f"{source_node_id}|{relationship_type}|{target_node_id}"
        metadata = {"source": source_node_id, "target": target_node_id, "type": relationship_type, "properties": properties or {}}
        # We don't need embeddings for edges, but Chroma requires an embedding if your collection uses them;
        # to be safe, store a tiny embedding (zeros) or reuse Source embedding if known. Here we omit embeddings.
        edges_col.add(
            documents=[edge_doc],
            metadatas=[metadata],
            ids=[edge_id]
        )
        return {"success": True, "data": {"edge_id": edge_id, "message": "Edge created."}}
    except Exception as e:
        logger.exception("Failed creating relationship")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def memory_list_edges(collection_name: str, node_id: Optional[str] = None, direction: str = "out") -> Dict[str, Any]:
    """
    List edges for a collection. If node_id provided, filter by source (out) or target (in).
    direction: "out" (default) or "in" or "both"
    """
    try:
        edge_col_name = _edge_collection_name(collection_name)
        edges_col = chroma_client.get_or_create_collection(name=edge_col_name)
        # Use metadata filters via .get() if supported
        where_clause = {}
        if node_id:
            if direction == "out":
                where_clause = {"source": node_id}
            elif direction == "in":
                where_clause = {"target": node_id}
            else:
                # both
                # we will do two gets and merge
                out = edges_col.get(where={"source": node_id})
                inn = edges_col.get(where={"target": node_id})
                # return combined
                return {"success": True, "data": {"out": out, "in": inn}}
        results = edges_col.get(where=where_clause) if where_clause else edges_col.get()
        return {"success": True, "data": {"edges": results}}
    except Exception as e:
        logger.exception("Failed listing edges")
        return {"success": False, "error": str(e)}


# ---------------------
# Querying & helpers
# ---------------------
def _truncate_text(s: str, max_chars: int = 400) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


@mcp.tool()
async def memory_query_graph(
    collection_name: str,
    query: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
    include_documents: bool = True,
    summarise: bool = False,
) -> Dict[str, Any]:
    """
    Query a collection using text (which will be embedded) or a provided embedding.
    - where: metadata filter dict
    - summarise: if True, returns a simple concatenated summary of the top documents
    Returns the raw Chroma results and optional summary.
    """
    try:
        col = chroma_client.get_or_create_collection(name=collection_name)
        if query_embedding is None:
            if not query:
                return {"success": False, "error": "Either query or query_embedding must be provided."}
            embeddings = await _fetch_embedding_async([query])
            query_embedding = embeddings[0]

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=[ "metadatas", "documents", "distances", "ids" ]  # ensure useful fields included
        )

        data = {"results": results}

        if summarise:
            # A conservative, library-free summariser: join top documents (truncated)
            docs = []
            # results may be nested depending on library version
            matches = results.get("documents") if isinstance(results, dict) else None
            if matches:
                # results["documents"] is typically list-of-lists when querying multiple queries;
                # since we passed single query, use first element
                if isinstance(matches, list) and matches and isinstance(matches[0], list):
                    docs = [d for d in matches[0] if d]
                elif isinstance(matches, list):
                    # fallback
                    docs = matches
            else:
                # try to fall back to results structure produced by some chroma versions
                docs = []
                docs_candidate = results.get("documents") if isinstance(results, dict) else None
                # as a very robust fallback: try extracting from top-level keys
                if not docs and isinstance(results, dict):
                    # sometimes results have 'ids' and 'metadatas'; try to assemble docs from 'documents'
                    _d = results.get("documents")
                    if _d:
                        docs = _d[0] if isinstance(_d[0], list) else _d

            # build a short summary
            top_texts = [ _truncate_text(d, max_chars=400) for d in docs[: n_results] ]
            summary = "\n\n".join(top_texts) if top_texts else ""
            data["summary"] = summary

        return {"success": True, "data": data}
    except Exception as e:
        logger.exception("Failed query")
        return {"success": False, "error": str(e)}


# ---------------------
# Example convenience function: hybrid search + follow relationships
# ---------------------
@mcp.tool()
async def memory_find_with_relations(
    collection_name: str,
    query: str,
    n_results: int = 5,
    follow_rel_depth: int = 1,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Hybrid operation:
      1. find top nodes for query
      2. optionally follow their outgoing edges up to follow_rel_depth and return connected nodes too
    Useful for returning context + related facts.
    """
    try:
        # 1) find top nodes
        qres = await memory_query_graph(collection_name, query=query, n_results=n_results, where=where, summarise=False)
        if not qres.get("success"):
            return qres

        results = qres["data"]["results"]
        ids = []
        # extract ids robustly
        ids_field = results.get("ids") or []
        # ids_field may be nested list (per query)
        if isinstance(ids_field, list):
            if ids_field and isinstance(ids_field[0], list):
                ids = [str(i) for i in ids_field[0]]
            else:
                ids = [str(i) for i in ids_field]

        related_nodes = []
        if follow_rel_depth > 0 and ids:
            # fetch edges for each id
            edge_col = chroma_client.get_or_create_collection(name=_edge_collection_name(collection_name))
            # We'll use metadata 'source' to find outgoing edges
            for node_id in ids:
                edges = edge_col.get(where={"source": node_id})
                # edges may include 'metadata' or 'metadatas' depending on chroma version
                related_nodes.append({ "node": node_id, "edges": edges })

        return {"success": True, "data": {"query_results": results, "related": related_nodes}}
    except Exception as e:
        logger.exception("Failed find_with_relations")
        return {"success": False, "error": str(e)}


# ---------------------
# Example main / run
# ---------------------
def main() -> None:
    """This function is left to call mcp.run() in your application entrypoint."""
    try:
        mcp.run()
    except Exception as e:
        logger.exception("Error starting MCP server")
        raise


if __name__ == "__main__":
    mcp.run()

