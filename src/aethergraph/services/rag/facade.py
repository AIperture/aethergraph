from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import time
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.services.scope.scope import Scope

from .chunker import TextSplitter
from .utils.hybrid import topk_fuse
from .utils.make_fs_key import make_fs_key


@dataclass
class SearchHit:
    """A single search hit from RAG retrieval."""

    chunk_id: str
    doc_id: str
    corpus_id: str
    score: float
    text: str
    meta: dict[str, Any]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_id(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


class RAGFacade:
    """Facade for RAG operations: corpus management, document ingestion, retrieval, and QA."""

    def __init__(
        self,
        *,
        corpus_root: str,
        artifacts,
        embed_client: LLMClientProtocol,
        llm_client: LLMClientProtocol,
        index_backend,
        chunker: TextSplitter,
        logger=None,
    ):
        """Initialize RAGFacade with storage paths and service clients.
        Args:
            corpus_root: Root directory for storing corpora.
            artifacts: Artifact storage facade.
            embed_client: Embedding service client.
            index_backend: Vector index backend.
            chunker: TextSplitter instance for chunking documents.
            logger: Optional logger for logging messages.
        """
        self.root = corpus_root
        self.artifacts = artifacts
        self.embed = embed_client
        self.llm = llm_client
        self.index = index_backend
        self.chunker = chunker
        self.logger = logger

        # self.logger.info(f"RAGFacade initialized with corpus root: {self.root}, index: {type(self.index).__name__}, embed model: {getattr(self.embed, 'embed_model', None)}, llm model: {getattr(self.llm, 'model', None)}")

    def set_llm_client(self, client: LLMClientProtocol) -> None:
        """Set the LLM client to use for answering questions."""
        assert client.model is not None, "RAG LLM client must have a model set"
        assert client.embed_model is not None, "RAG LLM client must have an embedding model set"
        self.llm = client
        self.logger.info(
            f"RAG LLM client set to model: {self.llm.model}, embed model: {self.llm.embed_model}"
        )

    def set_index_backend(self, index_backend) -> None:
        """Set the vector index backend."""
        self.index = index_backend
        self.logger.info(f"RAG index backend set to: {type(self.index).__name__}")

    def _cdir(self, corpus_id: str) -> str:
        """Get corpus directory path based on corpus ID while ensuring the path work safely across OS.
        Args:
            corpus_id: Unique identifier for the corpus.
        Returns:
            Path to the corpus directory.
        """

        return os.path.join(self.root, make_fs_key(corpus_id))

    # ---------- ingestion ----------
    async def add_corpus(
        self,
        corpus_id: str,
        meta: dict[str, Any] | None = None,
        *,
        scope_labels: dict[str, str] | None = None,
    ):
        """Create a new corpus with optional metadata.
        Args:
            corpus_id: Unique identifier for the corpus.
            meta: Optional metadata dictionary to store with the corpus.
        """
        p = self._cdir(corpus_id)
        os.makedirs(p, exist_ok=True)
        meta_path = os.path.join(p, "corpus.json")
        if not os.path.exists(meta_path):
            full_meta = {
                "corpus_id": corpus_id,
                "fs_key": make_fs_key(corpus_id),
                "created_at": _now_iso(),
                "meta": meta or {},
            }
            if scope_labels:
                full_meta.setdefault("meta", {})
                full_meta["meta"]["scope"] = dict(scope_labels)

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(full_meta, f)

    async def upsert_docs(
        self,
        corpus_id: str,
        docs: list[dict[str, Any]],
        *,
        scope: Scope | None = None,
        scope_id: str | None = None,  # e.g. memory_scope_id if tied to memory
    ) -> dict[str, Any]:
        """Ingest and index a list of documents into the specified corpus.
        Args:
            corpus_id: The target corpus identifier.
            docs: A list of document specifications.

        Docs can be specified as either:
            - File-based documents: {"path": "/path/to/doc.pdf", "labels": {...}}
            - Inline text documents: {"text": "Document content...", "title": "Doc Title", "labels": {...}}
        """

        if not self.embed:
            raise RuntimeError("RAGFacade: embed client not configured")

        scope_labels: dict[str, str] = {}
        if scope is not None:
            scope_labels = scope.rag_labels(scope_id=scope_id)

        await self.add_corpus(corpus_id, meta=None, scope_labels=scope_labels)

        cdir = self._cdir(corpus_id)
        docs_jl = os.path.join(cdir, "docs.jsonl")
        chunks_jl = os.path.join(cdir, "chunks.jsonl")
        os.makedirs(cdir, exist_ok=True)

        added_docs = 0
        all_chunk_ids, all_vecs, all_metas = [], [], []
        total_chunks = 0

        for d in docs:
            # Merge scope labels into provided labels
            labels = {**scope_labels, **(d.get("labels", {}) or {})}
            title = d.get("title") or os.path.basename(d.get("path", "")) or "untitled"
            doc_id = _stable_id({"title": title, "labels": labels, "ts": _now_iso()})
            text = None
            extra_meta = {}

            if "path" in d and os.path.exists(d["path"]):
                # save original file into artifacts CAS and parse
                uri = await self.artifacts.save_file(
                    path=d["path"],
                    kind="doc",
                    run_id=scope.run_id if scope else "rag",
                    graph_id=scope.graph_id if scope else "rag",
                    node_id=scope.node_id if scope else "rag",
                    tool_name="rag.upsert",
                    tool_version="0.1.0",
                    labels=labels,
                    cleanup=False,  # keep source file as this is the original
                )
                path = d["path"].lower()
                if path.endswith(".pdf"):
                    from .parsers.pdf import extract_text

                    text, extra_meta = extract_text(d["path"])  # type: ignore
                elif path.endswith(".md") or path.endswith(".markdown") or path.endswith(".mkd"):
                    from .parsers.md import extract_text

                    text, extra_meta = extract_text(d["path"])  # type: ignore
                else:
                    from .parsers.txt import extract_text

                    text, extra_meta = extract_text(d["path"])  # type: ignore
                doc_uri = uri.uri if hasattr(uri, "uri") else uri
            else:
                # inline text doc — persist as artifact first
                payload = d.get("text", "")

                # stage and save:
                staged = await self.artifacts.plan_staging_path(".txt")
                payload = d.get("text", "")

                def _write_staged(path: str, content: str) -> None:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)

                await asyncio.to_thread(_write_staged, staged, payload)

                a = await self.artifacts.save_file(
                    path=staged,
                    kind="doc",
                    run_id=scope.run_id if scope else "rag",
                    graph_id=scope.graph_id if scope else "rag",
                    node_id=scope.node_id if scope else "rag",
                    tool_name="rag.upsert",
                    tool_version="0.1.0",
                    labels=labels,
                )

                doc_uri = a.uri if hasattr(a, "uri") else a
                text = payload

            text = (text or "").strip()
            if not text:
                if self.logger:
                    self.logger.warning(f"RAG: empty text for doc {title}")
                continue

            # write doc record with labels including scope
            with open(docs_jl, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "corpus_id": corpus_id,
                            "uri": doc_uri,
                            "title": title,
                            "meta": {"labels": labels, **extra_meta},
                            "created_at": _now_iso(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            added_docs += 1

            # chunk + embed
            chunks = self.chunker.split(text)
            if not chunks:
                continue

            # batch embed
            vecs = await self.embed.embed(chunks)
            for i, (chunk_text, vec) in enumerate(zip(chunks, vecs, strict=True)):
                chunk_id = _stable_id({"doc": doc_id, "i": i})
                meta = {"doc_id": doc_id, "title": title, "i": i, "labels": labels}
                # append chunk record
                with open(chunks_jl, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "chunk_id": chunk_id,
                                "doc_id": doc_id,
                                "corpus_id": corpus_id,
                                "text": chunk_text,
                                "meta": meta,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                all_chunk_ids.append(chunk_id)
                all_vecs.append(vec)
                all_metas.append({**meta})
            total_chunks += len(chunks)

        # add to index
        if all_chunk_ids:
            await self.index.add(corpus_id, all_chunk_ids, all_vecs, all_metas)

        return {"added": added_docs, "chunks": total_chunks, "index": type(self.index).__name__}

    # ---------- retrieval ----------
    def _load_chunks_map(self, corpus_id: str) -> dict[str, dict[str, Any]]:
        """Load chunk metadata for a given corpus."""
        # Load latest chunk text+meta into a dict
        cdir = self._cdir(corpus_id)
        chunks_jl = os.path.join(cdir, "chunks.jsonl")
        out = {}
        if not os.path.exists(chunks_jl):
            return out
        with open(chunks_jl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                out[obj["chunk_id"]] = obj
        return out

    def _apply_filters(
        self,
        corpus_id: str,
        hits: list[dict[str, Any]],
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Apply filters to the search hits."""
        if not filters:
            return hits

        # We need labels to test filters. They are in meta["labels"] for each chunk.
        # hits come from index.search as [{"chunk_id", "score", "meta": {...}}, ...].
        # It works as follows:
        # 1. For each hit, we extract the labels from the meta information.
        # 2. We then check if the labels match the desired filters.
        out = []
        for h in hits:
            meta = h.get("meta", {}) or {}
            labels = meta.get("labels", {}) or {}
            ok = True
            for k, want in filters.items():
                val = labels.get(k)
                if isinstance(want, list | tuple | set):
                    if val not in want:
                        ok = False
                        break
                else:
                    if val != want:
                        ok = False
                        break
            if ok:
                out.append(h)
        return out

    async def search(
        self,
        corpus_id: str,
        query: str,
        k: int = 8,
        filters: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        """Search the corpus for relevant chunks given a query.
        Args:
            corpus_id: Target corpus identifier.
            query: The search query string.
            k: Number of top results to return.
            filters: Optional metadata filters to apply.
            mode: Search mode - "dense", "hybrid".
        """
        if not self.embed:
            raise RuntimeError("RAGFacade: embed client not configured")

        # dense search via index then optional lexical fusion
        qvec = (await self.embed.embed([query]))[0]
        dense_hits = await self.index.search(corpus_id, qvec, max(24, k))

        # apply filters before fusion
        dense_hits = self._apply_filters(corpus_id, dense_hits, filters=filters)

        chunks_map = self._load_chunks_map(corpus_id)

        # if only dense or no hits, return directly
        if mode == "dense" or not dense_hits:
            dense_hits = dense_hits[:k]
            return [
                SearchHit(
                    chunk_id=h["chunk_id"],
                    doc_id=chunks_map.get(h["chunk_id"], {}).get("doc_id", ""),
                    corpus_id=corpus_id,
                    score=h["score"],
                    text=chunks_map.get(h["chunk_id"], {}).get("text", ""),
                    meta=h.get("meta", {}),
                )
                for h in dense_hits
            ]

        # hybrid fusion: i.e. dense + lexical
        fused = topk_fuse(
            query, dense_hits, {cid: rec.get("text", "") for cid, rec in chunks_map.items()}, k
        )
        out = []
        for h in fused:
            rec = chunks_map.get(h["chunk_id"], {})
            out.append(
                SearchHit(
                    chunk_id=h["chunk_id"],
                    doc_id=rec.get("doc_id", ""),
                    corpus_id=corpus_id,
                    score=h["score"],
                    text=rec.get("text", ""),
                    meta=h.get("meta", {}),
                )
            )
        return out

    async def search_scoped(
        self,
        *,
        curpus_id: str,
        query: str,
        scope: Scope | None = None,
        scope_id: str | None = None,  # e.g. memory_scope_id if tied to memory, can be None
        k: int = 8,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        """
        Convenience wrapper to search with scope-based filters.
        Args:
            curpus_id: Target corpus identifier.
            query: The search query string.
            scope: Scope object for filtering.
            k: Number of top results to return.
            mode: Search mode - "dense", "hybrid".
        """
        filters: dict[str, Any] | None = None
        if scope is not None:
            # build filters from scope labels
            filters = scope.rag_filter(scope_id=scope_id)  # scope_id is optional

        return await self.search(
            curpus_id,
            query,
            k=k,
            filters=filters,
            mode=mode,
        )

    async def retrieve(
        self, corpus_id: str, query: str, k: int = 6, rerank: bool = True
    ) -> list[SearchHit]:
        """Retrieve top-k relevant chunks for a query from the corpus.
        Args:
            corpus_id: Target corpus identifier.
            query: The retrieval query string.
            k: Number of top results to return.
            rerank: Whether to rerank results using hybrid scoring.
        """
        print(
            f"🍏 RAGFacade.retrieve: corpus_id={corpus_id}, query={query}, k={k}, rerank={rerank}"
        )
        # For now, rerank flag is ignored; fused hybrid already sorts reasonably.
        return await self.search(corpus_id, query, k=k, mode="hybrid")

    async def answer(
        self,
        corpus_id: str,
        question: str,
        *,
        llm: LLMClientProtocol | None = None,
        style: str = "concise",
        with_citations: bool = True,
        k: int = 6,
        scope: Scope | None = None,
        scope_id: str | None = None,  # e.g. memory_scope_id if tied to memory, can be None
    ) -> dict[str, Any]:
        """Answer a question using retrieved context from the corpus.
        Args:
            corpus_id: Target corpus identifier.
            question: The question to answer.
            llm: Language model client for generating the answer. If None, uses default LLM.
            style: Answering style - "concise" or "detailed".
            with_citations: Whether to include citations in the answer.
            k: Number of context chunks to retrieve.
        """
        if not llm:
            # use default LLM client
            llm = self.llm

        filters: dict[str, Any] | None = None
        if scope is not None:
            # build filters from scope labels
            filters = scope.rag_filter(scope_id=scope_id)  # scope_id is optional

        hits = await self.search(
            corpus_id,
            question,
            k=k,
            filters=filters,
            mode="hybrid",
        )

        context = "\n\n".join([f"[{i + 1}] {h.text}" for i, h in enumerate(hits)])
        sys = "You answer strictly from the provided context. Cite chunk numbers like [1],[2]. If insufficient, say you don't know."
        if style == "detailed":
            sys += " Be structured and explain reasoning briefly."
        usr = f"Question: {question}\n\nContext:\n{context}"
        text, usage = await llm.chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
        )
        out = {
            "answer": text,
            "citations": [
                {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "rank": i + 1}
                for i, h in enumerate(hits)
            ],
            "usage": usage,
        }
        if with_citations:
            out["resolved_citations"] = self.resolve_citations(corpus_id, out["citations"])
        return out

    def resolve_citations(self, corpus_id: str, citations: list[dict]) -> list[dict]:
        """Return [{rank, doc_id, title, uri, chunk_id, snippet}] sorted by rank."""
        # load chunks + doc meta
        cdir = self._cdir(corpus_id)
        chunks_jl = os.path.join(cdir, "chunks.jsonl")
        docs_jl = os.path.join(cdir, "docs.jsonl")

        # build maps
        chunk_map, doc_map = {}, {}
        if os.path.exists(chunks_jl):
            with open(chunks_jl, encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    chunk_map[o["chunk_id"]] = o
        if os.path.exists(docs_jl):
            with open(docs_jl, encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    doc_map[o["doc_id"]] = o

        out = []
        for c in sorted(citations, key=lambda x: x["rank"]):
            ch = chunk_map.get(c["chunk_id"], {})
            dd = doc_map.get(c["doc_id"], {})
            text = (ch.get("text") or "").strip().replace("\n", " ")
            snippet = (text[:220] + "…") if len(text) > 220 else text
            out.append(
                {
                    "rank": c["rank"],
                    "doc_id": c["doc_id"],
                    "title": dd.get("title", "(untitled)"),
                    "uri": dd.get("uri"),  # CAS or file URI from artifact store
                    "chunk_id": c["chunk_id"],
                    "snippet": snippet,
                }
            )
        return out

    async def list_corpora(self) -> list[dict]:
        out = []
        for d in sorted(os.listdir(self.root)):
            # cdir = self._cdir(d)
            cdir = os.path.join(self.root, d)  # d is already fs_key
            if not os.path.isdir(cdir):
                continue
            meta_path = os.path.join(cdir, "corpus.json")
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}
            # Prefer the recorded logical id (i.e. corpus_id); fall back to folder name (which may be fs-safe key)
            logical_id = meta.get("corpus_id") or meta.get("logical_id") or d
            out.append({"corpus_id": logical_id, "meta": meta})
        return out

    async def list_docs(
        self, corpus_id: str, limit: int = 200, after: str | None = None
    ) -> list[dict]:
        cdir = self._cdir(corpus_id)
        docs_jl = os.path.join(cdir, "docs.jsonl")
        if not os.path.exists(docs_jl):
            return []
        acc: list[dict] = []
        seen_after = after is None
        with open(docs_jl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not seen_after:
                    if obj.get("doc_id") == after:
                        seen_after = True
                    continue
                acc.append(obj)
                if len(acc) >= limit:
                    break
        return acc

    async def delete_docs(self, corpus_id: str, doc_ids: list[str]) -> dict:
        """
        Removes docs from docs.jsonl and any chunks in chunks.jsonl; asks the index to drop vectors if supported.
        """
        cdir = self._cdir(corpus_id)
        docs_jl = os.path.join(cdir, "docs.jsonl")
        chunks_jl = os.path.join(cdir, "chunks.jsonl")
        kept_docs, kept_chunks = [], []
        removed_chunks = []
        doc_set = set(doc_ids)

        if os.path.exists(chunks_jl):
            with open(chunks_jl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    if o.get("doc_id") in doc_set:
                        removed_chunks.append(o.get("chunk_id"))
                    else:
                        kept_chunks.append(line)
            with open(chunks_jl, "w", encoding="utf-8") as f:
                f.writelines(kept_chunks)

        if os.path.exists(docs_jl):
            with open(docs_jl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    if o.get("doc_id") not in doc_set:
                        kept_docs.append(line)
            with open(docs_jl, "w", encoding="utf-8") as f:
                f.writelines(kept_docs)

        # drop from index if supported
        if hasattr(self.index, "remove"):
            await self.index.remove(corpus_id, removed_chunks)
        elif hasattr(self.index, "delete"):
            await self.index.delete(corpus_id, removed_chunks)

        return {"removed_docs": len(doc_ids), "removed_chunks": len(removed_chunks)}

    async def reembed(
        self, corpus_id: str, *, doc_ids: list[str] | None = None, batch: int = 64
    ) -> dict:
        """
        Re-embeds selected docs (or all) and re-adds vectors. Uses the configured embed client or a model override if your client supports it.
        """
        cdir = self._cdir(corpus_id)
        chunks_jl = os.path.join(cdir, "chunks.jsonl")
        if not os.path.exists(chunks_jl):
            return {"reembedded": 0}

        targets: list[dict] = []
        with open(chunks_jl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                if doc_ids is None or o.get("doc_id") in set(doc_ids):
                    targets.append(o)

        # set model on embed client if supported
        embed = self.embed

        # Re-embed in batches
        added = 0
        for i in range(0, len(targets), batch):
            batch_ch = targets[i : i + batch]
            vecs = await embed.embed([t["text"] for t in batch_ch])
            chunk_ids = [t["chunk_id"] for t in batch_ch]
            metas = [t.get("meta", {}) for t in batch_ch]
            await self.index.add(corpus_id, chunk_ids, vecs, metas)
            added += len(batch_ch)
        return {"reembedded": added, "model": getattr(embed, "embed_model", None)}

    async def stats(self, corpus_id: str) -> dict:
        cdir = self._cdir(corpus_id)
        docs_jl = os.path.join(cdir, "docs.jsonl")
        chunks_jl = os.path.join(cdir, "chunks.jsonl")

        def _count_lines(path: str) -> int:
            if not os.path.exists(path):
                return 0
            with open(path, encoding="utf-8") as f:
                return sum(1 for _ in f)

        n_docs = _count_lines(docs_jl)
        n_chunks = _count_lines(chunks_jl)

        meta = {}
        meta_path = os.path.join(cdir, "corpus.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        return {"corpus_id": corpus_id, "docs": n_docs, "chunks": n_chunks, "meta": meta}

    async def export(self, corpus_id: str) -> dict:
        """
        Create a simple tarball-like directory bundle (docs+chunks+corpus.json) and persist via artifacts store.
        """
        raise NotImplementedError("RAGFacade.export is not yet implemented")
        # TODO: implement proper temp dir cleanup
        cdir = self._cdir(corpus_id)
        bundle_dir = os.path.join(cdir, f"bundle_{_now_iso().replace(':', '').replace('-', '')}")
        os.makedirs(bundle_dir, exist_ok=True)
        for name in ("corpus.json", "docs.jsonl", "chunks.jsonl"):
            p = os.path.join(cdir, name)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(bundle_dir, name))
        # Save dir via artifacts as a bundle
        uri = await self.artifacts.save_dir(bundle_dir, labels={"corpus_id": corpus_id})
        return {"uri": getattr(uri, "uri", uri)}

    async def import_bundle(self, bundle_uri: str, into_corpus: str | None = None) -> dict:
        """
        Resolve artifact dir and merge into an existing/new corpus.
        """
        raise NotImplementedError("RAGFacade.import_bundle is not yet implemented")
        # TODO: implement proper temp dir cleanup
        # Assuming artifacts can resolve a dir path from URI
        bundle_path = await self.artifacts.resolve_dir(bundle_uri)
        with open(os.path.join(bundle_path, "corpus.json"), encoding="utf-8") as f:
            meta = json.load(f)
        target = into_corpus or meta.get("corpus_id")
        await self.add_corpus(target, meta=meta.get("meta", {}))

        # Append docs & chunks
        for name in ("docs.jsonl", "chunks.jsonl"):
            src = os.path.join(bundle_path, name)
            if not os.path.exists(src):
                continue
            dst = os.path.join(self._cdir(target), name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "a", encoding="utf-8") as out_f, open(src, encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)
        return {"imported_into": target}
