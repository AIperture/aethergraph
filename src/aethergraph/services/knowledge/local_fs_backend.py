from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
import time
from typing import Any

from aethergraph.contracts.services.knowledge import KBAnswer, KBSearchHit, KnowledgeBackend
from aethergraph.contracts.services.llm import (
    EmbeddingClientProtocol,
    LLMClientProtocol,
)
from aethergraph.contracts.storage.search_backend import ScoredItem, SearchBackend, SearchMode
from aethergraph.services.scope.scope import Scope, ScopeLevel

from .chunker import TextSplitter
from .rerank import lexical_score  # adjust path to wherever you put it


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_id(parts: Mapping[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


@dataclass
class LocalFSKnowledgeBackend(KnowledgeBackend):
    """
    Simple knowledge backend that:

    - Stores logical corpus metadata + doc/chunk records on the filesystem.
    - Uses a generic SearchBackend (vector index) for retrieval.
    - Relies on Scope to enforce org/user/scope_id isolation.

    This is intentionally "boring" and local. Other implementations
    (GraphRAG, hosted vector DB, etc.) can implement KnowledgeBackend
    with completely different internals.
    """

    corpus_root: str
    artifacts: Any  # ArtifactFacade
    search_backend: SearchBackend
    embed_client: EmbeddingClientProtocol
    llm_client: LLMClientProtocol
    chunker: TextSplitter
    logger: Any | None = None

    def _load_existing_doc_keys(self, corpus_id: str) -> set[str]:
        """
        Build a set of dedupe keys for existing docs in this corpus.

        Key shape: f"{kb_scope_id}|{kb_namespace}|{content_hash}"

        This lets us make ingestion idempotent:
        - Same user/org
        - Same kb_scope_id (i.e., same KB bucket)
        - Same kb_namespace (project/workspace)
        - Same content_hash

        => skip re-ingesting.
        """
        cdir = self._cdir(corpus_id)
        docs_jl = os.path.join(cdir, "docs.jsonl")
        keys: set[str] = set()

        if not os.path.exists(docs_jl):
            return keys

        with open(docs_jl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                meta = obj.get("meta") or {}
                content_hash = meta.get("content_hash")
                kb_scope_id = meta.get("kb_scope_id")
                kb_ns = meta.get("kb_namespace") or ""

                if content_hash and kb_scope_id:
                    keys.add(f"{kb_scope_id}|{kb_ns}|{content_hash}")

        return keys

    def _cdir(self, corpus_id: str) -> str:
        """
        Resolve logical corpus_id to a filesystem directory.

        We intentionally don't reuse old helpers. For now, we just keep a
        one-to-one mapping under corpus_root; if you want fs-safe keys,
        you can inject a helper or adjust here.
        """
        # TODO: optionally normalize / make fs-safe
        return os.path.join(self.corpus_root, corpus_id)

    async def _ensure_corpus(
        self,
        *,
        scope: Scope | None,
        corpus_id: str,
        kb_namespace: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        cdir = self._cdir(corpus_id)
        os.makedirs(cdir, exist_ok=True)
        meta_path = os.path.join(cdir, "corpus.json")

        if os.path.exists(meta_path):
            return

        scope_labels: dict[str, Any] = {}
        if scope is not None:
            scope_labels = scope.rag_labels(scope_id=scope.memory_scope_id())
        if kb_namespace:
            scope_labels.setdefault("kb_namespace", kb_namespace)

        full_meta = {
            "corpus_id": corpus_id,
            "created_at": _now_iso(),
            "meta": meta or {},
            "scope": scope_labels,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(full_meta, f, ensure_ascii=False)

    def _load_chunks_map(self, corpus_id: str) -> dict[str, dict[str, Any]]:
        cdir = self._cdir(corpus_id)
        chunks_jl = os.path.join(cdir, "chunks.jsonl")
        if not os.path.exists(chunks_jl):
            return {}

        out: dict[str, dict[str, Any]] = {}
        with open(chunks_jl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("chunk_id")
                if cid:
                    out[cid] = obj
        return out

    # ------------ knowledge backend implementation --------------------------------
    async def upsert_docs(
        self,
        *,
        scope: Scope | None,
        corpus_id: str,
        docs: list[dict[str, Any]],
        kb_namespace: str | None = None,
    ) -> dict[str, Any]:
        if not docs:
            return {"added": 0, "chunks": 0}

        await self._ensure_corpus(scope=scope, corpus_id=corpus_id, kb_namespace=kb_namespace)

        cdir = self._cdir(corpus_id)
        docs_jl = os.path.join(cdir, "docs.jsonl")
        chunks_jl = os.path.join(cdir, "chunks.jsonl")

        os.makedirs(cdir, exist_ok=True)

        # --- dedupe index: existing docs in this corpus for this KB scope/namespace ---
        existing_keys = self._load_existing_doc_keys(corpus_id)

        added_docs = 0
        total_chunks = 0

        for d in docs:
            # ---- merge labels ----
            # "user_labels" are user-facing labels (topic, source, etc.).
            user_labels: dict[str, Any] = d.get("labels") or {}

            # KB index labels: org/user + kb_scope_id (no session/run).
            kb_index_labels: dict[str, Any] = {}
            if scope is not None:
                kb_index_labels = scope.kb_index_labels()
            if kb_namespace:
                kb_index_labels["kb_namespace"] = kb_namespace

            # For doc-level metadata, we keep both:
            # - user_labels under "labels"
            # - kb_index_labels flattened for searching.
            labels = user_labels

            # If caller didn't specify a format, infer from path extension.
            if "format" not in user_labels:
                path = d.get("path")
                if path:
                    ext = os.path.splitext(path)[1].lstrip(".").lower()
                    if ext:
                        user_labels["format"] = ext

            title = d.get("title") or os.path.basename(d.get("path", "") or "") or "untitled"
            doc_id = _stable_id({"title": title, "labels": labels, "ts": _now_iso()})
            text: str | None = None
            doc_uri: str | None = None
            extra_meta: dict[str, Any] = {}

            # ---- load / persist raw content
            if "path" in d and d["path"]:
                path = d["path"]
                uri_obj = await self.artifacts.save_file(
                    path=path,
                    kind="doc",
                    run_id=scope.run_id if scope else "kb",
                    graph_id=scope.graph_id if scope else "kb",
                    node_id=scope.node_id if scope else "kb",
                    tool_name="kb.upsert",
                    tool_version="0.1.0",
                    labels=labels,
                    cleanup=False,
                )
                doc_uri = getattr(uri_obj, "uri", str(uri_obj))

                lower = path.lower()
                if lower.endswith(".pdf"):
                    from .parsers.pdf import extract_text  # type: ignore[assignment]

                    text, extra_meta = extract_text(path)
                elif lower.endswith((".md", ".markdown", ".mkd")):
                    from .parsers.md import extract_text  # type: ignore[assignment]

                    text, extra_meta = extract_text(path)
                elif lower.endswith(".rst"):
                    from .parsers.rst import extract_text  # type: ignore[assignment]

                    text, extra_meta = extract_text(path)
                else:
                    from .parsers.txt import extract_text  # type: ignore[assignment]

                    text, extra_meta = extract_text(path)

            else:
                # inline text -> stage as artifact
                payload = (d.get("text") or "").strip()
                staged_path = await self.artifacts.plan_staging_path(".txt")

                def _write(path: str, content: str) -> None:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)

                await asyncio.to_thread(_write, staged_path, payload)

                uri_obj = await self.artifacts.save_file(
                    path=staged_path,
                    kind="doc",
                    run_id=scope.run_id if scope else "kb",
                    graph_id=scope.graph_id if scope else "kb",
                    node_id=scope.node_id if scope else "kb",
                    tool_name="kb.upsert",
                    tool_version="0.1.0",
                    labels=labels,
                )
                doc_uri = getattr(uri_obj, "uri", str(uri_obj))
                text = payload
                extra_meta = {}

            text = (text or "").strip()
            if not text:
                if self.logger:
                    self.logger.warning("KB: empty text for doc %s", title)
                continue

            # --- dedupe: same user/org + kb_scope_id + kb_namespace + content_hash ---
            # content_hash is purely for dedupe; we also stash it in doc.meta.
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

            kb_scope_id = kb_index_labels.get("kb_scope_id")
            ns_for_key = kb_namespace or ""
            dedupe_key = f"{kb_scope_id or ''}|{ns_for_key}|{content_hash}"

            if dedupe_key in existing_keys:
                if self.logger:
                    self.logger.info(
                        "KB: skipping duplicate doc title=%r corpus=%s kb_ns=%s",
                        title,
                        corpus_id,
                        kb_namespace,
                    )
                # Don’t re-write docs.jsonl / chunks.jsonl / embeddings
                continue

            # Mark this doc as seen so repeated docs in the same call are also skipped.
            existing_keys.add(dedupe_key)

            # ---- write doc record ----
            doc_record = {
                "doc_id": doc_id,
                "corpus_id": corpus_id,
                "uri": doc_uri,
                "title": title,
                "meta": {
                    "labels": labels,  # user-facing labels
                    **kb_index_labels,  # org_id/user_id/kb_scope_id/kb_namespace
                    "kb_scope_id": kb_scope_id,
                    "kb_namespace": kb_namespace,
                    "content_hash": content_hash,
                    **extra_meta,
                },
                "created_at": _now_iso(),
            }

            with open(docs_jl, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

            added_docs += 1

            # ---- chunking + embedding ----
            # ---- chunking + embedding ----
            # Choose splitter mode based on format
            fmt = (labels.get("format") or "").lower()

            if fmt in {"md", "markdown", "mkd"}:
                split_mode = "markdown"
            elif fmt in {"rst"}:
                split_mode = "rst"
            elif fmt in {"py", "ipynb", "toml", "yaml", "yml"}:
                # In case you ever index code/config
                split_mode = "code"
            else:
                split_mode = "plain"

            # Reuse base chunker sizing but adjust mode per doc
            base_chunker = self.chunker

            chunker = TextSplitter(
                target_tokens=getattr(base_chunker, "n", 400),
                overlap_tokens=getattr(base_chunker, "o", 60),
                mode=split_mode,
            )

            chunks = chunker.split(text)
            if not chunks:
                continue

            for i, chunk_text in enumerate(chunks):
                chunk_id = _stable_id({"doc": doc_id, "i": i})

                meta = {
                    "doc_id": doc_id,
                    "corpus_id": corpus_id,
                    "title": title,
                    "chunk_index": i,
                    "labels": labels,  # keep user labels nested
                    **kb_index_labels,  # IMPORTANT: org_id/user_id/kb_scope_id at top level
                    "content_hash": content_hash,  # for chunk-level dedupe if needed later
                }

                chunk_record = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "corpus_id": corpus_id,
                    "text": chunk_text,
                    "meta": meta,
                }
                with open(chunks_jl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

                await self.search_backend.upsert(
                    corpus=f"{corpus_id}",
                    item_id=chunk_id,
                    text=chunk_text,
                    metadata=meta,
                )

                total_chunks += 1

        return {"added": added_docs, "chunks": total_chunks}

    async def search(
        self,
        *,
        scope: Scope | None,
        corpus_id: str,
        query: str,
        top_k: int = 10,
        kb_namespace: str | None = None,
        filters: Mapping[str, Any] | None = None,
        level: ScopeLevel | None = None,  # currently unused; reserved for future
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
        mode: SearchMode | None = None,
        lexical_rerank: bool = True,
        alpha: float = 0.8,  # fused = alpha * dense + (1-alpha) * lexical
    ) -> list[KBSearchHit]:
        """
        KB search over corpus chunks.

        Parameters:
            mode:
                Search mode passed to the underlying SearchBackend:

                    - "auto":      backend decides (usually semantic if query, structural otherwise)
                    - "semantic":  embedding-based ANN
                    - "lexical":   FTS/BM25 if available
                    - "hybrid":    backend-level fusion of semantic + lexical
                    - "structural": recency-only, ignores query vector

                If None, defaults to "semantic" for KB (dense first).

            lexical_rerank:
                If True, apply an additional lexical BM25-style rerank on top of the
                backend scores using `lexical_score(query, text)`:

                    fused_score = alpha * dense_score + (1 - alpha) * lexical_score

                If False, return backend scores as-is.

            alpha:
                Weight for backend score vs lexical score in the fusion.
        """
        # --- 1) Build filters based on scope + kb_namespace + user filters ----
        base_filters: dict[str, Any] = {}
        if scope is not None:
            # User-level KB filters: org_id + user_id + kb_scope_id
            base_filters.update(scope.kb_filter())

        if kb_namespace:
            base_filters["kb_namespace"] = kb_namespace

        merged: dict[str, Any] = {**base_filters, **(filters or {})}
        merged = {k: v for k, v in merged.items() if v is not None}

        # --- 2) Decide search mode for the SearchBackend ---------------------
        search_mode: SearchMode = mode or "semantic"

        # If we're going to lexical-rerank, over-fetch a bit from the backend.
        dense_k = min(top_k * 3, 100) if lexical_rerank else top_k

        rows: list[ScoredItem] = await self.search_backend.search(
            corpus=f"{corpus_id}",
            query=query,
            top_k=dense_k,
            filters=merged,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
            mode=search_mode,
        )

        if self.logger:
            self.logger.debug(
                "KB search: corpus=%s query=%r top_k=%s dense_k=%s "
                "mode=%s lexical_rerank=%s filters=%s -> %s hits",
                corpus_id,
                query,
                top_k,
                dense_k,
                search_mode,
                lexical_rerank,
                merged,
                len(rows),
            )

        # Need chunk map to get text/doc_id, since backend only returns metadata + score.
        chunks_map = self._load_chunks_map(corpus_id)

        # --- 3) Build KBSearchHit list and optionally lexical-rerank ---------
        hits: list[KBSearchHit] = []
        for row in rows:
            cid = row.item_id
            rec = chunks_map.get(cid, {})
            text = (rec.get("text") or "").strip()
            meta = dict(row.metadata or {})
            dense_score = row.score or 0.0

            if lexical_rerank:
                lex = lexical_score(query, text)
                fused = alpha * dense_score + (1.0 - alpha) * lex
                score = fused
            else:
                score = dense_score

            hits.append(
                KBSearchHit(
                    chunk_id=cid,
                    doc_id=rec.get("doc_id", ""),
                    corpus_id=corpus_id,
                    score=score,
                    text=text,
                    meta=meta,
                )
            )

        if lexical_rerank:
            hits.sort(key=lambda h: h.score, reverse=True)
            if len(hits) > top_k:
                hits = hits[:top_k]

        return hits

    async def answer(
        self,
        *,
        scope: Scope | None,
        corpus_id: str,
        question: str,
        style: str = "concise",
        kb_namespace: str | None = None,
        filters: Mapping[str, Any] | None = None,
        k: int = 10,
        level: ScopeLevel | None = None,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
        mode: SearchMode | None = None,
        lexical_rerank: bool = True,
    ) -> KBAnswer:
        # 1) retrieve relevant chunks
        hits = await self.search(
            scope=scope,
            corpus_id=corpus_id,
            query=question,
            top_k=k,
            kb_namespace=kb_namespace,
            filters=filters,
            level=level,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
            mode=mode,
            lexical_rerank=lexical_rerank,
        )

        if not hits:
            return KBAnswer(answer="", citations=[], usage=None, resolved_citations=[])

        context = "\n\n".join([f"[{i + 1}] {h.text}" for i, h in enumerate(hits)])
        sys = (
            "You are a helpful assistant that answers strictly from the provided context. "
            "Cite chunk numbers like [1], [2]. If the context is insufficient, you must say "
            "you don't know."
        )
        if style == "detailed":
            sys += " Be structured and explain your reasoning briefly."

        usr = f"Question: {question}\n\nContext:\n{context}"

        text, usage = await self.llm_client.chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ]
        )

        raw_citations = [
            {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "rank": i + 1} for i, h in enumerate(hits)
        ]
        resolved = self._resolve_citations(corpus_id=corpus_id, citations=raw_citations)

        return KBAnswer(
            answer=text,
            citations=raw_citations,
            usage=usage,
            resolved_citations=resolved,
        )

    def _resolve_citations(
        self, *, corpus_id: str, citations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        cdir = self._cdir(corpus_id)
        chunks_jl = os.path.join(cdir, "chunks.jl")
        docs_jl = os.path.join(cdir, "docs.jl")

        chunk_map: dict[str, Any] = {}
        doc_map: dict[str, Any] = {}

        if os.path.exists(chunks_jl):
            with open(chunks_jl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    chunk_map[o["chunk_id"]] = o

        if os.path.exists(docs_jl):
            with open(docs_jl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    doc_map[o["doc_id"]] = o

        out: list[dict[str, Any]] = []
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
                    "uri": dd.get("uri"),
                    "chunk_id": c["chunk_id"],
                    "snippet": snippet,
                }
            )
        return out

    async def list_corpora(self, *, scope: Scope | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not os.path.exists(self.corpus_root):
            return out

        for d in sorted(os.listdir(self.corpus_root)):
            cdir = os.path.join(self.corpus_root, d)
            if not os.path.isdir(cdir):
                continue
            meta_path = os.path.join(cdir, "corpus.json")
            meta: dict[str, Any] = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}

            logical_id = meta.get("corpus_id") or d
            out.append({"corpus_id": logical_id, "meta": meta})

        # NOTE: add scope-based filtering here later if needed
        return out

    async def list_docs(
        self,
        *,
        scope: Scope | None,
        corpus_id: str,
        limit: int = 200,
        after: str | None = None,
    ) -> list[dict[str, Any]]:
        cdir = self._cdir(corpus_id)
        docs_jl = os.path.join(cdir, "docs.jsonl")
        if not os.path.exists(docs_jl):
            return []

        acc: list[dict[str, Any]] = []
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

    async def delete_docs(
        self,
        *,
        scope: Scope | None,
        corpus_id: str,
        doc_ids: list[str],
    ) -> dict[str, Any]:
        cdir = self._cdir(corpus_id)
        docs_jl = os.path.join(cdir, "docs.jsonl")
        chunks_jl = os.path.join(cdir, "chunks.jsonl")

        doc_set = set(doc_ids)
        kept_docs: list[str] = []
        kept_chunks: list[str] = []
        removed_chunk_ids: list[str] = []

        if os.path.exists(chunks_jl):
            with open(chunks_jl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    if o.get("doc_id") in doc_set:
                        cid = o.get("chunk_id")
                        if cid:
                            removed_chunk_ids.append(cid)
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

        # remove from search backend
        if removed_chunk_ids:  # noqa: SIM102
            if hasattr(self.search_backend, "delete"):
                await self.search_backend.delete(
                    corpus=f"{corpus_id}",
                    item_ids=removed_chunk_ids,
                )
            # If not supported, we just leave orphaned vectors; tolerable for v1.

        return {"removed_docs": len(doc_ids), "removed_chunks": len(removed_chunk_ids)}

    async def reembed(
        self,
        *,
        scope: Scope | None,
        corpus_id: str,
        doc_ids: list[str] | None = None,
        batch: int = 64,
    ) -> dict[str, Any]:
        cdir = self._cdir(corpus_id)
        chunks_jl = os.path.join(cdir, "chunks.jsonl")
        if not os.path.exists(chunks_jl):
            return {"reembedded": 0, "model": getattr(self.embed_client, "embed_model", None)}

        targets: list[dict[str, Any]] = []
        doc_set = set(doc_ids) if doc_ids is not None else None

        with open(chunks_jl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                if doc_set is None or o.get("doc_id") in doc_set:
                    targets.append(o)

        added = 0
        for i in range(0, len(targets), batch):
            batch_ch = targets[i : i + batch]
            texts = [t["text"] for t in batch_ch]
            vecs = await self.embed_client.embed(texts)
            for t, _ in zip(batch_ch, vecs, strict=True):
                meta = t.get("meta") or {}
                await self.search_backend.upsert(
                    corpus=f"{corpus_id}",
                    item_id=t["chunk_id"],
                    text=t["text"],
                    metadata=meta,
                )
                added += 1

        return {"reembedded": added, "model": getattr(self.embed_client, "embed_model", None)}

    async def stats(
        self,
        *,
        scope: Scope | None,
        corpus_id: str,
    ) -> dict[str, Any]:
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

        meta: dict[str, Any] = {}
        meta_path = os.path.join(cdir, "corpus.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        return {
            "corpus_id": corpus_id,
            "docs": n_docs,
            "chunks": n_chunks,
            "meta": meta,
        }
