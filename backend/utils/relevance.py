# backend/utils/relevance.py
from __future__ import annotations
from typing import List, Tuple, Iterable
import math

# Lazy, optional: sentence-transformers; fallback to TF-IDF if not installed.
_EMB = None
_TFIDF = None

def _try_import_st():
    import os
    # Force TF-IDF if environment variable is set
    if os.environ.get("CEGO_EMBEDDER", "").lower() == "tfidf":
        return None

    try:
        from sentence_transformers import SentenceTransformer
        import pathlib
        # Only use ST if model is already cached locally (no downloads)
        cache_dir = pathlib.Path.home() / '.cache' / 'huggingface' / 'transformers'
        if not cache_dir.exists():
            return None  # No cache, use TF-IDF fallback
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None

def _norm(v):
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

def _cos(u, v):
    return float(sum(a*b for a, b in zip(u, v)))

def _embed_st(texts: List[str]) -> List[List[float]]:
    global _EMB
    if _EMB is None:
        _EMB = _try_import_st()
    if _EMB is None:
        raise RuntimeError("sentence-transformers unavailable")
    return _EMB.encode(texts, normalize_embeddings=True).tolist()

def _embed_tfidf(texts: List[str]) -> List[List[float]]:
    # Lightweight fallback
    global _TFIDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    if _TFIDF is None:
        _TFIDF = TfidfVectorizer(max_features=2048)
        _TFIDF.fit(texts)
    X = _TFIDF.transform(texts).astype("float32")
    # row-normalize to unit vectors
    norms = (X.multiply(X)).sum(axis=1).A1 ** 0.5
    norms[norms == 0] = 1.0
    X = X.multiply(1.0 / norms[:, None])
    return [X.getrow(i).toarray().ravel().tolist() for i in range(X.shape[0])]

def embed(texts: List[str]) -> List[List[float]]:
    try:
        return _embed_st(texts)
    except Exception:
        return _embed_tfidf(texts)

def embed_one(text: str) -> List[float]:
    return embed([text])[0]

def relevance_scores(query: str, chunks: List[str]) -> List[Tuple[str, float]]:
    if not chunks: return []
    embs = embed([query] + chunks)
    q = embs[0]
    pairs = []
    for i, c in enumerate(chunks):
        s = _cos(q, embs[i+1])
        pairs.append((c, max(0.0, min(1.0, s))))
    return pairs

import os
import re
from statistics import median

STOPWORDS = set("""
a an and are as at be by for from has have in is it its of on or that the to was were will with
""".split())

JUNK_PAT = os.getenv(
    "CEGO_JUNK_REGEXP",
    r"\b(weather|forecast|temperature|rain|sunny|cloudy|snow|humidity|wind|uv index|"
    r"recipe|cooking|cake|pizza|dessert|"
    r"sports?|football|baseball|basketball|soccer|tennis|golf|playoffs?|"
    r"horoscope|lottery|celebrity|entertainment|"
    r"stock\s+price|crypto|coin|token)\b"
)
JUNK_RE = re.compile(JUNK_PAT, re.IGNORECASE)

def _content_tokens(s: str):
    return [w for w in re.findall(r"[A-Za-z0-9]+", s.lower()) if w not in STOPWORDS]

def _content_overlap_ratio(query: str, chunk: str) -> float:
    q = set(_content_tokens(query))
    c = set(_content_tokens(chunk))
    if not q or not c:
        return 0.0
    inter = len(q & c)
    return inter / max(1, len(c))

# Tunables (env overrides allowed)
DEFAULT_REL_THRESH = float(os.getenv("CEGO_REL_THRESH", "0.25"))

# Optional domain hints (give a tiny boost when query suggests a domain)
DEFAULT_DOMAIN_HINTS = os.getenv(
    "CEGO_DOMAIN_HINTS",
    # comma-separated keyword groups; add your domains as needed
    "insurance:underwriting,premium,policy,claims,risk,applicant,carrier,actuarial,effective,endorsement"
)

def _parse_domain_hints():
    hints = {}
    for block in DEFAULT_DOMAIN_HINTS.split(";"):
        block = block.strip()
        if not block: continue
        if ":" in block:
            name, kws = block.split(":", 1)
            hints[name.strip()] = {k.strip().lower() for k in kws.split(",") if k.strip()}
        else:
            # flat list
            hints.setdefault("generic", set()).update({k.strip().lower() for k in block.split(",") if k.strip()})
    return hints

_DOMAIN_HINTS = _parse_domain_hints()

def _domain_boost(query: str, chunk: str) -> float:
    q = query.lower()
    c = chunk.lower()
    boost = 0.0
    for _, kws in _DOMAIN_HINTS.items():
        if any(k in q for k in kws) and any(k in c for k in kws):
            boost = max(boost, 0.10)  # small, safe bump
    return boost

def prefilter_by_relevance(
    query: str,
    chunks: List[str],
    thresh: float = DEFAULT_REL_THRESH,
    keep_min: int = 5,
    adapt: bool = True,
) -> List[str]:
    """
    Hardened prefilter with junk suppression and content overlap gate:
    - Hard suppress junk content unless query is about that domain
    - Cap scores for chunks with no content overlap with query
    - Use adaptive thresholds for coverage
    - Never returns empty unless chunks is empty
    """
    if not chunks:
        return []

    pairs = relevance_scores(query, chunks)  # [(chunk, score)]
    if not pairs:
        return []

    adj = []
    q_is_junk_domain = bool(JUNK_RE.search(query))

    for c, s in pairs:
        s_orig = s

        # 1) HARD junk suppression unless query itself is about junk domain
        if not q_is_junk_domain and JUNK_RE.search(c):
            s = 0.0

        # 2) Content-overlap gate: if no non-stopword overlap with query,
        #    cap score so it can't beat real content in adaptive top-K.
        if _content_overlap_ratio(query, c) == 0.0:
            s = min(s, 0.10)

        # 3) Domain-aware micro-boost
        s += _domain_boost(query, c)
        s = max(0.0, min(1.0, s))
        adj.append((c, s))

    # Absolute threshold
    keep = [c for c, s in adj if s >= thresh]

    # Adaptive threshold (percentile/median) if too few
    if adapt and len(keep) < min(keep_min, len(chunks)):
        scores = [s for _, s in adj]
        scores_sorted = sorted(scores)
        p30 = scores_sorted[max(0, int(0.30 * (len(scores_sorted)-1)))]
        dyn = max(0.15, p30, median(scores) * 0.75)
        keep = [c for c, s in adj if s >= dyn]

    # Backstop: top-K by adjusted score (already penalized junk & no-overlap)
    if len(keep) < min(keep_min, len(chunks)):
        keep = [c for c, _ in sorted(adj, key=lambda x: x[1], reverse=True)[:min(keep_min, len(chunks))]]

    # Final fallback - avoid original chunks if they include junk
    if not keep:
        non_junk_chunks = [c for c in chunks if not (not q_is_junk_domain and JUNK_RE.search(c))]
        keep = non_junk_chunks[:min(keep_min, len(non_junk_chunks))] or chunks[:1]

    return keep

def mmr_rank(query: str, candidates: List[str], gamma_entropy: List[float] | None = None,
             alpha: float = 0.55, beta: float = 0.25, gamma: float = 0.20, top_k: int | None = None) -> List[int]:
    """
    MMR with ΔH term: score = α*rel(q,c) - β*max_sim(c,S) + γ*ΔH_norm(c)
    Returns indices in pick order; ΔH list aligns with candidates (0..1 normalized). ΔH may be None -> treated as 0.
    """
    if not candidates: return []

    # Single embed for numerical stability (avoid TF-IDF re-fitting)
    embs_all = embed([query] + candidates)
    qv, embs = embs_all[0], embs_all[1:]
    rel = [_cos(qv, e) for e in embs]
    dh = gamma_entropy or [0.0] * len(candidates)

    selected: List[int] = []
    remain = set(range(len(candidates)))
    K = top_k or len(candidates)

    # Precompute candidate-candidate similarities lazily
    def max_sim_to_selected(i: int) -> float:
        if not selected: return 0.0
        ei = embs[i]
        return max(_cos(ei, embs[j]) for j in selected)

    for _ in range(K):
        best_i, best_score = None, -1e9
        for i in list(remain):
            score = alpha * rel[i] - beta * max_sim_to_selected(i) + gamma * dh[i]
            if score > best_score:
                best_score, best_i = score, i
        if best_i is None:
            break
        selected.append(best_i)
        remain.discard(best_i)  # Use discard instead of remove for safety
    return selected

def semantic_retention(original: List[str], kept: List[str]) -> float:
    """
    Calculate semantic retention between original and optimized context.

    Returns cosine similarity between embeddings of concatenated content.
    Higher values indicate better preservation of original meaning.
    """
    if not original or not kept:
        return 0.0

    orig_emb = embed_one("\n".join(original))
    kept_emb = embed_one("\n".join(kept))
    return float(_cos(orig_emb, kept_emb))
