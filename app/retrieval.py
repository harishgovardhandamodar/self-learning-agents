"""
Retrieval: similarity, unsupervised routing, and workflow fingerprinting.

The paper's central diagnosis of experience-augmentation agents is that
*query similarity alone* is not enough to decide whether a stored experience
is applicable:

  * "Before a group's block begins, 100% of retrievals come from earlier
     blocks"  -> foreign memories injected into unseen tasks.
  * "BFCL-Miss and BFCL-Base queries ... exhibit a mean pairwise cosine
     similarity of 0.667 ... No similarity threshold can separate the two
     groups."  -> retrieval collision: same topic, opposite workflow.

So the lab separates three things that similarity conflates:

  similarity   - topical closeness of the query
  scope        - which task family the experience was learned in
  workflow     - what the experience *does* (ask vs act, measure vs skip)

`Router` provides scope without labels (online clustering).
`probe_workflow` provides workflow from the request itself.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "and", "or",
    "of", "to", "in", "on", "for", "with", "at", "by", "from", "as", "that", "this",
    "these", "those", "it", "its", "i", "my", "me", "you", "your", "we", "our", "do",
    "does", "did", "can", "could", "would", "should", "will", "shall", "please", "give",
    "show", "list", "tell", "what", "which", "who", "how", "many", "much", "there",
    "have", "has", "had", "but", "not", "no", "so", "if", "then", "than", "about",
    "into", "over", "up", "down", "out", "off", "again", "more", "most", "some", "such",
}

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, split, drop stopwords, and stem plurals crudely."""
    out = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class SimilarityIndex:
    """
    TF-IDF cosine over content words, maintained incrementally.

    Used instead of an embedding model when running in simulated mode (no LLM
    round-trip per query). In live mode the runner swaps in real embeddings for
    the same role, because embeddings capture paraphrases that TF-IDF misses.
    """

    def __init__(self) -> None:
        self._df: Counter = Counter()
        self._n: int = 0
        self._docs: Dict[str, Counter] = {}
        self._texts: Dict[str, str] = {}

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        if doc_id in self._docs:
            # re-index: subtract old tokens
            for tok in self._docs[doc_id]:
                self._df[tok] -= 1
        self._docs[doc_id] = Counter(toks)
        self._texts[doc_id] = text
        for tok in self._docs[doc_id]:
            self._df[tok] += 1
        self._n = len(self._docs)

    def idf(self, tok: str) -> float:
        df = self._df.get(tok, 0)
        return math.log((self._n + 1) / (df + 1)) + 1.0

    def vector(self, text: str) -> Dict[str, float]:
        toks = tokenize(text)
        if not toks:
            return {}
        tf = Counter(toks)
        vec = {t: (1 + math.log(c)) * self.idf(t) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.vector(a), self.vector(b)
        if not va or not vb:
            return 0.0
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(v * vb.get(t, 0.0) for t, v in va.items())


# ------------------------------------------------------------ workflow probe

WORKFLOW_LABELS = {
    "missing-arg": "requests that are missing a required argument",
    "complete-arg": "requests whose required arguments are all supplied",
    "default": "general behaviour",
}

_ID_TOKEN_RE = re.compile(
    r"\b(u_\d+|sku[-_ ]?\w+|c_\d+|acct[-_ ]?\w+|account[-_ ]?\w+)\b", re.IGNORECASE
)
_HEDGE_RE = re.compile(
    r"(don't have|do not have|cannot remember|can't remember|need to look up|"
    r"you'll need|ill grab|i'll grab|look it up|missing|don't know|not to hand|"
    r"whatever the|search the catalogue for something)",
    re.IGNORECASE,
)
# A quoted literal is an argument value the requester supplied. The quotes
# must not look like an apostrophe in a contraction ("I'd", "I've"), so the
# opening quote may not follow a letter and the closing quote may not be
# followed by one.
_QUOTED_RE = re.compile(r"(?<![A-Za-z0-9])[\"'][^\"']{2,}[\"'](?![A-Za-z0-9])")


def probe_workflow(task) -> str:
    """
    Infer, from the request text alone, which workflow it demands.

    This is the piece similarity cannot give you: two requests can be about the
    same thing and still demand opposite behaviour. A local LLM can do the same
    job; the probe is the deterministic stand-in used in simulated mode.
    """
    env = getattr(task, "env", "")
    if env != "toolcall":
        return "default"
    q = task.query or ""
    if _HEDGE_RE.search(q):
        # The request says outright that something is missing or has to be
        # looked up. Hedging wins over any argument the text happens to carry.
        return "missing-arg"
    if _ID_TOKEN_RE.search(q) or _QUOTED_RE.search(q):
        return "complete-arg"
    return "missing-arg"


def workflow_descriptor(fp: str) -> str:
    return WORKFLOW_LABELS.get(fp, fp)


# ------------------------------------------------------------------- routing

@dataclass
class Cluster:
    id: str
    keywords: List[str] = field(default_factory=list)
    count: int = 0
    workflows: Counter = field(default_factory=Counter)
    conflict: bool = False
    examples: List[str] = field(default_factory=list)


def feature_text(task) -> str:
    """
    The text the router clusters on.

    It is the environment preamble (repeated, so it dominates) plus the
    request. The preamble is what a real agent actually sees, and it is the
    honest signal for a task family:

      * three different databases have three different preambles, so the SQL
        families separate cleanly;
      * the two BFCL families share ONE API catalogue, so their preambles are
        identical and no surface feature can tell them apart - which is
        precisely the shift the paper reports agents failing to notice, and the
        reason workflow fingerprinting is needed.
    """
    sig = ""
    ctx = getattr(task, "context", None) or {}
    if isinstance(ctx, dict):
        sig = str(ctx.get("ctx_sig", ""))
    q = getattr(task, "query", "") or ""
    return f"{sig} {sig} {sig} {q}"


class Router:
    """
    Unsupervised, label-free router over incoming tasks.

    Online clustering on TF-IDF cosine: a task joins the nearest cluster when
    it is similar enough, otherwise it starts a new one. The cluster id is the
    only scope signal the agent ever gets - it is never told the ground-truth
    task-group label.

    Each cluster also tracks the distribution of workflow fingerprints it has
    seen. A cluster that carries two incompatible fingerprints is flagged as a
    conflict, which is exactly the situation the paper reports as
    "retrieval collision": one region of the query space, two opposite
    workflows.
    """

    def __init__(self, threshold: float = 0.42) -> None:
        self.threshold = threshold
        # `index` clusters on the FEATURE text (preamble + request): this is the
        # routing signal, and it is what decides which family a task belongs to.
        self.index = SimilarityIndex()
        # `qindex` indexes the raw request alone: this is the RETRIEVAL signal,
        # the thing a memory store actually matches on. The two are kept apart
        # on purpose - conflating them is exactly the bug the paper describes.
        self.qindex = SimilarityIndex()
        self.clusters: Dict[str, Cluster] = {}
        self._order: List[str] = []
        self._counter = 0
        self._assigned: Dict[str, str] = {}

    def assign(self, task) -> Tuple[str, float, bool]:
        """
        Route a task. Returns (cluster_id, similarity, is_new).

        A new cluster is created when nothing is close enough - which is what
        keeps an unseen task family from inheriting another family's rules.
        """
        key = f"{task.group}#{getattr(task, 'id', '')}"
        if key in self._assigned:
            cid = self._assigned[key]
            return cid, 1.0, False

        feat = feature_text(task)
        self.index.add(key, feat)
        self.qindex.add(key, task.query or "")
        best_cid, best_sim = None, 0.0
        for cid in self._order:
            c = self.clusters[cid]
            if not c.examples:
                continue
            sim = max(self.index.similarity(feat, ex) for ex in c.examples)
            if sim > best_sim:
                best_cid, best_sim = cid, sim

        is_new = best_cid is None or best_sim < self.threshold
        if is_new:
            self._counter += 1
            cid = f"c{self._counter}"
            self.clusters[cid] = Cluster(id=cid)
            self._order.append(cid)
            best_sim = 1.0 if best_cid is None else best_sim
        else:
            cid = best_cid

        c = self.clusters[cid]
        c.count += 1
        if len(c.examples) < 6:
            c.examples.append(feat)
        fp = probe_workflow(task)
        c.workflows[fp] += 1
        if len(c.workflows) > 1:
            c.conflict = True
        self._assigned[key] = cid
        return cid, best_sim, is_new

    def scope_key(self, task, cluster_id: str) -> str:
        """Scope key: cluster + workflow fingerprint."""
        return f"{cluster_id}|{probe_workflow(task)}"

    def query_similarity(self, a: str, b: str) -> float:
        """Similarity between two raw requests - the retrieval signal."""
        return self.qindex.similarity(a or "", b or "")

    def clone(self) -> "Router":
        """
        A copy of the router's state.

        Evaluation must not feed the agent's own routing machinery: a hold-out
        query that helped build a cluster would leak into the next checkpoint.
        The evaluator therefore runs against a copy taken at the checkpoint.
        """
        new = Router(threshold=self.threshold)
        new.index = SimilarityIndex()
        for doc_id, text in self.index._texts.items():
            new.index.add(doc_id, text)
        new.qindex = SimilarityIndex()
        for doc_id, text in self.qindex._texts.items():
            new.qindex.add(doc_id, text)
        new.clusters = {
            cid: Cluster(id=c.id, keywords=list(c.keywords), count=c.count,
                         workflows=Counter(c.workflows), conflict=c.conflict,
                         examples=list(c.examples))
            for cid, c in self.clusters.items()
        }
        new._order = list(self._order)
        new._counter = self._counter
        new._assigned = dict(self._assigned)
        return new

    def labels(self) -> Dict[str, str]:
        """Human-readable label per cluster, from its most common content words."""
        out: Dict[str, str] = {}
        for cid, c in self.clusters.items():
            cnt: Counter = Counter()
            for ex in c.examples:
                cnt.update(tokenize(ex))
            top = [w for w, _ in cnt.most_common(4)]
            out[cid] = ", ".join(top) if top else cid
        return out

    def state(self) -> Dict[str, Any]:
        return {
            "clusters": [
                {"id": c.id, "count": c.count, "keywords": self.labels().get(c.id, ""),
                 "workflows": dict(c.workflows), "conflict": c.conflict}
                for c in (self.clusters[cid] for cid in self._order)
            ],
            "threshold": self.threshold,
        }


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
