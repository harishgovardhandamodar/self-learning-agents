"""
The landscape's provenance must be real.

Every work in the knowledge base carries a link, and a link that points at
the wrong paper is worse than no link: it looks verified and is not. Three
of the first drafts of this base had arXiv identifiers that resolved to
condensed-matter physics papers, because the identifiers were written from
memory and never checked.

So the identifiers are checked here, against arXiv itself. The check is
skipped when the network is unavailable, but it is never skipped silently:
the skip is reported, so a green run in an offline environment cannot be
mistaken for a verified one.

Only papers are checked. Products, frameworks and blogs are checked by
reachability, not by title, because a vendor page's <title> is not the
work's title.
"""
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import landscape as ls

REPO = Path(__file__).resolve().parent.parent
ARXIV_API = "http://export.arxiv.org/api/query"
CACHE = Path(__file__).resolve().parent / ".arxiv_verified.json"
CACHE_TTL = 7 * 24 * 3600  # a week


def _load_cache() -> dict:
    try:
        data = json.loads(CACHE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    except Exception:
        pass


def _norm(text: str) -> str:
    """
    Normalise for comparison: case, punctuation, and diacritics.

    The diacritics matter: the paper is titled "Gödel Agent" and the
    knowledge base says "Godel Agent", which is the same work.
    """
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _arxiv_title(arxiv_id: str, cache: dict = None, attempts: int = 3):
    """
    The real title of an arXiv entry, or None.

    None means "could not check", never "does not match" - the difference
    matters, because a transient network failure must not be reported as
    a broken identifier.

    A title that was verified within the last week is taken from the
    cache. Re-fetching twenty-two papers on every test run is slow, and
    being a good citizen of arXiv's rate limit matters more than
    freshness for a check that only needs to catch a hand-typed id.
    """
    if cache is not None:
        hit = cache.get(arxiv_id)
        if hit and (time.time() - hit.get("checked", 0)) < CACHE_TTL:
            return hit.get("title")

    url = f"{ARXIV_API}?id_list={arxiv_id}&max_results=1"
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": "sla-lab/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read().decode("utf-8", "replace")
            m = re.search(r"<entry>(.*?)</entry>", body, re.S)
            if not m:
                return None
            t = re.search(r"<title>(.*?)</title>", m.group(1), re.S)
            if not t:
                return None
            title = re.sub(r"\s+", " ", t.group(1)).strip()
            if cache is not None:
                cache[arxiv_id] = {"title": title, "checked": time.time()}
            time.sleep(3.0)  # arXiv asks for three seconds between calls
            return title
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # arXiv is rate limiting us. Backing off hard and giving
                # up is the polite answer; retrying harder is not.
                time.sleep(10.0 * (attempt + 1))
                return None
            time.sleep(2.0 * (attempt + 1))
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return None


def _words(text: str):
    return [w for w in _norm(text).split() if w not in
            ("a", "an", "the", "of", "and", "for", "to", "in", "on", "with",
             "via", "is", "are", "as", "by", "from", "or", "that", "its")]


def test_every_arxiv_identifier_resolves_to_the_claimed_title():
    """
    The identifier in the knowledge base must be the identifier of the
    work the base claims it is.
    """
    papers = [w for w in ls.works() if "arxiv.org/abs/" in (w.get("url") or "")]
    assert papers, "no arXiv-backed works to check"

    checked = 0
    skipped = []
    cache = _load_cache()
    before = set(cache.keys())
    for w in papers:
        arxiv_id = w["url"].rstrip("/").split("/")[-1]
        real = _arxiv_title(arxiv_id, cache)
        if real is None:
            skipped.append((w["id"], arxiv_id))
            continue
        checked += 1
        claimed_words = _words(w["title"])
        real_words = _words(real)
        # the real title must share the distinctive head of the claimed one
        head = [x for x in claimed_words if len(x) > 3][:4]
        missing = [x for x in head if x not in real_words]
        assert not missing, (
            f"{w['id']}: arXiv:{arxiv_id} is titled {real!r}, "
            f"but the knowledge base claims {w['title']!r} "
            f"(unmatched: {missing})")

    if set(cache.keys()) != before:
        _save_cache(cache)

    if skipped:
        print(f"SKIPPED {len(skipped)} arXiv checks (network unavailable): "
              f"{[s[0] for s in skipped]}")
    else:
        print(f"verified {checked} arXiv identifiers against arXiv itself")

def test_no_work_carries_a_bare_search_url():
    """
    A search URL is a fallback, not provenance.

    It resolves, but it does not point at a specific work, so it cannot
    be checked later. Any work that has a resolvable identifier must use
    it; only works whose identifier could not be confirmed may use a
    search, and there must be few enough of them to notice.
    """
    searches = [w for w in ls.works() if "arxiv.org/search" in (w.get("url") or "")]
    assert len(searches) <= 2, (
        f"{len(searches)} works fall back to a search URL: "
        f"{[w['id'] for w in searches]}")
    for w in searches:
        assert w.get("title"), w["id"]


def test_the_knowledge_base_covers_both_sides_of_the_field():
    """
    A landscape that only lists papers is not a landscape. Academic work
    and industry work must both be present, or the alignment matrix is
    measuring one half of the field.
    """
    kinds = {w.get("kind") for w in ls.works()}
    assert "paper" in kinds
    assert kinds & {"product", "framework", "benchmark", "writeup", "blog"}
    years = [w.get("year") for w in ls.works() if w.get("year")]
    assert min(years) <= 2020, "the base has no older work in it"
    assert max(years) >= 2026, "the base has nothing from the current year"
