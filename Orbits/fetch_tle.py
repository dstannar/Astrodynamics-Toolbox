import os
import time
import json
import random
import csv
import requests
from typing import Tuple, Iterable, Optional, Dict, List

'''
I will admit that chatgpt was used in creation of this file as my web scraping skills need some work. i will fix it soon!

'''

# ===========================
# Configuration
# ===========================
ST_BASE = "https://www.space-track.org"
ST_LOGIN_URL = f"{ST_BASE}/ajaxauth/login"
ST_LOGOUT_URL = f"{ST_BASE}/ajaxauth/logout"
ST_QUERY_BASE = f"{ST_BASE}/basicspacedata/query"

# Identify yourself politely (add a real contact email).
USER_AGENT = os.getenv("SPACETRACK_USER_AGENT", "dstannar@calpoly.edu")

# Cache settings
CACHE_DIR_TLE = "cache/tle_spacetrack"
GLOBAL_3LE_CACHE = os.path.join(CACHE_DIR_TLE, "gp_now3_3le.txt")
GLOBAL_3LE_INDEX = os.path.join(CACHE_DIR_TLE, "gp_now3_index.json")
FALLBACK_QUOTA_FILE = os.path.join(CACHE_DIR_TLE, "fallback_quota.json")

# TTLs
PER_ID_TTL_S = 12 * 3600
GLOBAL_TTL_S = 12 * 3600

# HTTP behavior
REQUEST_TIMEOUT = (5, 30)   # (connect, read) seconds
POLITE_DELAY_S = 1.1        # ~1 req/sec
MAX_RETRIES = 3             # gentle backoff for transient errors
BACKOFF_BASE = 1.7

# satcat CSV locations (first existing wins)
_SATCAT_CANDIDATES = [
    os.path.join("Orbits", "satcat.csv"),
    os.getenv("SATCAT_CSV") or "",
    os.path.join(".", "Orbits", "satcat.csv"),
    os.path.join(os.getcwd(), "Orbits", "satcat.csv"),
    os.path.join(os.getcwd(), "satcat.csv"),
    "/mnt/data/satcat.csv",
    "satcat.csv",
]

# In-memory satcat cache
_SATCAT_ROWS: Optional[Dict[int, Dict[str, str]]] = None
_SATCAT_MTIME: Optional[float] = None
_SATCAT_PATH: Optional[str] = None

# ===========================
# Exceptions
# ===========================
class SpaceTrackAuthError(RuntimeError):
    pass

class SpaceTrackRateLimitError(RuntimeError):
    pass

class SpaceTrackNoTLEError(RuntimeError):
    def __init__(self, norad_id: int, message: str):
        super().__init__(message)
        self.norad_id = norad_id

# ===========================
# Cache utilities
# ===========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _cache_read_lines(path: str, ttl_s: float) -> Optional[List[str]]:
    if ttl_s <= 0 or not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > ttl_s:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def _cache_write_lines(path: str, lines: Iterable[str]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")

def _cache_read_json(path: str, ttl_s: Optional[float] = None) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    if ttl_s is not None:
        age = time.time() - os.path.getmtime(path)
        if age > ttl_s:
            return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _cache_write_json(path: str, obj: dict) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ===========================
# TLE helpers
# ===========================
def checksum_ok(line: str) -> bool:
    """TLE line checksum per standard rule."""
    if len(line) < 69 or line[0] not in ("1", "2"):
        return False
    s = 0
    for ch in line[:68]:
        if ch.isdigit():
            s += int(ch)
        elif ch == "-":
            s += 1
    try:
        return (s % 10) == int(line[68])
    except ValueError:
        return False

def _norad_from_l1(line1: str) -> int:
    # TLE line 1 columns 3–7 are the 5-digit catalog number
    return int(line1[2:7])

def _iter_triplets_from_3le_text(text: str):
    """Yield (name, L1, L2). Name may be None (absent)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        name = None
        if lines[i].startswith("0 "):
            name = lines[i][2:].strip()
            i += 1
        if i + 1 >= len(lines):
            break
        l1, l2 = lines[i], lines[i + 1]
        i += 2
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue
        if not (checksum_ok(l1) and checksum_ok(l2)):
            continue
        yield name, l1, l2

def _index_global_3le(text: str) -> Dict[int, Tuple[Optional[str], str, str]]:
    idx: Dict[int, Tuple[Optional[str], str, str]] = {}
    for name, l1, l2 in _iter_triplets_from_3le_text(text):
        try:
            nid = _norad_from_l1(l1)
        except Exception:
            continue
        # GP endpoint returns the newest elset per object already
        idx[nid] = (name, l1, l2)
    return idx

def _normalize_triplet(lines: List[str]) -> Tuple[Optional[str], str, str]:
    """Normalize either [L1, L2] or [0 name, L1, L2] to (name, L1, L2)."""
    if not lines:
        raise ValueError("empty TLE lines")
    if lines[0].startswith("1 "):
        name = None
        l1, l2 = lines[0], lines[1]
    else:
        name = lines[0][2:].strip() if lines[0].startswith("0 ") else lines[0]
        l1, l2 = lines[1], lines[2]
    return name, l1, l2

# ===========================
# satcat.csv loader (OBJECT_NAME only; this sheet has no TLE columns)
# ===========================
def _satcat_path() -> Optional[str]:
    global _SATCAT_PATH
    if _SATCAT_PATH and os.path.exists(_SATCAT_PATH):
        return _SATCAT_PATH
    for cand in _SATCAT_CANDIDATES:
        if not cand:
            continue
        if os.path.exists(cand):
            _SATCAT_PATH = cand
            return cand
    return None

def _load_satcat_rows() -> None:
    """Load satcat rows into a dict keyed by NORAD_CAT_ID. No TLEs in this sheet."""
    global _SATCAT_ROWS, _SATCAT_MTIME
    p = _satcat_path()
    if not p:
        _SATCAT_ROWS = {}
        _SATCAT_MTIME = None
        return
    mtime = os.path.getmtime(p)
    if _SATCAT_ROWS is not None and _SATCAT_MTIME == mtime:
        return
    rows: Dict[int, Dict[str, str]] = {}
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                nid = int(str(row.get("NORAD_CAT_ID", "")).strip())
            except Exception:
                continue
            rows[nid] = row
    _SATCAT_ROWS = rows
    _SATCAT_MTIME = mtime

def _satcat_name(norad_id: int) -> Optional[str]:
    _load_satcat_rows()
    if not _SATCAT_ROWS:
        return None
    row = _SATCAT_ROWS.get(int(norad_id))
    if not row:
        return None
    name = str(row.get("OBJECT_NAME", "")).strip()
    return name or None

# ===========================
# HTTP helpers (session + single GET)
# ===========================
def _sleep_with_jitter(base: float, attempt: int) -> None:
    time.sleep((base ** attempt) + random.uniform(0, 0.25))

def _st_login_session(identity: str, password: str, timeout=REQUEST_TIMEOUT) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    time.sleep(POLITE_DELAY_S)
    resp = s.post(ST_LOGIN_URL, data={"identity": identity, "password": password}, timeout=timeout)
    resp.raise_for_status()
    try:
        j = resp.json()
        if isinstance(j, dict) and j.get("Login") == "Failed":
            raise SpaceTrackAuthError("Space-Track login failed: check credentials.")
    except ValueError:
        pass
    return s

def _st_one_gp_call(session: requests.Session, suffix: str, timeout=REQUEST_TIMEOUT) -> str:
    """
    Perform exactly ONE /class/gp/ call (with polite delay and backoff).
    """
    url = f"{ST_QUERY_BASE}/{suffix.lstrip('/')}"
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            time.sleep(POLITE_DELAY_S)
            r = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code == 429:
                last_exc = SpaceTrackRateLimitError("HTTP 429 Too Many Requests")
                _sleep_with_jitter(BACKOFF_BASE, attempt + 1)
                continue
            r.raise_for_status()
            text = r.text.strip()
            if text.startswith("{") and '"Login":"Failed"' in text:
                raise SpaceTrackAuthError("Space-Track session expired.")
            return text
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code and 400 <= code < 500 and code != 429:
                if attempt >= 1:
                    break
            _sleep_with_jitter(BACKOFF_BASE, attempt + 1)
    if isinstance(last_exc, Exception):
        raise last_exc
    raise RuntimeError("Unknown Space-Track request failure")

# ===========================
# Global (12h) pull – ONE gp call when stale
# ===========================
def _refresh_global_now3(session: requests.Session) -> Dict[int, Tuple[Optional[str], str, str]]:
    """
    Recommended endpoint: newest propagable element sets for all on-orbit objects.
    /class/gp/decay_date/null-val/epoch/>now-3/format/3le
    """
    suffix = "class/gp/decay_date/null-val/epoch/%3Enow-3/format/3le"
    text = _st_one_gp_call(session, suffix)  # **one** /class/gp call
    _cache_write_lines(GLOBAL_3LE_CACHE, text.splitlines())
    idx = _index_global_3le(text)
    _cache_write_json(GLOBAL_3LE_INDEX, {str(k): v for k, v in idx.items()})
    return idx

def _load_global_index(ttl_s: float = GLOBAL_TTL_S) -> Optional[Dict[int, Tuple[Optional[str], str, str]]]:
    j = _cache_read_json(GLOBAL_3LE_INDEX, ttl_s)
    if j is not None:
        return {int(k): tuple(v) for k, v in j.items()}
    raw = _cache_read_lines(GLOBAL_3LE_CACHE, ttl_s)
    if raw is not None:
        return _index_global_3le("\n".join(raw))
    return None

# ===========================
# Fallback quota (max 10 objects/hour)
# ===========================
def _fallback_quota_status(now: Optional[float] = None) -> dict:
    """
    Returns the current quota record, resetting if window expired.
    Tracks objects (not HTTP calls). Max 10 objects per wall-clock hour.
    """
    now = now or time.time()
    rec = _cache_read_json(FALLBACK_QUOTA_FILE, ttl_s=None) or {}
    window_start = rec.get("window_start", 0.0)
    used = int(rec.get("used", 0))
    # If more than an hour has passed, reset
    if now - float(window_start or 0.0) >= 3600:
        rec = {"window_start": now, "used": 0}
        _cache_write_json(FALLBACK_QUOTA_FILE, rec)
        return rec
    # Ensure structure persisted
    if "window_start" not in rec:
        rec["window_start"] = now
    if "used" not in rec:
        rec["used"] = 0
    _cache_write_json(FALLBACK_QUOTA_FILE, rec)
    return rec

def _fallback_quota_take(n: int) -> int:
    """
    Consume up to n from the hourly quota. Returns the allowed amount (0..n).
    """
    rec = _fallback_quota_status()
    used = int(rec.get("used", 0))
    remaining = max(0, 10 - used)
    allow = min(n, remaining)
    if allow > 0:
        rec["used"] = used + allow
        _cache_write_json(FALLBACK_QUOTA_FILE, rec)
    return allow

# ===========================
# Fallback fetch (combined, single call)
# ===========================
def _fallback_query_gp_latest_combined(session: requests.Session, norad_ids: Iterable[int]) -> Dict[int, Tuple[Optional[str], str, str]]:
    """
    Combined fallback: query gp for up to 10 IDs (comma-delimited) **once**,
    then return newest TLE per ID. We rely on GP returning the newest elset.
    """
    ids = [int(n) for n in dict.fromkeys(int(n) for n in norad_ids)]  # de-dup, preserve order
    if not ids:
        return {}
    id_list = ",".join(str(n) for n in ids)
    suffix = f"class/gp/NORAD_CAT_ID/{id_list}/format/3le"
    text = _st_one_gp_call(session, suffix)  # one HTTP GET
    return _index_global_3le(text)

# ===========================
# Public API
# ===========================
def fetch_tle(
    norad_id: int,
    # Keep your hard-coded credentials in the defaults
    identity: str = 'dstannar@calpoly.edu',
    password: str = 'sxiAzkbs8M-jPQg',
    cache_dir: str = CACHE_DIR_TLE,
) -> Tuple[Optional[str], str, str]:
    """
    Returns newest TLE (name, L1, L2) for NORAD id, with minimal API usage.

    Lookup order:
      0) satcat.csv (OBJECT_NAME only; this sheet has no TLEs) – name adornment only.
      1) Per-ID cache (TTL 12h).
      2) Global cache (TTL 12h) built from a single /class/gp "now-3" refresh.
      3) If still missing, and within quota, do **one** combined fallback gp call
         for this ID (counts toward max 10 objects/hour).
    """
    # 1) Per-ID cache first (satcat only carries name; no TLE)
    per_id_cache = os.path.join(cache_dir, f"{int(norad_id)}.tle")
    lines = _cache_read_lines(per_id_cache, PER_ID_TTL_S)
    if lines:
        return _normalize_triplet(lines)

    # 2) Global cache
    idx = _load_global_index(GLOBAL_TTL_S)
    if idx and int(norad_id) in idx:
        name, l1, l2 = idx[int(norad_id)]
        # Write per-ID cache
        if name:
            _cache_write_lines(per_id_cache, [f"0 {name}", l1, l2])
        else:
            _cache_write_lines(per_id_cache, [l1, l2])
        return name, l1, l2

    # 3) If global cache is stale/missing, refresh once
    if idx is None:
        sess = _st_login_session(identity, password, timeout=REQUEST_TIMEOUT)
        idx = _refresh_global_now3(sess)  # one global /class/gp call
        if int(norad_id) in idx:
            name, l1, l2 = idx[int(norad_id)]
            if name:
                _cache_write_lines(per_id_cache, [f"0 {name}", l1, l2])
            else:
                _cache_write_lines(per_id_cache, [l1, l2])
            return name, l1, l2

    # 4) Fallback path: allow exceptions (up to 10 objects/hour), combined query
    allow = _fallback_quota_take(1)
    if allow <= 0:
        raise SpaceTrackNoTLEError(int(norad_id),
            f"No TLE for NORAD {norad_id} in newest GP set and fallback quota exhausted (10/hour).")

    sess = _st_login_session(identity, password, timeout=REQUEST_TIMEOUT)
    partial = _fallback_query_gp_latest_combined(sess, [int(norad_id)])
    if int(norad_id) not in partial:
        raise SpaceTrackNoTLEError(int(norad_id),
            f"No TLE for NORAD {norad_id} available via fallback gp query.")
    name, l1, l2 = partial[int(norad_id)]
    if not name:
        # adorn with satcat OBJECT_NAME if present
        satname = _satcat_name(int(norad_id))
        if satname:
            name = satname
    # write per-ID cache
    if name:
        _cache_write_lines(per_id_cache, [f"0 {name}", l1, l2])
    else:
        _cache_write_lines(per_id_cache, [l1, l2])
    return name, l1, l2

def fetch_tle_bulk(
    norad_ids: Iterable[int],
    identity: str = 'dstannar@calpoly.edu',
    password: str = 'sxiAzkbs8M-jPQg',
    cache_dir: str = CACHE_DIR_TLE,
) -> Dict[int, Tuple[Optional[str], str, str]]:
    """
    Bulk helper with identical policy:
      - Serve from per-ID cache (12h) or global cache (12h).
      - If global cache is stale/missing, do **one** global /class/gp "now-3" refresh.
      - For any IDs still missing, allow up to **10 objects/hour** via a **single**
        combined fallback gp call and fill results from that.
    """
    wanted = [int(n) for n in norad_ids]
    results: Dict[int, Tuple[Optional[str], str, str]] = {}

    # 1) Per-ID cache
    missing: List[int] = []
    for nid in wanted:
        per_id_cache = os.path.join(cache_dir, f"{nid}.tle")
        lines = _cache_read_lines(per_id_cache, PER_ID_TTL_S)
        if lines:
            results[nid] = _normalize_triplet(lines)
        else:
            missing.append(nid)

    if not missing:
        return results

    # 2) Global cache
    idx = _load_global_index(GLOBAL_TTL_S)
    if idx:
        for nid in list(missing):
            if nid in idx:
                results[nid] = idx[nid]
                per_id_cache = os.path.join(cache_dir, f"{nid}.tle")
                name, l1, l2 = results[nid]
                if name:
                    _cache_write_lines(per_id_cache, [f"0 {name}", l1, l2])
                else:
                    _cache_write_lines(per_id_cache, [l1, l2])
                missing.remove(nid)
    else:
        # 2b) If no global cache, refresh once
        sess = _st_login_session(identity, password, timeout=REQUEST_TIMEOUT)
        idx = _refresh_global_now3(sess)  # one global /class/gp call
        for nid in list(missing):
            if nid in idx:
                results[nid] = idx[nid]
                per_id_cache = os.path.join(cache_dir, f"{nid}.tle")
                name, l1, l2 = results[nid]
                if name:
                    _cache_write_lines(per_id_cache, [f"0 {name}", l1, l2])
                else:
                    _cache_write_lines(per_id_cache, [l1, l2])
                missing.remove(nid)

    if not missing:
        return results

    # 3) Fallback combined query for remaining IDs (quota: 10 objects/hour)
    allow = _fallback_quota_take(len(missing))
    if allow <= 0:
        # nothing left in quota; return what we have and let caller handle misses
        return results

    ask = missing[:allow]
    sess = _st_login_session(identity, password, timeout=REQUEST_TIMEOUT)
    partial = _fallback_query_gp_latest_combined(sess, ask)  # one HTTP GET for up to 'allow' objects
    for nid in ask:
        if nid in partial:
            results[nid] = partial[nid]
            per_id_cache = os.path.join(cache_dir, f"{nid}.tle")
            name, l1, l2 = results[nid]
            # adorn with satcat name if needed
            if not name:
                satname = _satcat_name(nid)
                if satname:
                    name = satname
                    results[nid] = (name, l1, l2)
            if name:
                _cache_write_lines(per_id_cache, [f"0 {name}", l1, l2])
            else:
                _cache_write_lines(per_id_cache, [l1, l2])
    return results

def spacetrack_logout(identity: str, password: str) -> None:
    """Best-effort logout (not required; included for completeness)."""
    try:
        sess = _st_login_session(identity, password, timeout=REQUEST_TIMEOUT)
        time.sleep(POLITE_DELAY_S)
        sess.get(ST_LOGOUT_URL, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
    except Exception:
        pass
