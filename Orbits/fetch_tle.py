import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

# ===========================
# Configuration (Space-Track bulk GP + cache + rate limiting)
# ===========================
ST_BASE = "https://www.space-track.org"
ST_LOGIN_URL = f"{ST_BASE}/ajaxauth/login"
ST_QUERY_BASE = f"{ST_BASE}/basicspacedata/query"

USER_AGENT = os.getenv("SPACETRACK_USER_AGENT", "dstannar@calpoly.edu")

# Keep the externally-facing defaults unchanged.
CACHE_DIR_TLE = "cache/tle_spacetrack"

# Space-Track guideline: do not query GP more than once per hour.
# We keep the cache slightly longer than 1 hour so most calls are cache hits.
_CACHE_HOURS = 2.0

_MAX_REQUESTS_PER_MINUTE = 30
_MAX_REQUESTS_PER_HOUR = 300

# Persisted API attempt log (shared across scripts in this repo).
# Put it in the repo root (../ from Orbits/).
_RATE_STATE_FILE = Path(__file__).resolve().parents[1] / ".spacetrack_rate_state.json"

# satcat CSV locations (first existing wins) for optional name adornment
_SATCAT_CANDIDATES = [
    os.path.join("Orbits", "satcat.csv"),
    os.getenv("SATCAT_CSV") or "",
    os.path.join(".", "Orbits", "satcat.csv"),
    os.path.join(os.getcwd(), "Orbits", "satcat.csv"),
    os.path.join(os.getcwd(), "satcat.csv"),
    "/mnt/data/satcat.csv",
    "satcat.csv",
]

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
# Rate state helpers (persistent API attempt log)
# ===========================
def _load_rate_state() -> List[datetime]:
    if not _RATE_STATE_FILE.exists():
        return []

    try:
        raw = json.loads(_RATE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

    timestamps: List[datetime] = []

    if isinstance(raw, dict):
        log = raw.get("request_log_utc")
        if isinstance(log, list):
            for ts in log:
                if not isinstance(ts, str):
                    continue
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except Exception:
                    continue

        if not timestamps:
            ts = raw.get("last_gp_query_utc")
            if isinstance(ts, str):
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except Exception:
                    pass

    elif isinstance(raw, list):
        for ts in raw:
            if not isinstance(ts, str):
                continue
            try:
                timestamps.append(datetime.fromisoformat(ts))
            except Exception:
                continue

    return timestamps


def _save_rate_state(request_log: List[datetime]) -> None:
    payload = {"request_log_utc": [dt.isoformat() for dt in request_log if isinstance(dt, datetime)]}
    try:
        _RATE_STATE_FILE.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to save rate state to {_RATE_STATE_FILE}: {e}") from e


def _prune_old_requests(request_log: List[datetime], now: datetime) -> List[datetime]:
    cutoff = now - timedelta(hours=1)
    return [ts for ts in request_log if ts >= cutoff]

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
# Space-Track request helpers
# ===========================
def _get_credentials(identity: Optional[str], password: Optional[str]) -> Tuple[str, str]:
    """
    Resolve Space-Track credentials.

    Priority:
    - explicit args (identity/password) if both are non-empty
    - environment variables SPACETRACK_USERNAME/SPACETRACK_PASSWORD
    """
    if identity and password:
        return identity, password

    username = os.getenv("SPACETRACK_USERNAME")
    pw = os.getenv("SPACETRACK_PASSWORD")
    if not username or not pw:
        raise SpaceTrackAuthError(
            "Space-Track credentials not found. Provide identity/password or set "
            "SPACETRACK_USERNAME and SPACETRACK_PASSWORD environment variables."
        )
    return username, pw


def _build_gp_query_url() -> str:
    """
    Global on-orbit, recent-EPOCH GP dataset in JSON format.

    We include emptyresult/show so empty queries don't return a blank page/body.
    """
    # on-orbit only: decay_date/null-val
    # recent TLEs: epoch/>now-10
    # JSON so additional metadata is available and parsing is robust.
    return (
        f"{ST_QUERY_BASE}"
        "/class/gp"
        "/decay_date/null-val"
        "/epoch/%3Enow-10"
        "/orderby/EPOCH%20desc"
        "/format/json"
        "/emptyresult/show"
    )


def _http_get_spacetrack_json(url: str, username: str, password: str, timeout: float = 30.0) -> str:
    with requests.Session() as s:
        s.headers.update({"User-Agent": USER_AGENT})
        login = s.post(
            ST_LOGIN_URL,
            data={"identity": username, "password": password},
            timeout=timeout,
        )
        login.raise_for_status()

        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text

# ===========================
# Bulk GP cache (global dataset)
# ===========================
def _cache_file(cache_dir: Optional[str]) -> Path:
    root = Path(cache_dir) if cache_dir is not None else (Path.home() / ".spacetrack-cache")
    root.mkdir(parents=True, exist_ok=True)
    return root / "tles_gp_full.json"


def _load_cached_gp(cache_path: Path, cache_hours: float) -> Optional[List[dict]]:
    if not cache_path.exists():
        return None

    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    age_hours = (now - mtime).total_seconds() / 3600.0
    if age_hours > cache_hours:
        return None

    text = cache_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse cached GP JSON from {cache_path}: {e}") from e

    if not isinstance(data, list):
        raise RuntimeError(
            f"Unexpected GP cache format in {cache_path}: expected list, got {type(data).__name__}"
        )

    return data


def _save_cached_gp(cache_path: Path, records: List[dict]) -> None:
    try:
        cache_path.write_text(json.dumps(records), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to save cached GP JSON to {cache_path}: {e}") from e


def _index_gp_records_by_norad(records: List[dict]) -> Dict[int, dict]:
    """
    Build a NORAD_CAT_ID -> record index.

    Records are assumed to be ordered by newest EPOCH first (we query with orderby/EPOCH desc),
    so we keep the first occurrence per NORAD.
    """
    idx: Dict[int, dict] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        nid = rec.get("NORAD_CAT_ID")
        try:
            nid_i = int(nid)
        except Exception:
            continue
        if nid_i not in idx:
            idx[nid_i] = rec
    return idx

# ===========================
# Public API
# ===========================
def fetch_tle(
    norad_id: int,
    identity: Optional[str] = None,
    password: Optional[str] = None,
    cache_dir: str = CACHE_DIR_TLE,
) -> Tuple[Optional[str], str, str]:
    """
    Return newest TLE (name, L1, L2) for a NORAD id.

    Internals mimic a "single bulk GP request + global cache + persistent rate-state" strategy:
    - Uses a global cached GP JSON dataset for fast local lookup.
    - If the cache is stale/missing, performs exactly one bulk GP request and caches it.
    - Enforces Space-Track API limits via a shared, persisted request log file.
    """
    nid = int(norad_id)
    cache_path = _cache_file(cache_dir)

    cached_records = _load_cached_gp(cache_path, _CACHE_HOURS)
    if cached_records is None:
        now = datetime.now(timezone.utc)
        request_log = _prune_old_requests(_load_rate_state(), now)

        one_minute_ago = now - timedelta(minutes=1)
        recent_minute = [ts for ts in request_log if ts >= one_minute_ago]
        if len(recent_minute) >= _MAX_REQUESTS_PER_MINUTE:
            raise SpaceTrackRateLimitError(
                "Space-Track request suppressed (30/min limit). "
                f"{len(recent_minute)} requests logged in the last 60 seconds."
            )

        if len(request_log) >= _MAX_REQUESTS_PER_HOUR:
            raise SpaceTrackRateLimitError(
                "Space-Track request suppressed (300/hour limit). "
                f"{len(request_log)} requests logged in the last hour."
            )

        if request_log:
            last_gp = max(request_log)
            if now - last_gp < timedelta(hours=1):
                raise SpaceTrackRateLimitError(
                    "Space-Track GP query suppressed (1/hour guideline). "
                    f"Last GP query was at {last_gp.isoformat()} UTC."
                )

        request_log.append(now)
        _save_rate_state(request_log)

        username, pw = _get_credentials(identity, password)
        url = _build_gp_query_url()
        response_text = _http_get_spacetrack_json(url, username, pw)
        stripped = response_text.strip()
        if not stripped:
            raise RuntimeError(f"Space-Track GP query returned an empty response. URL: {url}")

        try:
            records = json.loads(stripped)
        except Exception as e:
            raise RuntimeError("Failed to parse Space-Track GP JSON response.") from e

        if not isinstance(records, list) or not records:
            raise SpaceTrackNoTLEError(nid, f"Space-Track GP query returned no records. URL: {url}")

        _save_cached_gp(cache_path, records)
        cached_records = records

    idx = _index_gp_records_by_norad(cached_records)
    rec = idx.get(nid)
    if not rec:
        raise SpaceTrackNoTLEError(nid, f"No TLE found in cached Space-Track GP dataset for NORAD {nid}.")

    l1 = rec.get("TLE_LINE1")
    l2 = rec.get("TLE_LINE2")
    if not (isinstance(l1, str) and isinstance(l2, str) and l1.strip() and l2.strip()):
        raise SpaceTrackNoTLEError(nid, f"Cached GP record for NORAD {nid} is missing TLE lines.")

    name = rec.get("OBJECT_NAME")
    if not isinstance(name, str) or not name.strip():
        name = _satcat_name(nid)

    return (name.strip() if isinstance(name, str) else None), l1.strip(), l2.strip()

def fetch_tle_bulk(
    norad_ids: Iterable[int],
    identity: Optional[str] = None,
    password: Optional[str] = None,
    cache_dir: str = CACHE_DIR_TLE,
) -> Dict[int, Tuple[Optional[str], str, str]]:
    """
    Bulk helper for fetching newest TLEs for many NORAD ids.

    Uses the same global GP cache and rate-state enforcement as `fetch_tle()`.
    """
    ids = [int(n) for n in norad_ids]
    results: Dict[int, Tuple[Optional[str], str, str]] = {}
    if not ids:
        return results

    cache_path = _cache_file(cache_dir)
    cached_records = _load_cached_gp(cache_path, _CACHE_HOURS)

    if cached_records is None:
        # Delegate to `fetch_tle()` to ensure rate-state is enforced consistently,
        # but avoid N API calls by performing the single bulk refresh here.
        now = datetime.now(timezone.utc)
        request_log = _prune_old_requests(_load_rate_state(), now)

        one_minute_ago = now - timedelta(minutes=1)
        recent_minute = [ts for ts in request_log if ts >= one_minute_ago]
        if len(recent_minute) >= _MAX_REQUESTS_PER_MINUTE:
            raise SpaceTrackRateLimitError(
                "Space-Track request suppressed (30/min limit). "
                f"{len(recent_minute)} requests logged in the last 60 seconds."
            )

        if len(request_log) >= _MAX_REQUESTS_PER_HOUR:
            raise SpaceTrackRateLimitError(
                "Space-Track request suppressed (300/hour limit). "
                f"{len(request_log)} requests logged in the last hour."
            )

        if request_log:
            last_gp = max(request_log)
            if now - last_gp < timedelta(hours=1):
                raise SpaceTrackRateLimitError(
                    "Space-Track GP query suppressed (1/hour guideline). "
                    f"Last GP query was at {last_gp.isoformat()} UTC."
                )

        request_log.append(now)
        _save_rate_state(request_log)

        username, pw = _get_credentials(identity, password)
        url = _build_gp_query_url()
        response_text = _http_get_spacetrack_json(url, username, pw)
        stripped = response_text.strip()
        if not stripped:
            raise RuntimeError(f"Space-Track GP query returned an empty response. URL: {url}")

        try:
            records = json.loads(stripped)
        except Exception as e:
            raise RuntimeError("Failed to parse Space-Track GP JSON response.") from e

        if not isinstance(records, list) or not records:
            raise RuntimeError(f"Space-Track GP query returned no records. URL: {url}")

        _save_cached_gp(cache_path, records)
        cached_records = records

    idx = _index_gp_records_by_norad(cached_records)
    for nid in ids:
        rec = idx.get(nid)
        if not rec:
            continue
        l1 = rec.get("TLE_LINE1")
        l2 = rec.get("TLE_LINE2")
        if not (isinstance(l1, str) and isinstance(l2, str) and l1.strip() and l2.strip()):
            continue
        name = rec.get("OBJECT_NAME")
        if not isinstance(name, str) or not name.strip():
            name = _satcat_name(nid)
        results[nid] = ((name.strip() if isinstance(name, str) else None), l1.strip(), l2.strip())

    return results
