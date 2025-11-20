import os
import time
import json
import math
import random
import requests
from typing import Tuple, List, Iterable, Optional, Dict

# ===========================
# Configuration
# ===========================
ST_BASE = "https://www.space-track.org"
ST_LOGIN_URL = f"{ST_BASE}/ajaxauth/login"
ST_LOGOUT_URL = f"{ST_BASE}/ajaxauth/logout"
ST_QUERY_BASE = f"{ST_BASE}/basicspacedata/query"

USER_AGENT = "Astrodynamics-Toolbox/SpaceTrack (contact: you@example.com)"
POLITE_DELAY_S = 1.1                # ~1 req/sec
DEFAULT_TTL_S = 12 * 3600           # 12 hours
CACHE_DIR_TLE = "cache/tle_spacetrack"

REQUEST_TIMEOUT = (5, 30)           # (connect, read) seconds
MAX_RETRIES = 4
BACKOFF_BASE = 1.5                  # exponential backoff multiplier

# ===========================
# Exceptions
# ===========================
class SpaceTrackAuthError(RuntimeError):
    pass

class SpaceTrackRateLimitError(RuntimeError):
    pass

class SpaceTrackNoTLEError(RuntimeError):
    """Raised when no TLE could be retrieved for the given NORAD ID."""
    def __init__(self, norad_id: int, message: str, omm_kvn: Optional[str] = None):
        super().__init__(message)
        self.norad_id = norad_id
        self.omm_kvn = omm_kvn  # raw text if we grabbed OMM (KVN) as a hint

# ===========================
# Utilities: cache & checksum
# ===========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _cache_read(path: str, ttl_s: float) -> Optional[List[str]]:
    if ttl_s <= 0:
        return None
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age <= ttl_s:
            with open(path, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
    return None

def _cache_write(path: str, lines: Iterable[str]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def checksum_ok(line: str) -> bool:
    """TLE line checksum across cols 1–68 (col 69 is checksum digit)."""
    if len(line) < 69 or not line[0] in ("1", "2"):
        return False
    s = 0
    for ch in line[:68]:
        if ch.isdigit():
            s += int(ch)
        elif ch == '-':
            s += 1
    try:
        return (s % 10) == int(line[68])
    except ValueError:
        return False

def _normalize_tle_triplet(lines: List[str]) -> Tuple[Optional[str], str, str]:
    """
    Normalize to (name, L1, L2). Space-Track often returns:
      0 NAME
      1 ....
      2 ....
    but may also return just the 2 TLE lines.
    """
    if not lines:
        raise ValueError("Empty response")

    # Find consecutive L1/L2 pair
    # Try with and without a name line
    if lines[0].startswith("1 "):
        if len(lines) < 2:
            raise ValueError(f"Unexpected TLE content: {lines}")
        name, l1, l2 = None, lines[0], lines[1]
    else:
        # Could be "0 NAME" or a bare name; accept both
        if len(lines) < 3:
            # Some endpoints may return *only* 1/2 without 0 NAME; handle below
            # Try to detect L1/L2 anywhere in the list
            idx1 = next((i for i, ln in enumerate(lines) if ln.startswith("1 ")), None)
            if idx1 is not None and idx1 + 1 < len(lines) and lines[idx1 + 1].startswith("2 "):
                name = lines[idx1 - 1] if idx1 >= 1 and not lines[idx1 - 1].startswith(("1 ", "2 ")) else None
                l1, l2 = lines[idx1], lines[idx1 + 1]
            else:
                raise ValueError(f"Unexpected TLE content: {lines}")
        else:
            name_candidate, l1, l2 = lines[0], lines[1], lines[2]
            name = name_candidate
            # Space-Track usually prefixes with "0 "
            if name is not None and name.startswith("0 "):
                name = name[2:].strip()

    assert l1.startswith("1 ") and l2.startswith("2 "), "Malformed TLE lines"
    assert checksum_ok(l1) and checksum_ok(l2), "TLE checksum failed"
    return name, l1, l2

# ===========================
# HTTP helpers (session + retry)
# ===========================
def _sleep_with_jitter(base: float, attempt: int) -> None:
    # Exponential backoff with small jitter
    delay = (base ** attempt) + random.uniform(0, 0.3)
    time.sleep(delay)

def _st_login_session(identity: str, password: str, timeout=REQUEST_TIMEOUT) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    time.sleep(POLITE_DELAY_S)
    resp = s.post(ST_LOGIN_URL, data={"identity": identity, "password": password}, timeout=timeout)
    resp.raise_for_status()
    # Common login-fail signatures: JSON with {"Login":"Failed"} or HTML message
    try:
        j = resp.json()
        if isinstance(j, dict) and j.get("Login") == "Failed":
            raise SpaceTrackAuthError("Space-Track login failed: check username/password.")
    except ValueError:
        pass  # Not JSON; assume cookie set via HTML redirect
    return s

def _st_get(session: requests.Session, path_suffix: str, timeout=REQUEST_TIMEOUT) -> str:
    """
    GET with retry/backoff. Returns response text.
    Raises on repeated 429/5xx.
    """
    url = f"{ST_QUERY_BASE}/{path_suffix.lstrip('/')}"
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            time.sleep(POLITE_DELAY_S)
            r = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code == 429:
                # Rate limited
                last_exc = SpaceTrackRateLimitError("HTTP 429 Too Many Requests")
                _sleep_with_jitter(BACKOFF_BASE, attempt + 1)
                continue
            r.raise_for_status()
            text = r.text.strip()

            # If cookie expired or bounced to login
            if text.startswith("{") and '"Login":"Failed"' in text:
                raise SpaceTrackAuthError("Space-Track session expired; re-login required.")

            # Some errors come back as JSON error payloads
            if text.startswith("{") and '"error"' in text.lower():
                # Let caller handle content; but surface a helpful message
                raise RuntimeError(f"Space-Track error payload: {text[:200]}")

            return text
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            # Backoff for transient 5xx or timeouts
            code = getattr(e.response, "status_code", None)
            if code and 400 <= code < 500 and code != 429:
                # Likely permanent (e.g., 404 or bad query); don't spin too long
                if attempt >= 1:
                    break
            _sleep_with_jitter(BACKOFF_BASE, attempt + 1)
    # Out of retries
    if isinstance(last_exc, Exception):
        raise last_exc
    raise RuntimeError("Unknown Space-Track request failure")

# ===========================
# Core query functions
# ===========================
def _query_tle_latest(session: requests.Session, norad_id: int) -> Optional[List[str]]:
    """
    Try GP (newest elset) in TLE format. Returns lines or None if not found.
    """
    # Limit to newest record explicitly to keep responses small
    suffix = f"class/gp/NORAD_CAT_ID/{int(norad_id)}/orderby/EPOCH%20DESC/limit/1/format/tle"
    text = _st_get(session, suffix)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    # Heuristic: if the response includes a JSON error blob or plain HTML, reject
    if lines and not any(ln.startswith("1 ") for ln in lines) and not any(ln.startswith("0 ") for ln in lines):
        return None
    # Some objects may have name + 2 lines, some only 2 lines
    if any(ln.startswith("1 ") for ln in lines) and any(ln.startswith("2 ") for ln in lines):
        return lines
    return None

def _query_tle_latest_from_history(session: requests.Session, norad_id: int) -> Optional[List[str]]:
    """
    Try GP_History (latest historical elset) in TLE format. Returns lines or None.
    """
    suffix = f"class/gp_history/NORAD_CAT_ID/{int(norad_id)}/orderby/EPOCH%20DESC/limit/1/format/tle"
    text = _st_get(session, suffix)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    if any(ln.startswith("1 ") for ln in lines) and any(ln.startswith("2 ") for ln in lines):
        return lines
    return None

def _query_omm_latest_kvn(session: requests.Session, norad_id: int) -> Optional[str]:
    """
    As a last resort, fetch OMM (KVN). We can't turn this into a true TLE,
    but including it in the error helps the caller decide what to do.
    """
    suffix = f"class/gp/NORAD_CAT_ID/{int(norad_id)}/orderby/EPOCH%20DESC/limit/1/format/kvn"
    text = _st_get(session, suffix)
    return text if text else None


def fetch_tle(
    norad_id: int,
    identity: str = 'dstannar@calpoly.edu',
    password: str ='sxiAzkbs8M-jPQg',
    ttl_s: float = DEFAULT_TTL_S,
    cache_dir: str = CACHE_DIR_TLE,
) -> Tuple[Optional[str], str, str]:
    """
    Fetch the newest TLE for a NORAD ID from Space-Track with caching and retries.

    Returns:
        (name, L1, L2) where 'name' may be None if no name line was returned.

    Raises:
        SpaceTrackAuthError       -> bad credentials or expired session
        SpaceTrackRateLimitError  -> hit 429 repeatedly
        SpaceTrackNoTLEError      -> no public TLE available (includes OMM snippet if possible)
        ValueError/AssertionError -> malformed or checksum-failed TLE content
    """
    cache_path = os.path.join(cache_dir, f"{int(norad_id)}.tle")

    # Cache
    lines = _cache_read(cache_path, ttl_s)
    if lines is not None:
        return _normalize_tle_triplet(lines)

    # Live pull
    sess = _st_login_session(identity, password, timeout=REQUEST_TIMEOUT)

    # 1) Try GP (latest)
    lines = _query_tle_latest(sess, norad_id)
    if lines is None:
        # 2) Try GP_History (most recent historical)
        lines = _query_tle_latest_from_history(sess, norad_id)

    if lines is None:
        # 3) Optional helpful diagnostics: try OMM (KVN)
        omm = _query_omm_latest_kvn(sess, norad_id)
        snippet = None
        if omm:
            # Keep only the first ~40 lines to avoid dumping huge payloads
            snippet = "\n".join(omm.splitlines()[:40])
        raise SpaceTrackNoTLEError(
            norad_id,
            f"No public TLE available for NORAD {norad_id} via GP/GP_History.",
            omm_kvn=snippet
        )

    # Normalize, verify checksum, and write cache
    triplet = _normalize_tle_triplet(lines)
    _cache_write(cache_path, [ln for ln in lines])
    return triplet

def fetch_tle_spacetrack_bulk(
    norad_ids: Iterable[int],
    identity: str,
    password: str,
    ttl_s: float = DEFAULT_TTL_S,
    cache_dir: str = CACHE_DIR_TLE,
) -> Dict[int, Tuple[Optional[str], str, str]]:
    """
    Convenience bulk helper. Iterates IDs, honoring cache and rate limits.
    Returns a dict of id -> (name, L1, L2). Raises on the first hard error.
    """
    results: Dict[int, Tuple[Optional[str], str, str]] = {}
    sess = _st_login_session(identity, password, timeout=REQUEST_TIMEOUT)

    for nid in norad_ids:
        cache_path = os.path.join(cache_dir, f"{int(nid)}.tle")
        lines = _cache_read(cache_path, ttl_s)
        if lines is None:
            # Live pull per ID (keeps throttling)
            lines = _query_tle_latest(sess, nid)
            if lines is None:
                lines = _query_tle_latest_from_history(sess, nid)
            if lines is None:
                # Don’t attempt OMM in bulk; let caller handle missing IDs
                raise SpaceTrackNoTLEError(nid, f"No public TLE available for NORAD {nid}.")
            _cache_write(cache_path, [ln for ln in lines])

        results[nid] = _normalize_tle_triplet(lines)

    return results

def spacetrack_logout(identity: str, password: str) -> None:
    """logout of spacetrak"""
    try:
        sess = _st_login_session(identity, password, timeout=REQUEST_TIMEOUT)
        time.sleep(POLITE_DELAY_S)
        sess.get(ST_LOGOUT_URL, timeout=REQUEST_TIMEOUT)
    except Exception:
        pass
