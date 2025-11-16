import os
import time
import requests
from typing import Tuple, List, Iterable, Optional

# configuration
CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"
USER_AGENT = "Astrodynamics-Toolbox (contact: dstannar@calpoly.edu)"
CACHE_TLE_DIR = "cache/tle"
CACHE_GROUP_DIR = "cache/groups"
DEFAULT_TTL_S = 12 * 3600        # 12 hours in secs
POLITE_DELAY_S = 1.1             # 1 req/sec throttle


def checksum_ok(line: str) -> bool:
    '''
    Checksum for TLE pull
    Inputs:
        line: str => single TLE line (69 characters, ch69 is checksum)
    Outputs:
        good: bool => Checksum pass or fail

    Functionality Overview:
        - Ensures TLE was pulled correctly and completely
        - Recreates TLE checksum and compares with returned checksum
        - Iterates through characters in columns 1-68 (excludes checksum)
        - Digits add their numeric value to accumulator 's'
        - '-' adds 1; everything else adds 0
        - Pass = (s % 10) equals character 69 (checksum)
    '''
    if len(line) < 69:
        return False
    s = 0
    for ch in line[:68]:
        if ch.isdigit():
            s += int(ch)
        elif ch == '-':
            s += 1
    return (s % 10) == int(line[68])


def _ensure_dir(path: str) -> None:
    '''Creates directory path if it doesn’t exist'''
    os.makedirs(path, exist_ok=True)


def _cache_read(path: str, ttl_s: float) -> Optional[List[str]]:
    '''
    Read cached file
    Inputs:
        path: str => cache file path
        ttl_s: float => time-to-live (seconds); if <= 0, always refresh (no cache hit)
    Outputs:
        lines: list[str] or None => file lines if fresh, else None

    Functionality:
        - Checks file mtime age against ttl_s
        - Returns stripped, non-empty lines on cache hit
    '''
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if ttl_s > 0 and age <= ttl_s:
            with open(path, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
    return None


def _cache_write(path: str, lines: Iterable[str]) -> None:
    '''Writes lines to cache (simple atomicity is fine for this use).'''
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _get(url: str, params: dict, timeout=(5, 30)) -> List[str]:
    '''
    Polite GET with throttle.
    Inputs:
        url: str => endpoint
        params: dict => query params
        timeout: (connect_s, read_s) => requests timeout tuple
    Outputs:
        lines: list[str] => stripped, non-empty response lines

    Functionality Overview:
        - Sleeps POLITE_DELAY_S before request (rate-limit friendly)
        - Sets a clear User-Agent
        - Raises HTTPError on non-2xx
    '''
    time.sleep(POLITE_DELAY_S)
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return [ln.strip() for ln in r.text.splitlines() if ln.strip()]


def _normalize_tle_triplet(lines: List[str]) -> Tuple[Optional[str], str, str]:
    '''
    Normalize raw response lines to (name, L1, L2).
    Inputs:
        lines: list[str] => 2 or 3 lines typical of CelesTrak gp.php
    Outputs:
        (name, L1, L2): Tuple[str|None, str, str]

    Functionality:
        - If first line starts with "1 ", treat as [L1, L2] (no name)
        - Else treat as [name, L1, L2]
        - Verifies checksum on both TLE lines
    '''
    if not lines:
        raise ValueError("Empty TLE response")

    if lines[0].startswith("1 "):
        if len(lines) < 2:
            raise ValueError(f"Unexpected TLE content: {lines}")
        name, l1, l2 = None, lines[0], lines[1]
    else:
        if len(lines) < 3:
            raise ValueError(f"Unexpected TLE content: {lines}")
        name, l1, l2 = lines[0], lines[1], lines[2]

    assert checksum_ok(l1) and checksum_ok(l2), "TLE checksum failed"
    return name, l1, l2


def fetch_tle(norad_id: int,
                     ttl_s: float = DEFAULT_TTL_S,
                     cache_dir: str = CACHE_TLE_DIR,
                     timeout=(5, 30)) -> Tuple[Optional[str], str, str]:
    '''
    Cached per-ID TLE fetch (CATNR query)
    Inputs:
        - norad_id: int => NORAD catalog number
        - ttl_s: float => cache Time-To-Live (seconds) — default 12 hours
        - cache_dir: str => where to store per-ID cache files
        - timeout: (connect_s, read_s) => requests timeouts
    Outputs:
        - (name, L1, L2) => normalized TLE triplet

    Functionality Overview:
        - Checks cache/tle/<id>.tle and returns if fresh (file age <= ttl_s)
        - Otherwise queries CelesTrak gp.php with CATNR
        - Throttles to ~1 req/sec and sets User-Agent
        - Verifies checksum, writes cache, returns triplet
    '''
    path = os.path.join(cache_dir, f"{int(norad_id)}.tle")

    lines = _cache_read(path, ttl_s)
    if lines is None:
        lines = _get(CELESTRAK_URL, {"CATNR": str(int(norad_id)), "FORMAT": "TLE"}, timeout=timeout)
        _cache_write(path, lines)

    return _normalize_tle_triplet(lines)


def fetch_group_cached(group: str,
                       ttl_s: float = DEFAULT_TTL_S,
                       fmt: str = "TLE",
                       cache_dir: str = CACHE_GROUP_DIR,
                       timeout=(5, 30)) -> List[str]:
    '''
    Cached GROUP fetch
    Inputs:
        - group: str => CelesTrak group name (e.g., 'geo', 'active', 'gps-ops')
        - ttl_s: float => cache TTL seconds (default 12 hours)
        - fmt: str => 'TLE' (default), 'CSV', or 'JSON'
        - cache_dir: str => where to store group cache files
        - timeout: (connect_s, read_s) => requests timeouts
    Outputs:
        - lines: list[str] => raw response lines (for TLE/CSV/JSON)

    Functionality Overview:
        - Checks cache/groups/<group>.<ext>
        - If stale/missing, GETs GROUP data once and caches
        - Use parse_group_tles(...) to split into TLE triplets when fmt='TLE'
    '''
    ext = fmt.lower()
    path = os.path.join(cache_dir, f"{group}.{ext}")

    lines = _cache_read(path, ttl_s)
    if lines is None:
        lines = _get(CELESTRAK_URL, {"GROUP": group, "FORMAT": fmt}, timeout=timeout)
        _cache_write(path, lines)
    return lines


def parse_group_tles(lines: List[str]) -> List[Tuple[Optional[str], str, str]]:
    '''
    Parse a GROUP TLE dump into (name, L1, L2) records
    Inputs:
        - lines: list[str] => raw lines from fetch_group_cached(..., fmt='TLE')
    Outputs:
        - records: list[tuple] => list of (name|None, L1, L2)

    Functionality Overview:
        - Group TLEs come as repeating blocks:
            [name, L1, L2] OR sometimes just [L1, L2] (no name line)
        - Walks the list and emits normalized triplets
        - Verifies checksum on each L1/L2
    '''
    recs: List[Tuple[Optional[str], str, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].startswith("1 "):
            if i + 1 >= n:
                raise ValueError("Unexpected end of TLE list (missing L2)")
            name, l1, l2 = None, lines[i], lines[i + 1]
            i += 2
        else:
            if i + 2 >= n:
                raise ValueError("Unexpected end of TLE list (missing L1/L2)")
            name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
            i += 3

        assert checksum_ok(l1) and checksum_ok(l2), "TLE checksum failed in group"
        recs.append((name, l1, l2))
    return recs