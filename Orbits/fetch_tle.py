import requests

def checksum_ok(line: str) -> bool:
    '''
    Checksum for TLE pull
    Inputs: 
        line: str => single TLE line (69 characters, ch69 is checksum)
    Outputs:
        good: bool => Checksum pass or fail
    
    Functionality Overview:
        - Ensures TLE was pulled correctly and completely
        - Recreates TLE checksum and compares returned value with TLE checksum
        - iterates through characters in TLE line besides internal checksum
        - each character gets its numeric value added to checksum accumulator 's'
        - (-) sign adds 1
        - everything besides integers and (-) sign adds 0
        - Checks S % 10 & compares to character 69 (checksum) - if matched check passes
    '''
    # init accumulator for checksum
    s = 0

    # check for short lines
    if len(line) < 69:
        good = False
        return good

    # iterate cols 1-68 to check for completeness (not tle checksum in 69)
    for ch in line[:68]:
        # for integers, directly add value
        if ch.isdigit():
            s += int(ch)
        # for (-) sign, add 1
        elif ch == '-':
            s += 1
    # compare with TLE's own checksum, if equal returns True
    good = (s % 10) == int(line[68])
    return good

def fetch_tle(norad_id: int, timeout=10):
    '''
    Fetches TLE from celestrak
    Inputs:
        - norad_id: int => norad identifier of object
        - timeout=10 => timeout in seconds, defaults to 10

    Outputs:
        - name => name of object queried
        - (l1, l2) => TLE

    
    '''
    # URL to fetch from (general perturbations api endpoint from celestrak)
    url = "https://celestrak.org/NORAD/elements/gp.php"
    # params for fetcher (gives norad id, asks for tle)
    params = {"CATNR": norad_id, "FORMAT": "TLE"}
    # use requests library to get TLE
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "orbit-class/1.0"})
    # converts http errors into python requests.HTTPError to prevent returning error as tle
    r.raise_for_status()
    # strip http response into lines, drop whitespace, drop empties
    # will probably either return 3 lines w/ 1line = name or simply TLE
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    # get name, tle lines or just tle lines depending on whether first line starts with 1 or not
    if not lines[0].startswith("1 "):
        name, l1, l2 = lines[0], lines[1], lines[2]
    else:
        name, l1, l2 = None, lines[0], lines[1]
    # use checksum_ok() to verify celestrak's response
    assert checksum_ok(l1) and checksum_ok(l2), "TLE checksum failed"
    # return name, TLE line 1, TLE line 2
    return name, l1, l2
