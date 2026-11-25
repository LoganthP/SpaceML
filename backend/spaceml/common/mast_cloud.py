"""Public STScI archive objects, independent of the MAST portal search service."""
from urllib.parse import quote, urlencode
from urllib.request import urlopen
from xml.etree import ElementTree

ROOT = "https://stpubdata.s3.amazonaws.com"


def keys(prefix: str, *, limit: int = 1000) -> list[str]:
    params = urlencode({"list-type": 2, "prefix": prefix, "max-keys": limit})
    with urlopen(f"{ROOT}/?{params}", timeout=12) as response:
        body = response.read(2_000_001)
    if len(body) > 2_000_000:
        raise ValueError("STScI cloud listing exceeded its size limit")
    data = ElementTree.fromstring(body)
    return [key.text for key in data.findall("./{*}Contents/{*}Key") if key.text and key.text.startswith(prefix)]


def url(key: str) -> str:
    return f"{ROOT}/{quote(key, safe='/')}"
