"""Latest public solar-wind plasma observation from NOAA SWPC."""
import json
import math
import threading
from datetime import datetime, timezone
from urllib.request import urlopen

SOURCE = "https://services.swpc.noaa.gov/products/geospace/propagated-solar-wind-1-hour.json"
LOCK = threading.Lock()
CACHE: dict = {}
FETCHED = 0.0


def latest() -> dict:
    global CACHE, FETCHED
    with LOCK:
        now = datetime.now(timezone.utc)
        if now.timestamp() - FETCHED >= 60:
            FETCHED = now.timestamp()
            try:
                with urlopen(SOURCE, timeout=6) as response:
                    raw = response.read(250_001)
                if len(raw) > 250_000:
                    raise ValueError("Unexpected NOAA response size")
                header, *rows = json.loads(raw)
                indices = tuple(header.index(name) for name in ("time_tag", "speed", "density", "temperature"))
                for row in reversed(rows):
                    try:
                        stamp, speed, density, temp = (row[index] for index in indices)
                        observed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                        if observed.tzinfo is None:
                            observed = observed.replace(tzinfo=timezone.utc)
                        values = [float(value) for value in (speed, density, temp)]
                        if all(math.isfinite(value) and value >= 0 for value in values):
                            CACHE = dict(zip(("speed", "density", "temperature"), values),
                                         observed_at=observed.isoformat(), fetched_at=now.isoformat())
                            break
                    except (TypeError, ValueError, AttributeError, IndexError):
                        continue
            except (OSError, ValueError, TypeError, IndexError, AttributeError):
                pass
        if not CACHE:
            return {"status": "unavailable", "source": SOURCE}
        observed = datetime.fromisoformat(CACHE["observed_at"])
        fresh = 0 <= (now - observed).total_seconds() <= 15 * 60
        return {**CACHE, "status": "live" if fresh else "stale", "source": SOURCE}
