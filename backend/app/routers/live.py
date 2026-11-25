"""In-app live view of what Hubble and Webb are observing (Space Telescope Live data, cached for offline use)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from spaceml.common import solarwind, stlive
from spaceml.common.fetch import SourceUnavailable

router = APIRouter(prefix="/api/live", tags=["live"])


@router.get("/solar-wind")
def solar_wind():
    return solarwind.latest()


@router.get("/{telescope}")
def live(telescope: str, refresh: bool = False):
    if telescope not in stlive.TELESCOPES:
        raise HTTPException(404, "unknown telescope")
    try:
        return stlive.current(telescope, refresh)
    except SourceUnavailable as e:
        raise HTTPException(503, f"Live observing data is not available right now ({e.reason}). It appears here "
                                 "automatically once the app can reach Space Telescope Live.") from e


@router.get("/{telescope}/sky/{obs_id}")
def sky(telescope: str, obs_id: str, fov: float = Query(0.5)):
    try:
        path = stlive.sky(telescope, obs_id, fov)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'")) from e
    except SourceUnavailable as e:
        raise HTTPException(503, f"Sky image not available right now ({stlive.friendly(e.reason)}).") from e
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})
