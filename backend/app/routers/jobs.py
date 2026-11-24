from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobCreate(BaseModel):
    task: str = Field(max_length=40)
    params: dict = Field(default_factory=dict)


@router.get("/tasks")
def tasks():
    return [{"task": k, "module": t.module, "title": t.title, "params": t.params} for k, t in jobs.TASKS.items()]


@router.get("")
def list_jobs(limit: int = 30):
    return jobs.list_jobs(min(max(limit, 1), 200))


@router.post("")
def create(body: JobCreate):
    try:
        return jobs.submit(body.task, body.params)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/{job_id}")
def get(job_id: str, lines: int = 200):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    j["log"] = jobs.log_tail(job_id, min(max(lines, 10), 2000))
    return j


@router.post("/{job_id}/retry")
def retry(job_id: str):
    try:
        return jobs.retry(job_id)
    except KeyError as e:
        raise HTTPException(404, "Job not found") from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@router.post("/{job_id}/dismiss")
def dismiss(job_id: str):
    j = jobs.dismiss(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j


@router.post("/{job_id}/cancel")
def cancel(job_id: str):
    try:
        return jobs.cancel(job_id)
    except KeyError as e:
        raise HTTPException(404, "Job not found") from e
