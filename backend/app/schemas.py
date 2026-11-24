from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TelemetryInput(BaseModel):
    chan_id: str = Field(min_length=1, max_length=16, pattern=r"^[A-Z]-\d{1,2}$")


class ExoplanetInput(BaseModel):
    kepid: int | None = Field(default=None, gt=0, lt=100_000_000)
    tce_plnt_num: int = Field(default=1, ge=1, le=20)
    dataset_id: str | None = Field(default=None, max_length=32, pattern=r"^[a-f0-9]+$")
    target_id: str | None = Field(default=None, max_length=16, pattern=r"^t\d+$")
    period: float | None = Field(default=None, gt=0.05, lt=2000)
    t0: float | None = None
    duration_hours: float | None = Field(default=None, gt=0.1, lt=72)

    @model_validator(mode="after")
    def one_source(self):
        if bool(self.kepid) == bool(self.dataset_id):
            raise ValueError("Provide either kepid or dataset_id")
        if self.dataset_id and not self.target_id:
            raise ValueError("target_id is required with dataset_id")
        given = [self.period is not None, self.t0 is not None, self.duration_hours is not None]
        if any(given) and not all(given):
            raise ValueError("To set the ephemeris yourself give period, t0 and duration_hours together")
        return self


class MarsInput(BaseModel):
    sample_id: str | None = Field(default=None, max_length=64, pattern=r"^[\w\-]+$")
    upload_id: str | None = Field(default=None, max_length=64, pattern=r"^[a-f0-9]+$")
    raw_image_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Z0-9_]{20,60}$")

    @model_validator(mode="after")
    def one_source(self):
        if sum(map(bool, (self.sample_id, self.upload_id, self.raw_image_id))) != 1:
            raise ValueError("Provide exactly one of sample_id, upload_id or raw_image_id")
        return self


class SpaceWeatherInput(BaseModel):
    event_id: str | None = Field(default=None, max_length=16, pattern=r"^[\w\-]+$")
    start: str | None = Field(default=None, max_length=40)
    end: str | None = Field(default=None, max_length=40)
    horizons: list[int] | None = Field(default=None, max_length=8)

    @model_validator(mode="after")
    def window(self):
        if not self.event_id and not (self.start and self.end):
            raise ValueError("Provide event_id or start and end")
        return self


class RunCreate(BaseModel):
    module: Literal["telemetry", "exoplanet", "mars", "workbench", "spaceweather", "missions"]
    input: dict
    model_version: str | None = Field(default=None, max_length=40, pattern=r"^[\w.\-]+$")


class MissionInput(BaseModel):
    mission: str = Field(pattern=r"^[a-z0-9-]{2,20}$")
    action: Literal["evaluate", "score"] = "score"
    item: str | None = Field(default=None, max_length=120)
    column: str | None = Field(default=None, max_length=120)


class WorkbenchInput(BaseModel):
    dataset_id: str = Field(pattern=r"^[a-f0-9]{12}$")
    analysis: str = Field(max_length=40)
    params: dict = Field(default_factory=dict)


class ReviewIn(BaseModel):
    verdict: Literal["confirmed", "dismissed", "unsure", "corrected"]
    note: str = Field(default="", max_length=2000)
