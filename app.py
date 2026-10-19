"""FastAPI service. Run: uvicorn app:app --host 0.0.0.0 --port 7860 (HF Spaces default port).

POST /v1/skin-analysis   {"image_url": "<Cloudinary URL>", "real_age": 24, "masks": true}
    -> YouCam-shaped reply (output[]: wrinkle, firmness, age_spot, radiance, all, skin_age) for TintKin.
POST /analyze/url, /analyze/file   -> native result (scores, raw measures, quality, ages).
Set SKINAI_API_KEY to require an `x-api-key` header on every analyze call.
"""
from __future__ import annotations

import asyncio
import os
import secrets
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from skinai.analyze import analyze
from skinai.contract import to_youcam

app = FastAPI(title="Tintkin skin analysis")
MAX_BYTES = 10 * 1024 * 1024
API_KEY = os.environ.get("SKINAI_API_KEY", "")
CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "")  # same value as TintKin's NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME


def require_key(x_api_key: str = Header(default="")) -> None:
    if API_KEY and not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "invalid api key")


class UrlBody(BaseModel):
    image_url: str  # Cloudinary URL
    real_age: float | None = Field(default=None, ge=5, le=110)
    masks: bool = True


@app.on_event("startup")
def warm() -> None:
    """Load the models once so the first request isn't slow."""
    from skinai import age, face
    face._get_landmarker()
    face._get_segmenter()
    age._session()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _check_cloudinary_url(url: str) -> None:
    """Same rule as TintKin's lib/utils/cloudinary.js: https, host res.cloudinary.com, path /<cloud>/image/...
    The cloud name is only enforced when CLOUDINARY_CLOUD_NAME is set."""
    u = urlparse(url)
    ok = u.scheme == "https" and u.hostname == "res.cloudinary.com" and u.port in (None, 443) and not u.username
    if ok:
        ok = u.path.startswith(f"/{CLOUD_NAME}/image/") if CLOUD_NAME else u.path.startswith("/")
    if not ok:
        raise HTTPException(400, "image_url must be a Cloudinary URL")


async def _fetch(url: str) -> bytes:
    _check_cloudinary_url(url)
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as c:
            async with c.stream("GET", url) as r:
                if r.status_code != 200:
                    raise HTTPException(400, f"could not fetch image ({r.status_code})")
                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) > MAX_BYTES:
                        raise HTTPException(413, "image too large")
                return bytes(buf)
    except httpx.HTTPError as e:
        raise HTTPException(400, f"could not fetch image ({type(e).__name__})") from e


def _run(data: bytes, real_age: float | None, with_maps: bool = False) -> dict:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large")
    try:
        return analyze(data, real_age=real_age, with_maps=with_maps)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


_ONE_AT_A_TIME = asyncio.Semaphore(1)  # the MediaPipe models are shared singletons: scan one image at a time


async def _run_async(data: bytes, real_age: float | None, with_maps: bool = False) -> dict:
    """Analysis is CPU-bound: run it in a worker thread so /health and downloads stay responsive."""
    async with _ONE_AT_A_TIME:
        return await run_in_threadpool(_run, data, real_age, with_maps)


@app.post("/v1/skin-analysis", dependencies=[Depends(require_key)])
async def skin_analysis(body: UrlBody) -> dict:
    res = await _run_async(await _fetch(body.image_url), body.real_age, with_maps=body.masks)
    return to_youcam(res, with_masks=body.masks)


@app.post("/analyze/url", dependencies=[Depends(require_key)])
async def analyze_url(body: UrlBody) -> dict:
    return await _run_async(await _fetch(body.image_url), body.real_age)


@app.post("/analyze/file", dependencies=[Depends(require_key)])
async def analyze_file(file: UploadFile = File(...), real_age: float | None = Form(default=None)) -> dict:
    return await _run_async(await file.read(), real_age)
