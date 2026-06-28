import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse


LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = int(os.getenv("VOICEBOX_PROXY_PORT", "17493"))

APPDATA = Path(os.getenv("APPDATA", str(Path.home())))
CONFIG_PATH = Path(
    os.getenv(
        "VOICEBOX_PROXY_CONFIG",
        str(APPDATA / "sh.voicebox.app" / "remote_proxy.json"),
    )
)

DEFAULT_CONFIG = {
    "upstream_url": "https://REPLACE-ME.trycloudflare.com",
    "health": {
        "status": "healthy",
        "model_loaded": False,
        "gpu_available": False,
    },
}

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

app = FastAPI(
    title="Voicebox Local Remote Proxy",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def ensure_config() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            json.dumps(DEFAULT_CONFIG, indent=2),
            encoding="utf-8",
        )


def load_config() -> dict[str, Any]:
    ensure_config()

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    merged = dict(DEFAULT_CONFIG)
    merged.update(data)

    # Environment variable always wins.
    env_url = os.getenv("VOICEBOX_UPSTREAM_URL")
    if env_url:
        merged["upstream_url"] = env_url

    return merged


def save_config(data: dict[str, Any]) -> None:
    ensure_config()
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_upstream_url() -> str:
    upstream = str(load_config().get("upstream_url", "")).strip()

    if not upstream or "REPLACE-ME" in upstream:
        raise RuntimeError(
            f"No valid upstream_url configured. Edit: {CONFIG_PATH}"
        )

    return upstream.rstrip("/")


def filtered_request_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}

    for key, value in request.headers.items():
        if key.lower() not in HOP_BY_HOP_HEADERS:
            headers[key] = value

    return headers


def filtered_response_headers(headers: httpx.Headers) -> dict[str, str]:
    clean: dict[str, str] = {}

    for key, value in headers.items():
        if key.lower() not in HOP_BY_HOP_HEADERS:
            clean[key] = value

    return clean


@app.get("/health")
async def health() -> JSONResponse:
    """
    Local health shim.

    Do not forward this to Cloudflare/Colab. The desktop app polls /health
    during startup. If Cloudflare returns 429, startup fails.
    """
    cfg = load_config()
    payload = cfg.get("health", DEFAULT_CONFIG["health"])

    return JSONResponse(
        {
            "status": str(payload.get("status", "healthy")),
            "model_loaded": bool(payload.get("model_loaded", False)),
            "gpu_available": bool(payload.get("gpu_available", False)),
        },
        status_code=200,
    )


@app.get("/_proxy/status")
async def proxy_status() -> JSONResponse:
    cfg = load_config()

    return JSONResponse(
        {
            "proxy": "running",
            "listen": f"http://{LOCAL_HOST}:{LOCAL_PORT}",
            "config_path": str(CONFIG_PATH),
            "upstream_url": cfg.get("upstream_url"),
            "health_is_local": True,
        }
    )


@app.put("/_proxy/upstream")
async def update_upstream(request: Request) -> JSONResponse:
    """
    Update the Cloudflare/remote URL without editing the Python file.

    Example body:
    {
      "upstream_url": "https://new-url.trycloudflare.com"
    }
    """
    body = await request.json()
    new_url = str(body.get("upstream_url", "")).strip().rstrip("/")

    if not new_url.startswith(("http://", "https://")):
        return JSONResponse(
            {"error": "upstream_url must start with http:// or https://"},
            status_code=400,
        )

    cfg = load_config()
    cfg["upstream_url"] = new_url
    save_config(cfg)

    return JSONResponse(
        {
            "updated": True,
            "upstream_url": new_url,
            "config_path": str(CONFIG_PATH),
        }
    )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_all(path: str, request: Request) -> Response:
    upstream = get_upstream_url()

    query = request.url.query
    target_url = f"{upstream}/{path}"
    if query:
        target_url = f"{target_url}?{query}"

    body = await request.body()
    headers = filtered_request_headers(request)

    client = httpx.AsyncClient(
        timeout=None,
        follow_redirects=False,
    )

    try:
        upstream_request = client.build_request(
            request.method,
            target_url,
            headers=headers,
            content=body,
        )

        upstream_response = await client.send(
            upstream_request,
            stream=True,
        )

    except Exception as exc:
        await client.aclose()
        return JSONResponse(
            {
                "error": "upstream_proxy_error",
                "detail": str(exc),
                "upstream_url": upstream,
            },
            status_code=502,
        )

    response_headers = filtered_response_headers(upstream_response.headers)
    media_type = upstream_response.headers.get("content-type")

    async def stream_body():
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=media_type,
    )


def main() -> None:
    ensure_config()

    print("=" * 72)
    print("Voicebox local remote proxy")
    print(f"Listening:    http://{LOCAL_HOST}:{LOCAL_PORT}")
    print(f"Config file:  {CONFIG_PATH}")
    print(f"Upstream:     {load_config().get('upstream_url')}")
    print("Health shim:  http://127.0.0.1:17493/health")
    print("=" * 72)

    uvicorn.run(
        app,
        host=LOCAL_HOST,
        port=LOCAL_PORT,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
