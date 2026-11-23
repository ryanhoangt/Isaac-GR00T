"""Dedicated stateful API; the existing reactive GR00T server is unchanged."""

import logging

from fastapi import FastAPI, HTTPException, Request, Response
import json_numpy
from starlette.concurrency import run_in_threadpool

from examples.RoboMME.memory_client import PROTOCOL
from examples.RoboMME.memory_policy import EpisodeConflict


def create_memory_app(controller, *, server_metadata=None):
    app = FastAPI()

    async def decode(request):
        try:
            return json_numpy.loads(await request.body())
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(400, "Invalid JSON/NumPy request") from error

    def encode(value):
        return Response(content=json_numpy.dumps(value), media_type="application/json")

    @app.get("/health")
    def health():
        return {
            **(server_metadata or {}),
            "status": "ok",
            "protocol": PROTOCOL,
            "stateful": True,
            "checkpoint": controller.checkpoint,
        }

    @app.post("/reset")
    async def reset(request: Request):
        payload = await decode(request)
        if not isinstance(payload, dict) or payload:
            raise HTTPException(400, "Reset body must be an empty object")
        return encode(await run_in_threadpool(controller.reset))

    @app.post("/act")
    async def act(request: Request):
        payload = await decode(request)
        try:
            result = await run_in_threadpool(controller.infer, payload)
        except EpisodeConflict as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            logging.exception("Memory inference failed")
            raise HTTPException(500, "Memory inference failed; reset is required") from error
        return encode(result)

    return app
