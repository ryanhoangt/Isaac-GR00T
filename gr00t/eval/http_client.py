"""Stateless NumPy HTTP transport for the existing GR00T /act endpoint."""

from collections.abc import Mapping
import math
from urllib.parse import urlsplit


class HTTPInferenceClient:
    def __init__(self, server_url="http://127.0.0.1:5555", *, timeout=60.0):
        parsed = urlsplit(server_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("server_url must be an HTTP(S) base URL without a query or fragment")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        import json_numpy
        import requests

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._json_numpy = json_numpy
        self._session = requests.Session()

    def get_action(self, observation):
        # Do not retry: a failed response may already have advanced sampling RNG.
        response = self._session.post(
            f"{self.server_url}/act",
            data=self._json_numpy.dumps({"observation": observation}),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        prediction = self._json_numpy.loads(response.content)
        if not isinstance(prediction, Mapping):
            raise ValueError("GR00T /act response must be a mapping of physical action fields")
        return prediction

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
