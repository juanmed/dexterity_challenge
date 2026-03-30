from __future__ import annotations

import pytest
import httpx

from .mock_server import make_app
from dexterity.client import DexterityClient


TEST_API_KEY = "test-key"
N_BOXES = 5


@pytest.fixture
def mock_app():
    return make_app(require_api_key=TEST_API_KEY, n_boxes=N_BOXES)


@pytest.fixture
async def mock_client(mock_app):
    transport = httpx.ASGITransport(app=mock_app)
    async with DexterityClient(
        api_key=TEST_API_KEY,
        base_url="http://testserver",
        max_retries=2,
        retry_base_delay=0.01,
    ) as client:
        client._client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=httpx.Timeout(5.0),
        )
        yield client
