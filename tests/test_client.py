from __future__ import annotations

import pytest
import httpx
import respx

from dexterity.client import DexterityClient, DexterityAPIError


TEST_KEY = "test-key"
BASE = "http://test.local"


def make_client(**kwargs):
    defaults = {"max_retries": 2, "retry_base_delay": 0.0}
    defaults.update(kwargs)
    return DexterityClient(api_key=TEST_KEY, base_url=BASE, **defaults)


@pytest.mark.asyncio
async def test_start_success(mock_client):
    resp = await mock_client.start(mode="dev")
    assert resp.game_id
    assert resp.truck.depth == 2.0
    assert resp.current_box.id


@pytest.mark.asyncio
async def test_place_success(mock_client):
    start = await mock_client.start(mode="dev")
    place = await mock_client.place(
        start.game_id,
        start.current_box.id,
        (0.5, 0.5, 0.5),
        (1.0, 0.0, 0.0, 0.0),
    )
    assert place.status == "ok"
    assert len(place.placed_boxes) == 1


@pytest.mark.asyncio
async def test_status_success(mock_client):
    start = await mock_client.start(mode="dev")
    status = await mock_client.status(start.game_id)
    assert status.game_id == start.game_id
    assert status.game_status == "in_progress"


@pytest.mark.asyncio
async def test_stop_success(mock_client):
    start = await mock_client.start(mode="dev")
    await mock_client.stop(start.game_id)
    status = await mock_client.status(start.game_id)
    assert status.game_status == "completed"


@pytest.mark.asyncio
async def test_my_games(mock_client):
    await mock_client.start(mode="dev")
    result = await mock_client.my_games(mode="dev")
    assert "games" in result
    assert len(result["games"]) >= 1


@pytest.mark.asyncio
async def test_invalid_api_key():
    app = __import__("tests.mock_server", fromlist=["make_app"]).make_app(require_api_key="real-key")
    transport = httpx.ASGITransport(app=app)
    async with DexterityClient(
        api_key="wrong-key",
        base_url="http://testserver",
        max_retries=0,
    ) as client:
        client._client = httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=5.0)
        with pytest.raises(DexterityAPIError) as exc_info:
            await client.start()
        assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_retry_on_500():
    """Client retries on 500 and eventually raises after max_retries."""
    with respx.mock:
        respx.post(f"{BASE}/start").mock(return_value=httpx.Response(500, json={"detail": "server error"}))
        async with make_client(max_retries=2) as client:
            with pytest.raises(DexterityAPIError) as exc_info:
                await client.start()
            assert exc_info.value.status_code == 500 or "max_retries" in exc_info.value.error_code


@pytest.mark.asyncio
async def test_no_retry_on_401():
    """Client raises immediately on 401 without retrying."""
    call_count = 0

    with respx.mock:
        def handler(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(401, json={"detail": {"error": "invalid_api_key", "message": "bad key"}})

        respx.post(f"{BASE}/start").mock(side_effect=handler)
        async with make_client(max_retries=3) as client:
            with pytest.raises(DexterityAPIError) as exc_info:
                await client.start()
            assert exc_info.value.status_code == 401
            assert call_count == 1


@pytest.mark.asyncio
async def test_api_key_in_body_for_post():
    """API key must be in JSON body for POST /start."""
    received_body = {}

    with respx.mock:
        def handler(request):
            import json
            nonlocal received_body
            received_body = json.loads(request.content)
            return httpx.Response(200, json={
                "game_id": "g1",
                "truck": {"depth": 10.0, "width": 5.0, "height": 5.0},
                "current_box": {"id": "b1", "dimensions": [1.0, 1.0, 1.0], "weight": 1.0},
                "boxes_remaining": 4,
                "mode": "dev",
            })

        respx.post(f"{BASE}/start").mock(side_effect=handler)
        async with make_client() as client:
            await client.start()

        assert received_body.get("api_key") == TEST_KEY


@pytest.mark.asyncio
async def test_api_key_as_query_param_for_my_games():
    """API key must be a query parameter for GET /my-games."""
    received_params = {}

    with respx.mock:
        def handler(request):
            nonlocal received_params
            received_params = dict(request.url.params)
            return httpx.Response(200, json={"games": []})

        respx.get(f"{BASE}/my-games").mock(side_effect=handler)
        async with make_client() as client:
            await client.my_games()

        assert received_params.get("api_key") == TEST_KEY
