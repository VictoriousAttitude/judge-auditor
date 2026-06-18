from __future__ import annotations

import httpx
import pytest
import respx

from judge_auditor.config import JudgeConfig, JudgeMode
from judge_auditor.runner.backends.openai import OpenAIBackend

URL = "https://api.openai.com/v1/chat/completions"


@pytest.fixture
def config() -> JudgeConfig:
    return JudgeConfig(
        model="gpt-4o",
        prompt_template="Question: {prompt}\nResponse: {response}\nScore 1-10.",
        mode=JudgeMode.SCALAR,
    )


def _completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        },
    )


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        OpenAIBackend()


@respx.mock
async def test_successful_call_parses_text_and_usage(config):
    respx.post(URL).mock(return_value=_completion('{"score": 8}'))
    backend = OpenAIBackend(api_key="test")
    resp = await backend.call([{"role": "user", "content": "hi"}], config)
    assert resp.text == '{"score": 8}'
    assert resp.prompt_tokens == 11
    assert resp.completion_tokens == 3
    assert resp.latency_s is not None
    await backend.aclose()


@respx.mock
async def test_retries_on_429_then_succeeds(config, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    route = respx.post(URL)
    route.side_effect = [
        httpx.Response(429, headers={"retry-after": "0"}),
        _completion('{"score": 5}'),
    ]
    backend = OpenAIBackend(api_key="test", max_retries=3)
    resp = await backend.call([{"role": "user", "content": "hi"}], config)
    assert resp.text == '{"score": 5}'
    assert route.call_count == 2
    await backend.aclose()


@respx.mock
async def test_includes_response_format_when_set(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return _completion('{"score": 7}')

    respx.post(URL).mock(side_effect=handler)
    config = JudgeConfig(
        model="gpt-4o",
        prompt_template="Score {prompt} {response}",
        mode=JudgeMode.SCALAR,
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    backend = OpenAIBackend(api_key="test")
    await backend.call([{"role": "user", "content": "hi"}], config)
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.0
    await backend.aclose()


async def _noop_sleep(_seconds: float) -> None:
    return None


@respx.mock
async def test_persistent_retryable_status_exhausts_then_raises(config, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    route = respx.post(URL)
    route.mock(return_value=httpx.Response(500))
    backend = OpenAIBackend(api_key="test", max_retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        await backend.call([{"role": "user", "content": "hi"}], config)
    # max_retries backoffs, then a final attempt that raises: max_retries + 1 calls.
    assert route.call_count == 3
    await backend.aclose()


@respx.mock
async def test_non_retryable_4xx_raises_immediately(config):
    route = respx.post(URL)
    route.mock(return_value=httpx.Response(401))
    backend = OpenAIBackend(api_key="test", max_retries=5)
    with pytest.raises(httpx.HTTPStatusError):
        await backend.call([{"role": "user", "content": "hi"}], config)
    assert route.call_count == 1  # 401 is not retryable: no second attempt
    await backend.aclose()


@respx.mock
async def test_network_errors_raise_runtimeerror_after_all_attempts(config, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    route = respx.post(URL)
    route.mock(side_effect=httpx.ConnectError("boom"))
    backend = OpenAIBackend(api_key="test", max_retries=2)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        await backend.call([{"role": "user", "content": "hi"}], config)
    assert route.call_count == 3
    await backend.aclose()


@respx.mock
async def test_retry_after_header_is_honored_over_jitter(config, monkeypatch):
    slept: list[float] = []

    async def _record_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("asyncio.sleep", _record_sleep)
    route = respx.post(URL)
    route.side_effect = [
        httpx.Response(429, headers={"retry-after": "2.5"}),
        _completion('{"score": 9}'),
    ]
    backend = OpenAIBackend(api_key="test", max_retries=3)
    resp = await backend.call([{"role": "user", "content": "hi"}], config)
    assert resp.text == '{"score": 9}'
    assert slept == [2.5]  # exact server-specified wait, not the jittered backoff
    await backend.aclose()
