from __future__ import annotations

import json

import httpx
import pytest
import respx

from judge_auditor.config import JudgeConfig, JudgeMode
from judge_auditor.runner.backends.anthropic import AnthropicBackend

URL = "https://api.anthropic.com/v1/messages"


@pytest.fixture
def config() -> JudgeConfig:
    return JudgeConfig(
        model="claude-sonnet-4-6",
        prompt_template="Question: {prompt}\nResponse: {response}\nScore 1-10.",
        mode=JudgeMode.SCALAR,
        system_prompt="You are a strict grader.",
    )


def _message(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 12, "output_tokens": 4},
        },
    )


async def _noop_sleep(_seconds: float) -> None:
    return None


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        AnthropicBackend()


@respx.mock
async def test_successful_call_parses_text_and_usage(config):
    respx.post(URL).mock(return_value=_message('{"score": 8}'))
    backend = AnthropicBackend(api_key="test")
    resp = await backend.call([{"role": "user", "content": "hi"}], config)
    assert resp.text == '{"score": 8}'
    assert resp.prompt_tokens == 12
    assert resp.completion_tokens == 4
    assert resp.latency_s is not None
    await backend.aclose()


@respx.mock
async def test_system_message_is_hoisted_to_top_level(config):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return _message('{"score": 5}')

    respx.post(URL).mock(side_effect=handler)
    backend = AnthropicBackend(api_key="test")
    await backend.call(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "judge this"},
        ],
        config,
    )
    assert captured["system"] == "be terse"
    assert captured["messages"] == [{"role": "user", "content": "judge this"}]
    assert captured["max_tokens"] == config.max_tokens
    assert "anthropic-version" not in captured  # header, not body
    await backend.aclose()


@respx.mock
async def test_retries_on_503_then_succeeds(config, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    route = respx.post(URL)
    route.side_effect = [httpx.Response(503), _message('{"score": 7}')]
    backend = AnthropicBackend(api_key="test", max_retries=3)
    resp = await backend.call([{"role": "user", "content": "hi"}], config)
    assert resp.text == '{"score": 7}'
    assert route.call_count == 2
    await backend.aclose()


@respx.mock
async def test_persistent_retryable_status_exhausts_then_raises(config, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    route = respx.post(URL)
    route.mock(return_value=httpx.Response(503))
    backend = AnthropicBackend(api_key="test", max_retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        await backend.call([{"role": "user", "content": "hi"}], config)
    assert route.call_count == 3
    await backend.aclose()


@respx.mock
async def test_non_retryable_4xx_raises_immediately(config):
    route = respx.post(URL)
    route.mock(return_value=httpx.Response(400))
    backend = AnthropicBackend(api_key="test", max_retries=5)
    with pytest.raises(httpx.HTTPStatusError):
        await backend.call([{"role": "user", "content": "hi"}], config)
    assert route.call_count == 1
    await backend.aclose()


@respx.mock
async def test_network_errors_raise_runtimeerror_after_all_attempts(config, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    route = respx.post(URL)
    route.mock(side_effect=httpx.ConnectTimeout("boom"))
    backend = AnthropicBackend(api_key="test", max_retries=2)
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
        httpx.Response(429, headers={"retry-after": "1.5"}),
        _message('{"score": 9}'),
    ]
    backend = AnthropicBackend(api_key="test", max_retries=3)
    resp = await backend.call([{"role": "user", "content": "hi"}], config)
    assert resp.text == '{"score": 9}'
    assert slept == [1.5]
    await backend.aclose()


async def test_non_numeric_retry_after_falls_back_to_jitter(monkeypatch):
    slept: list[float] = []

    async def _record_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("asyncio.sleep", _record_sleep)
    backend = AnthropicBackend(api_key="test")
    # A malformed retry-after header can't be parsed, so jittered backoff is used.
    await backend._backoff(0, retry_after="soon")
    assert len(slept) == 1
    assert slept[0] > 0  # jittered delay, not the un-parseable header
    await backend.aclose()


@respx.mock
async def test_sends_version_and_key_headers(config):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["x-api-key"] = request.headers.get("x-api-key")
        captured["anthropic-version"] = request.headers.get("anthropic-version")
        return _message('{"score": 6}')

    respx.post(URL).mock(side_effect=handler)
    backend = AnthropicBackend(api_key="secret")
    await backend.call([{"role": "user", "content": "hi"}], config)
    assert captured["x-api-key"] == "secret"
    assert captured["anthropic-version"] == "2023-06-01"
    await backend.aclose()
