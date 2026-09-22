"""Offline checks for the supplied OpenAI-compatible course client."""

import json
import unittest
from unittest.mock import patch

from llm_client import OpenAICompatibleClient


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": '{"status":"ok"}'}}]}
        ).encode("utf-8")


class OpenAICompatibleClientTests(unittest.TestCase):
    def test_chat_sends_required_request_and_returns_content(self) -> None:
        client = OpenAICompatibleClient(
            model="example-model",
            api_key="required-placeholder",
            base_url="http://example.test/v1/",
            temperature=0.25,
            seed=7,
            max_tokens=64,
        )
        messages = ({"role": "user", "content": "test"},)
        with patch("llm_client.urllib.request.urlopen", return_value=_Response()) as urlopen:
            content = client.chat(messages)

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer required-placeholder")
        self.assertEqual(payload["model"], "example-model")
        self.assertEqual(payload["messages"], list(messages))
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertFalse(payload["stream"])
        self.assertEqual(content, '{"status":"ok"}')

    def test_chat_requires_nonempty_key(self) -> None:
        client = OpenAICompatibleClient("example-model", "")
        with self.assertRaisesRegex(ValueError, "api_key must be provided"):
            client.chat(())


if __name__ == "__main__":
    unittest.main()
