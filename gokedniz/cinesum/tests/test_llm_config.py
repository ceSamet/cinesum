import os
import unittest
from unittest.mock import patch

from src.core.llm_config import load_llm_config


class TestLLMConfig(unittest.TestCase):
    def test_missing_key_disables_optional_llm(self):
        with patch.dict(os.environ, {}, clear=True):
            config = load_llm_config(load_env_file=False)
        self.assertFalse(config.enabled)
        self.assertEqual(config.timeout_sec, 60.0)

    def test_secret_is_excluded_from_repr_and_public_metadata(self):
        sentinel = "prlc_test_secret_never_expose"
        with patch.dict(os.environ, {"PARALON_API_KEY": sentinel}, clear=True):
            config = load_llm_config(load_env_file=False)
        self.assertTrue(config.enabled)
        self.assertNotIn(sentinel, repr(config))
        self.assertNotIn(sentinel, str(config.to_public_dict()))

    def test_remote_plain_http_is_rejected(self):
        with patch.dict(
            os.environ,
            {"PARALON_BASE_URL": "http://example.com/v1"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                load_llm_config(load_env_file=False)

    def test_local_http_is_allowed_for_tests(self):
        with patch.dict(
            os.environ,
            {"PARALON_BASE_URL": "http://127.0.0.1:9000/v1"},
            clear=True,
        ):
            config = load_llm_config(load_env_file=False)
        self.assertEqual(config.base_url, "http://127.0.0.1:9000/v1")

    def test_timeout_has_safe_bounds(self):
        with patch.dict(os.environ, {"PARALON_TIMEOUT_SEC": "999"}, clear=True):
            with self.assertRaises(ValueError):
                load_llm_config(load_env_file=False)


if __name__ == "__main__":
    unittest.main()
