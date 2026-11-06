"""
NOTE: PopcornNer component is not yet finished. These tests are temporarily
commented out and will be re-enabled once the component is production-ready.
"""

# from unittest.mock import MagicMock, patch
#
# import pytest
#
# from document_common.components.base.popcorn_ner import PopcornNerClient, PopcornNerResponse
#
#
# class TestPopcornNerResponse:
#     def test_valid_response(self):
#         raw = {"entities": [{"text": "test", "label": "PERSON"}]}
#         resp = PopcornNerResponse(raw=raw)
#         assert resp.raw == raw
#
#
# class TestPopcornNerClient:
#     @patch("document_common.components.base.popcorn_ner.config")
#     def test_init(self, mock_config, mock_logger):
#         mock_config.get_current_config.return_value = MagicMock(
#             popcorn_ner=MagicMock(
#                 url="http://localhost:8000",
#                 endpoints=MagicMock(health="/health", ner="/extract"),
#                 request=MagicMock(timeout_s=60, retries=2, backoff_s=0.5),
#                 payload=MagicMock(default_lang="eng", text_field="text", lang_field="lang"),
#             )
#         )
#         client = PopcornNerClient()
#         assert client.base_url == "http://localhost:8000"
#         assert client.health_path == "/health"
#         assert client.ner_path == "/extract"
#         assert client.timeout == 60
#         assert client.retries == 2
#         assert client.default_lang == "eng"
#         assert client.text_field == "text"
#         assert client.lang_field == "lang"
#
#     @patch("document_common.components.base.popcorn_ner.config")
#     def test_init_missing_config_raises(self, mock_config, mock_logger):
#         mock_config.get_current_config.return_value = {}
#         with pytest.raises(KeyError, match="Cannot find 'popcorn_ner'"):
#             PopcornNerClient()
#
#     @patch("document_common.components.base.popcorn_ner.requests.get")
#     def test_health(self, mock_get, mock_config, mock_logger):
#         mock_config.get_current_config.return_value = MagicMock(
#             popcorn_ner=MagicMock(
#                 url="http://localhost:8000",
#                 endpoints=MagicMock(health="/health", ner="/extract"),
#                 request=MagicMock(timeout_s=60, retries=2, backoff_s=0.5),
#                 payload=MagicMock(default_lang="eng", text_field="text", lang_field="lang"),
#             )
#         )
#         mock_response = MagicMock()
#         mock_response.headers = {"content-type": "application/json"}
#         mock_response.json.return_value = {"status": "healthy"}
#         mock_response.raise_for_status.return_value = None
#         mock_get.return_value = mock_response
#
#         client = PopcornNerClient()
#         result = client.health()
#         assert result == {"status": "healthy"}
#
#     @patch("document_common.components.base.popcorn_ner.requests.post")
#     def test_ner_success(self, mock_post, mock_config, mock_logger):
#         mock_config.get_current_config.return_value = MagicMock(
#             popcorn_ner=MagicMock(
#                 url="http://localhost:8000",
#                 endpoints=MagicMock(health="/health", ner="/extract"),
#                 request=MagicMock(timeout_s=60, retries=2, backoff_s=0.5),
#                 payload=MagicMock(default_lang="eng", text_field="text", lang_field="lang"),
#             )
#         )
#         mock_response = MagicMock()
#         mock_response.json.return_value = {"entities": []}
#         mock_response.raise_for_status.return_value = None
#         mock_post.return_value = mock_response
#
#         client = PopcornNerClient()
#         result = client.ner(text="Hello world")
#         assert isinstance(result, PopcornNerResponse)
#
#     @patch("document_common.components.base.popcorn_ner.requests.post")
#     def test_ner_with_lang(self, mock_post, mock_config, mock_logger):
#         mock_config.get_current_config.return_value = MagicMock(
#             popcorn_ner=MagicMock(
#                 url="http://localhost:8000",
#                 endpoints=MagicMock(health="/health", ner="/extract"),
#                 request=MagicMock(timeout_s=60, retries=2, backoff_s=0.5),
#                 payload=MagicMock(default_lang="eng", text_field="text", lang_field="lang"),
#             )
#         )
#         mock_response = MagicMock()
#         mock_response.json.return_value = {"entities": []}
#         mock_response.raise_for_status.return_value = None
#         mock_post.return_value = mock_response
#
#         client = PopcornNerClient()
#         client.ner(text="Hello", lang="fra")
#         call_kwargs = mock_post.call_args.kwargs
#         assert call_kwargs["json"]["lang"] == "fra"
#
#     @patch("document_common.components.base.popcorn_ner.requests.post")
#     def test_ner_failure_after_retries(self, mock_post, mock_config, mock_logger):
#         mock_config.get_current_config.return_value = MagicMock(
#             popcorn_ner=MagicMock(
#                 url="http://localhost:8000",
#                 endpoints=MagicMock(health="/health", ner="/extract"),
#                 request=MagicMock(timeout_s=1, retries=1, backoff_s=0.01),
#                 payload=MagicMock(default_lang="eng", text_field="text", lang_field="lang"),
#             )
#         )
#         mock_post.side_effect = Exception("Connection error")
#
#         client = PopcornNerClient()
#         with pytest.raises(RuntimeError, match="failed after"):
#             client.ner(text="Hello")
#
#     @patch("document_common.components.base.popcorn_ner.config")
#     def test_init_with_dict_config(self, mock_config, mock_logger):
#         mock_config.get_current_config.return_value = {
#             "popcorn_ner": MagicMock(
#                 url="http://localhost:9000",
#                 endpoints=MagicMock(health="/health", ner="/extract"),
#                 request=MagicMock(timeout_s=30, retries=1, backoff_s=0.25),
#                 payload=MagicMock(default_lang="fra", text_field="text", lang_field="lang"),
#             )
#         }
#         client = PopcornNerClient()
#         assert client.base_url == "http://localhost:9000"
#         assert client.timeout == 30