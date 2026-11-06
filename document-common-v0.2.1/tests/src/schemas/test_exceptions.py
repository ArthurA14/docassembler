import pytest

from document_common.schemas.exceptions import FilesNotImportedException, RequestTimeoutException


class TestFilesNotImportedException:
    def test_default_message(self):
        e = FilesNotImportedException()
        assert e.details == "Some files are not imported yet."

    def test_custom_message(self):
        e = FilesNotImportedException(details="File xyz.pdf is not imported")
        assert e.details == "File xyz.pdf is not imported"

    def test_is_exception(self):
        e = FilesNotImportedException()
        assert isinstance(e, Exception)


class TestRequestTimeoutException:
    def test_default_message(self):
        e = RequestTimeoutException()
        assert e.message == "Request timed out."
        assert hasattr(e, "error_id")
        assert isinstance(e.error_id, str)

    def test_custom_message(self):
        e = RequestTimeoutException(message="Connection timeout")
        assert e.message == "Connection timeout"
        assert hasattr(e, "error_id")

    def test_is_exception(self):
        e = RequestTimeoutException()
        assert isinstance(e, Exception)

    def test_error_id_is_uuid_format(self):
        e = RequestTimeoutException()
        # Basic UUID format check: contains hyphens and has expected length
        parts = e.error_id.split("-")
        assert len(parts) == 5  # standard UUID format
