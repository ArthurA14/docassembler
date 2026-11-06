import uuid

# document assembly Exception:
class FilesNotImportedException(Exception):
    def __init__(self, details: str = "Some files are not imported yet."):
        self.details = details

# importation Exception:
class RequestTimeoutException(Exception):
    def __init__(self, message: str="Request timed out.") :
        super().__init__(message)
        self.error_id = str(uuid.uuid4())
        self.message = message