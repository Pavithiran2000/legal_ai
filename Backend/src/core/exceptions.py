"""Application exceptions."""


class LegalAppException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class OutOfScopeError(LegalAppException):
    def __init__(self, message: str = "Query is out of scope", scope_category: str = "unknown"):
        self.scope_category = scope_category
        super().__init__(message, status_code=422)


class LLMServiceError(LegalAppException):
    def __init__(self, message: str = "LLM service error"):
        super().__init__(message, status_code=503)


class EmbeddingError(LegalAppException):
    def __init__(self, message: str = "Embedding service error"):
        super().__init__(message, status_code=503)


class FAISSError(LegalAppException):
    def __init__(self, message: str = "FAISS index error"):
        super().__init__(message, status_code=500)


class DocumentError(LegalAppException):
    def __init__(self, message: str = "Document processing error"):
        super().__init__(message, status_code=400)
