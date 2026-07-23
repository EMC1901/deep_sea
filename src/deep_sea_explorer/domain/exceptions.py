class DomainError(Exception):
    code = "DOMAIN_ERROR"


class ValidationError(DomainError):
    code = "INVALID_REQUEST"


class ModelUnavailableError(DomainError):
    code = "MODEL_UNAVAILABLE"


class NotFoundError(DomainError):
    code = "NOT_FOUND"
