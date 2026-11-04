from typing import Any


class DomainError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_error_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }

class InvalidFileFormatError(DomainError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(code="INVALID_FILE_FORMAT", message=message, details=details)

class DatasetNotFoundError(DomainError):
    def __init__(self, dataset_id: str):
        super().__init__(code="DATASET_NOT_FOUND", 
                        message=f"Dataset '{dataset_id}' não foi encontrado",
                        details={"dataset_id": dataset_id})

class DatasetNotValidatedError(DomainError):
    def __init__(self, dataset_id: str, reason: str | None = None):
        #Só aparece o reason caso o reason for escrito
        details: dict[str, Any] = {"dataset_id": dataset_id}
        if reason is not None:
            details["reason"] = reason

        super().__init__(code="DATASET_NOT_VALIDATED",
                        message="O dataset precisa ser validado antes dessa operação.",
                        details=details,
        )

class InvalidFeaturesError(DomainError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(code="INVALID_FEATURES", message=message, details=details)