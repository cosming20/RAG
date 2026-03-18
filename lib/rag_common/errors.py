"""Shared error types for the RAG swarm system."""

from __future__ import annotations


class RagError(Exception):
    """Base error for all RAG swarm operations."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class TaskError(RagError):
    """Error during task processing."""


class TaskTimeoutError(TaskError):
    """Task exceeded its deadline."""

    def __init__(self, task_id: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Task {task_id} timed out after {timeout_seconds}s",
            code="TASK_TIMEOUT",
        )
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds


class TaskPoisonError(TaskError):
    """Task exceeded max retry attempts."""

    def __init__(self, task_id: str, attempts: int) -> None:
        super().__init__(
            f"Task {task_id} poisoned after {attempts} attempts",
            code="TASK_POISON",
        )
        self.task_id = task_id
        self.attempts = attempts


class PipelineDepthError(TaskError):
    """Pipeline depth limit exceeded (loop prevention)."""

    def __init__(self, task_id: str, depth: int, max_depth: int) -> None:
        super().__init__(
            f"Task {task_id} reached pipeline depth {depth} (max {max_depth})",
            code="PIPELINE_DEPTH_EXCEEDED",
        )


class AgentError(RagError):
    """Error in agent lifecycle."""


class AgentRegistrationError(AgentError):
    """Failed to register agent in Redis."""


class CircuitOpenError(RagError):
    """Circuit breaker is open for target service."""

    def __init__(self, target: str) -> None:
        super().__init__(f"Circuit breaker open for {target}", code="CIRCUIT_OPEN")
        self.target = target


class DocumentError(RagError):
    """Error in document processing."""


class DocumentValidationError(DocumentError):
    """Document failed preflight validation."""

    def __init__(self, filename: str, errors: list[str]) -> None:
        super().__init__(
            f"Validation failed for {filename}: {'; '.join(errors)}",
            code="DOCUMENT_VALIDATION_FAILED",
        )
        self.filename = filename
        self.validation_errors = errors


class DocumentTooLargeError(DocumentError):
    """Document exceeds size limit."""

    def __init__(self, filename: str, size: int, max_size: int) -> None:
        super().__init__(
            f"{filename} is {size} bytes, max is {max_size}",
            code="DOCUMENT_TOO_LARGE",
        )


class EmbeddingError(RagError):
    """Error generating embeddings."""


class RetrievalError(RagError):
    """Error during vector/graph retrieval."""


class SynthesisError(RagError):
    """Error during answer synthesis."""


class ValidationError(RagError):
    """Error during answer validation."""
