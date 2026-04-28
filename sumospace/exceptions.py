# sumospace/exceptions.py


class SumoError(Exception):
    """Base exception for all Sumospace errors."""


class KernelBootError(SumoError):
    """Raised when kernel fails to initialize."""


class ExecutionHaltedError(SumoError):
    """Raised when a critical tool step fails."""


class ConsensusFailedError(SumoError):
    """Raised when the committee rejects an execution plan."""


class ProviderError(SumoError):
    """Raised when a model provider call fails."""


class IngestError(SumoError):
    """Raised during file ingestion."""


class ToolError(SumoError):
    """Raised when a tool execution fails."""


class ProviderNotConfiguredError(SumoError):
    """
    Raised when a cloud provider is used without the required API key or package.
    Provides clear install + key setup instructions in the message.
    """


class QuotaExceededError(SumoError):
    """
    Raised when chunk ingestion would exceed the configured max_chunks quota.
    Contains current count, attempted addition, and the limit.
    """

    def __init__(self, current: int, attempted: int, limit: int):
        self.current = current
        self.attempted = attempted
        self.limit = limit
        super().__init__(
            f"Quota exceeded: collection has {current} chunks, "
            f"tried to add {attempted}, limit is {limit}"
        )
