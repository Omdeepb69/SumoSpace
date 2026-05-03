# sumospace/exceptions.py


class SumoSpaceError(Exception):
    """Base exception for all SumoSpace errors."""


class KernelBootError(SumoSpaceError):
    """Raised when kernel fails to initialize."""


class ExecutionHaltedError(SumoSpaceError):
    """Raised when a critical tool step fails."""


class ConsensusFailedError(SumoSpaceError):
    """Raised when the committee rejects an execution plan."""


class ProviderError(SumoSpaceError):
    """Raised when a model provider call fails."""


class IngestError(SumoSpaceError):
    """Raised during file ingestion."""


class ToolError(SumoSpaceError):
    """Raised when a tool execution fails."""


class ProviderNotConfiguredError(SumoSpaceError):
    """
    Raised when a cloud provider is used without the required API key or package.
    Provides clear install + key setup instructions in the message.
    """


class QuotaExceededError(SumoSpaceError):
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
