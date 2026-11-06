"""
Execution utilities for citation components.

Provides a context manager and utilities for executing pipeline components with
centralized error handling, timing, logging, and result wrapping.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, TypeVar

from document_common.utils.logger import Logger

from src.utils.component_results import ComponentMetadata, ComponentResult

T = TypeVar("T", bound=ComponentResult)


class ComponentExecutor:
    """Executor for pipeline components with unified error handling."""

    def __init__(self, component_name: str, request_id: Optional[str] = None):
        """
        Initialize the component executor.

        Args:
            component_name: Name of the component being executed
            request_id: Request ID for tracking across logs
        """
        self.component_name = component_name
        self.request_id = request_id or "unknown"
        self.logger = Logger().get_logger("citation_executor")
        self.start_time = time.time()
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def log_info(self, message: str) -> None:
        """Log info message with request ID."""
        self.logger.info(f"[{self.request_id}] {message}")

    def log_warning(self, message: str) -> None:
        """Log warning message and track it."""
        self.logger.warning(f"[{self.request_id}] {message}")
        self.warnings.append(message)

    def log_error(self, message: str) -> None:
        """Log error message and track it."""
        self.logger.error(f"[{self.request_id}] {message}")
        self.errors.append(message)

    def log_debug(self, message: str) -> None:
        """Log debug message with request ID."""
        self.logger.debug(f"[{self.request_id}] {message}")

    def get_duration(self) -> float:
        """Get elapsed time since executor creation."""
        return time.time() - self.start_time

    def create_metadata(self) -> ComponentMetadata:
        """Create metadata for the executed component."""
        return ComponentMetadata(
            component_name=self.component_name,
            duration_seconds=self.get_duration(),
            request_id=self.request_id,
        )

    def handle_error(self, exception: Exception, message: str) -> None:
        """
        Handle an error that occurred during component execution.

        Args:
            exception: The exception that occurred
            message: Additional message context
        """
        error_msg = f"{message}: {str(exception)}"
        self.log_error(error_msg)


@contextmanager
def execute_component(
    component_name: str,
    request_id: Optional[str] = None,
    log_start: bool = True,
    log_end: bool = True,
):
    """
    Context manager for component execution with error handling and timing.

    Provides unified error handling, logging, and timing for component execution.

    Args:
        component_name: Name of the component
        request_id: Request ID for tracking
        log_start: Whether to log component start
        log_end: Whether to log component end

    Yields:
        ComponentExecutor instance

    Example:
        >>> with execute_component("preprocessor", request_id) as executor:
        ...     result = component.run(...)
        ...     executor.log_info(f"Processed {len(result)} items")
    """
    executor = ComponentExecutor(component_name, request_id)

    if log_start:
        executor.log_info(f"Starting {component_name}")

    try:
        yield executor
    except Exception as e:
        executor.handle_error(e, f"Unhandled exception in {component_name}")
        raise
    finally:
        duration = executor.get_duration()
        if log_end:
            executor.log_info(
                f"{component_name} completed (duration: {duration:.3f}s, "
                f"errors: {len(executor.errors)}, warnings: {len(executor.warnings)})"
            )


class ExecutionMetrics:
    """Tracks metrics for multiple component executions."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.component_durations: Dict[str, float] = {}
        self.component_errors: Dict[str, List[str]] = {}
        self.component_warnings: Dict[str, List[str]] = {}
        self.total_start_time = time.time()

    def record_execution(self, executor: ComponentExecutor) -> None:
        """Record metrics from a component execution."""
        component_name = executor.component_name
        self.component_durations[component_name] = executor.get_duration()
        self.component_errors[component_name] = executor.errors.copy()
        self.component_warnings[component_name] = executor.warnings.copy()

    def get_total_duration(self) -> float:
        """Get total time since tracker creation."""
        return time.time() - self.total_start_time

    def get_component_duration(self, component_name: str) -> float:
        """Get duration for specific component."""
        return self.component_durations.get(component_name, 0.0)

    def get_all_errors(self) -> List[str]:
        """Get all errors from all components."""
        all_errors = []
        for errors in self.component_errors.values():
            all_errors.extend(errors)
        return all_errors

    def get_all_warnings(self) -> List[str]:
        """Get all warnings from all components."""
        all_warnings = []
        for warnings in self.component_warnings.values():
            all_warnings.extend(warnings)
        return all_warnings

    def had_errors(self) -> bool:
        """Check if any component had errors."""
        return len(self.get_all_errors()) > 0

    def had_warnings(self) -> bool:
        """Check if any component had warnings."""
        return len(self.get_all_warnings()) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "total_duration_seconds": self.get_total_duration(),
            "component_durations": self.component_durations,
            "errors_count": {k: len(v) for k, v in self.component_errors.items()},
            "warnings_count": {k: len(v) for k, v in self.component_warnings.items()},
            "total_errors": len(self.get_all_errors()),
            "total_warnings": len(self.get_all_warnings()),
        }
