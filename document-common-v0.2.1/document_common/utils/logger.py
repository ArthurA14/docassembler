import logging
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("haystack").setLevel(logging.WARNING)


class NestedObjectFilter(logging.Filter):
    """Filter that truncates nested objects in log messages."""

    def __init__(self, max_length: int = 10, max_depth: int = 3):
        self.max_length = max_length
        self.max_depth = max_depth

    def _truncate_nested(self, obj: Any, current_depth: int = 0) -> Any:
        """Recursively truncate nested objects."""
        if current_depth > self.max_depth:
            return "[[nested too deep...]]"

        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            shape = obj.shape
            if np.prod(shape) > self.max_length:
                return f"[ndarray(shape={shape}, dtype={obj.dtype})]"
            return obj

        # Handle sequences (list, tuple)
        if isinstance(obj, (list, tuple)):
            if len(obj) > self.max_length:
                return f"[{type(obj).__name__}({len(obj)} items)]"
            return type(obj)(self._truncate_nested(item, current_depth + 1) for item in obj)

        # Handle sets
        if isinstance(obj, set):
            if len(obj) > self.max_length:
                return f"[set({len(obj)} items)]"
            return {self._truncate_nested(item, current_depth + 1) for item in obj}

        # Handle dictionaries
        if isinstance(obj, dict):
            if len(obj) > self.max_length:
                return f"[dict({len(obj)} items)]"
            return {k: self._truncate_nested(v, current_depth + 1) for k, v in obj.items()}

        # Handle custom objects with __dict__
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            attrs = obj.__dict__
            if len(attrs) > self.max_length:
                return f"[{obj.__class__.__name__}({len(attrs)} attributes)]"
            return {k: self._truncate_nested(v, current_depth + 1) for k, v in attrs.items()}

        return obj

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record by truncating nested objects."""
        # Handle args
        if record.args:
            record.args = tuple(self._truncate_nested(arg) for arg in record.args)

        # Handle message if it's an object
        if not isinstance(record.msg, str):
            record.msg = self._truncate_nested(record.msg)

        return True


class Logger:
    """Singleton logger providing module-specific loggers with Rich console output and file logging.

    Each call to `get_logger(name)` returns a configured logger with nested object filtering
    and automatic file handler creation.
    """

    _instance: Optional["Logger"] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.nested_filter = NestedObjectFilter(max_length=10, max_depth=5)
        if not hasattr(self, "log_level"):
            self.log_level = logging.INFO
        if not Logger._initialized:
            # Install rich traceback
            install(show_locals=True)

            # Create console instance
            console = Console()

            # Create logs directory
            Path("logs").mkdir(exist_ok=True)

            # Base configuration
            self.log_format = "%(name)s - %(message)s"
            self.file_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
            self.date_format = "[%X]"
            self.log_level = logging.INFO

            # Configure root logger first
            root_logger = logging.getLogger()
            root_logger.setLevel(self.log_level)

            # Remove any existing handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            # Add single Rich handler to root logger
            rich_handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_path=True,
                enable_link_path=True,
            )
            rich_handler.setFormatter(logging.Formatter(self.log_format, datefmt=self.date_format))
            root_logger.addHandler(rich_handler)
            root_logger.addFilter(self.nested_filter)

            Logger._initialized = True

    def get_logger(self, name: str) -> logging.Logger:
        """Get a configured logger instance for a module"""
        logger = logging.getLogger(name)
        logger.addFilter(self.nested_filter)
        logger.setLevel(self.log_level)

        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(f"logs/{name.replace('.', '_')}.log")
            file_handler.setLevel(self.log_level)
            file_formatter = logging.Formatter(self.file_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def log(self, func):
        """Decorator for function logging"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = self.get_logger(func.__module__)
            logger.addFilter(self.nested_filter)

            for handler in logger.handlers:
                if isinstance(handler, RichHandler):
                    handler.markup = True

            logger.debug(
                f"[bold cyan]Entering[/] {func.__name__} - Args: {args}, Kwargs: {kwargs}",
                extra={"markup": True},
            )
            try:
                result = func(*args, **kwargs)
                logger.debug(
                    f"[bold green]Exiting[/] {func.__name__} - Result: {result}",
                    extra={"markup": True},
                )
                return result
            except Exception:
                logger.exception(f"[bold red]Error in {func.__name__}[/]", extra={"markup": True})
                raise

        return wrapper

    def set_level(self, level: str):
        self.log_level = getattr(logging, level, logging.INFO)
        for logger_name in logging.root.manager.loggerDict:
            logging.getLogger(logger_name).setLevel(self.log_level)
