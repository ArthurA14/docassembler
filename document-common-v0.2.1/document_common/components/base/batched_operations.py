"""
Generic Batched Operations Utility

A simple, standalone utility for batching any function with any arguments.
No dependencies on other batch processors - completely independent and reusable.

Usage:
    # Batch any function with any arguments
    batcher = BatchedOperations(batch_size=32)

    # Example 1: Simple function
    results = batcher.batch_call(print, ["hello", "world", "test"], n_jobs=3)

    # Example 2: Function with keyword args
    results = batcher.batch_call(
        my_function,
        positional_args=[["arg1", "arg2"], ["arg3", "arg4"]],
        keyword_args=[{"param": "value1"}, {"param": "value2"}]
    )

    # Example 3: Citation classification
    results = batcher.batch_call(
        classifier.run,
        positional_args=[[text1, source1, score1], [text2, source2, score2]],
        keyword_args=[None, None]  # No keyword arguments
    )
"""

import asyncio
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from document_common.utils.logger import Logger


class BatchedOperations:
    """
    A simple, independent utility class for batching any function calls.

    This is a lightweight, standalone batch processor that can handle
    any function with any combination of positional and keyword arguments.
    No dependencies on other batch processing components.

    Features:
    - Generic function batching (works with any callable)
    - Configurable batch size
    - Error handling with exceptions collected
    - Progress tracking with tqdm
    - Sync and async operation support
    - Memory-efficient for large datasets

    Args:
        batch_size: Number of function calls to process concurrently per batch
                   (default: 32, recommended: 16-128 for LLM operations)
        show_progress: Whether to display progress bar (default: True)
        max_workers: Maximum worker threads for CPU-bound operations (default: 4)
    """

    def __init__(self, batch_size: int = 32, show_progress: bool = True, max_workers: int = 4):
        """
        Initialize BatchedOperations utility.

        Args:
            batch_size: Items per batch for concurrent processing
            show_progress: Whether to show progress bars during processing
            max_workers: Max worker threads for CPU-intensive operations
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.batch_size = batch_size
        self.show_progress = show_progress
        self.max_workers = max_workers
        self.logger = Logger().get_logger(self.__class__.__name__)

        self.logger.debug(f"Initialized BatchedOperations (batch_size={batch_size}, max_workers={max_workers})")

    def batch_call(
        self,
        func: Callable,
        positional_args_list: List[List[Any]],
        keyword_args_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        use_async: bool = False,
        **kwargs,
    ) -> List[Any]:
        """
        Batch execute a function with multiple argument sets.

        This is the main entry point for batching any function. It takes a function
        and a list of argument sets, then processes them in batches.

        Args:
            func: The function to call for each argument set
            positional_args_list: List of lists, each containing positional arguments
                                Example: [["arg1", "arg2"], ["arg3", "arg4"]]
            keyword_args_list: List of keyword argument dicts, parallel to positional_args_list
                              Example: [{"param1": "value1"}, {"param2": "value2"}]
                              Use None for items with no keyword arguments
            use_async: If True, run the function asynchronously (default: False)
            **kwargs: Additional keyword arguments to pass to ALL function calls

        Returns:
            List of results in the same order as input arguments.
            Exceptions are returned as results (not raised) for partial success.

        Example:
            ```python
            def my_function(text, source, score):
                return f"Processing: {text[:20]} (score: {score})"

            # Batch the function calls
            batcher = BatchedOperations(batch_size=4)
            results = batcher.batch_call(
                my_function,
                [["text1", "source1", 0.8], ["text2", "source2", 0.9]],
                [None, None]  # No keyword arguments
            )
            print(results)  # ['Processing: text1 (score: 0.8)', 'Processing: text2 (score: 0.9)']
            ```
        """
        if not positional_args_list:
            self.logger.debug("No arguments to process")
            return []

        # Validate inputs
        if not isinstance(positional_args_list, list):
            raise ValueError("positional_args_list must be a list")

        if keyword_args_list is None:
            keyword_args_list = [None] * len(positional_args_list)
        elif len(keyword_args_list) != len(positional_args_list):
            raise ValueError(
                f"keyword_args_list length ({len(keyword_args_list)}) must match "
                f"positional_args_list length ({len(positional_args_list)})"
            )

        self.logger.debug(
            f"Batch calling function {func.__name__} with {len(positional_args_list)} "
            f"argument sets in batches of {self.batch_size}"
        )

        if use_async:
            return asyncio.run(self._batch_call_async(func, positional_args_list, keyword_args_list, kwargs))
        else:
            return self._batch_call_sync(func, positional_args_list, keyword_args_list, kwargs)

    async def _batch_call_async(
        self,
        func: Callable,
        positional_args_list: List[List[Any]],
        keyword_args_list: List[Optional[Dict[str, Any]]],
        global_kwargs: Dict[str, Any],
    ) -> List[Any]:
        """
        Internal async implementation of batch calling.
        """
        results = []
        num_batches = (len(positional_args_list) + self.batch_size - 1) // self.batch_size

        iterator = (
            tqdm(range(num_batches), disable=not self.show_progress, desc="Processing batches")
            if self.show_progress
            else range(num_batches)
        )

        for batch_idx in iterator:
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(positional_args_list))

            batch_args = positional_args_list[batch_start:batch_end]
            batch_kwargs = keyword_args_list[batch_start:batch_end]

            self.logger.debug(
                f"Processing batch {batch_idx + 1}/{num_batches} "
                f"({len(batch_args)} calls, indices {batch_start}-{batch_end - 1})"
            )

            # Create tasks for this batch
            tasks = []
            for args, kwargs in zip(batch_args, batch_kwargs):
                if kwargs is None:
                    kwargs = {}
                # Merge global kwargs with per-call kwargs (per-call takes precedence)
                merged_kwargs = {**global_kwargs, **kwargs}

                if asyncio.iscoroutinefunction(func):
                    tasks.append(func(*args, **merged_kwargs))
                else:
                    # If it's a sync function, run it in executor
                    loop = asyncio.get_running_loop()
                    tasks.append(loop.run_in_executor(None, func, *args, **merged_kwargs))

            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)

        # Log completion statistics
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful

        message = f"Batch calling complete: {successful} successful, {failed} failed "
        message += f"(total calls: {len(positional_args_list)}, batches: {num_batches})"
        if failed :
            self.logger.info(message)
        else:
            self.logger.debug(message)
        return results

    def _batch_call_sync(
        self,
        func: Callable,
        positional_args_list: List[List[Any]],
        keyword_args_list: List[Optional[Dict[str, Any]]],
        global_kwargs: Dict[str, Any],
    ) -> List[Any]:
        """
        Internal sync implementation of batch calling.
        """
        try:
            # Check if we're in an existing event loop
            loop = asyncio.get_running_loop()
            # We're in an async context, use thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future = pool.submit(
                    asyncio.run, self._batch_call_async(func, positional_args_list, keyword_args_list, global_kwargs)
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(self._batch_call_async(func, positional_args_list, keyword_args_list, global_kwargs))

    def batch_method(
        self,
        method_name: str,
        objects: List[Any],
        positional_args_list: Optional[List[List[Any]]] = None,
        keyword_args_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Batch call a method on multiple objects.

        Convenience method for calling the same method on multiple objects.

        Args:
            method_name: Name of method to call on each object
            objects: List of objects to call method on
            positional_args_list: Arguments to pass to each method call
            keyword_args_list: Keyword arguments for each method call
            **kwargs: Global keyword arguments for all method calls

        Returns:
            List of method call results

        Example:
            ```python
            # Call 'classify' method on multiple classifier instances
            results = batcher.batch_method(
                "run",
                [classifier1, classifier2, classifier3],
                [["text1", "source1", 0.8], ["text2", "source2", 0.9], ["text3", "source3", 0.7]],
                [None, None, None]
            )
            ```
        """
        if not objects:
            return []

        # Create function that calls method on object
        def call_method(obj, args, kwargs):
            method = getattr(obj, method_name)
            if kwargs is None:
                kwargs = {}
            merged_kwargs = {**kwargs, **global_kwargs}
            return method(*args, **merged_kwargs)

        # Create positional args list if not provided
        if positional_args_list is None:
            positional_args_list = [[] for _ in objects]

        return self.batch_call(call_method, positional_args_list, keyword_args_list, **kwargs)

    def validate_results(
        self, positional_args_list: List[List[Any]], results: List[Any]
    ) -> tuple[List[Any], List[tuple[int, Exception]]]:
        """
        Separate successful results from failed calls.

        Args:
            positional_args_list: Original argument list (for indexing)
            results: Results from batch_call (may contain exceptions)

        Returns:
            Tuple of (successful_results, failed_calls) where failed_calls
            is list of (index, exception) tuples
        """
        successful = []
        failures = []

        for idx, (args, result) in enumerate(zip(positional_args_list, results)):
            if isinstance(result, Exception):
                failures.append((idx, result))
            else:
                successful.append(result)

        return successful, failures

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about this batch operations instance.

        Returns:
            Dictionary with configuration and capabilities
        """
        return {
            "class": self.__class__.__name__,
            "batch_size": self.batch_size,
            "show_progress": self.show_progress,
            "max_workers": self.max_workers,
            "supports_async": True,
            "supports_sync": True,
            "error_handling": "exceptions_collected",
            "progress_tracking": self.show_progress,
        }


# Convenience function for one-off batching
def batch_function_calls(
    func: Callable,
    positional_args_list: List[List[Any]],
    keyword_args_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    batch_size: int = 32,
    **kwargs,
) -> List[Any]:
    """
    Convenience function for batching function calls without creating a BatchedOperations instance.

    Args:
        func: Function to call
        positional_args_list: List of positional argument lists
        keyword_args_list: List of keyword argument dicts
        batch_size: Batch size for processing
        **kwargs: Additional keyword arguments

    Returns:
        List of function call results

    Example:
        ```python
        # One-off batch call
        results = batch_function_calls(
            print,
            [["Hello"], ["World"], ["Test"]],
            [None, None, None]
        )
        ```
    """
    batcher = BatchedOperations(batch_size=batch_size)
    return batcher.batch_call(func, positional_args_list, keyword_args_list, **kwargs)
