import time
import functools
from typing import Callable, Any, Optional
import logging

# Set up logging for timing information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def timeit(
    func: Optional[Callable] = None,
    *,
    unit: str = "seconds",
    precision: int = 4,
    log_result: bool = True,
    return_time: bool = False,
) -> Callable:
    """
    A decorator to measure and optionally log the execution time of functions.

    Args:
        func: The function to be decorated (automatically passed when used as @timeit)
        unit: Time unit for display ("seconds", "milliseconds", "microseconds")
        precision: Number of decimal places to show
        log_result: Whether to log the timing result
        return_time: Whether to return execution time along with function result

    Returns:
        Decorated function or decorator function

    Usage:
        @timeit
        def my_function():
            pass

        @timeit(unit="milliseconds", precision=2)
        def another_function():
            pass

        @timeit(return_time=True)
        def function_with_timing():
            pass
        # This will return (result, execution_time)
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()

            try:
                result = f(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Convert time to requested unit
                if unit == "milliseconds":
                    display_time = execution_time * 1000
                    unit_symbol = "ms"
                elif unit == "microseconds":
                    display_time = execution_time * 1_000_000
                    unit_symbol = "μs"
                else:  # seconds
                    display_time = execution_time
                    unit_symbol = "s"

                # Log the result if requested
                if log_result:
                    logger.info(
                        f"Function '{f.__name__}' executed in "
                        f"{display_time:.{precision}f} {unit_symbol}"
                    )

                # Return result with or without timing
                if return_time:
                    return result, execution_time
                else:
                    return result

            except Exception as e:
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                if log_result:
                    logger.error(
                        f"Function '{f.__name__}' failed after "
                        f"{execution_time:.{precision}f} seconds with error: {e}"
                    )
                raise

        return wrapper

    # Handle both @timeit and @timeit(...) usage
    if func is None:
        # Called with arguments: @timeit(...)
        return decorator
    else:
        # Called without arguments: @timeit
        return decorator(func)


class TimeitContext:
    """
    Context manager for timing code blocks.

    Usage:
        with TimeitContext("My operation"):
            # code to time
            pass

        with TimeitContext("My operation", unit="milliseconds") as timer:
            # code to time
            pass
        print(f"Elapsed time: {timer.elapsed_time}")
    """

    def __init__(
        self,
        name: str = "Operation",
        unit: str = "seconds",
        precision: int = 4,
        log_result: bool = True,
    ):
        self.name = name
        self.unit = unit
        self.precision = precision
        self.log_result = log_result
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

        # Convert time to requested unit
        if self.unit == "milliseconds":
            display_time = self.elapsed_time * 1000
            unit_symbol = "ms"
        elif self.unit == "microseconds":
            display_time = self.elapsed_time * 1_000_000
            unit_symbol = "μs"
        else:  # seconds
            display_time = self.elapsed_time
            unit_symbol = "s"

        if self.log_result:
            if exc_type is None:
                logger.info(
                    f"'{self.name}' completed in "
                    f"{display_time:.{self.precision}f} {unit_symbol}"
                )
            else:
                logger.error(
                    f"'{self.name}' failed after "
                    f"{display_time:.{self.precision}f} {unit_symbol}"
                )


class FunctionProfiler:
    """
    Class to collect and analyze timing statistics for multiple function calls.

    Usage:
        profiler = FunctionProfiler()

        @profiler.profile
        def my_function():
            pass

        # Call function multiple times
        my_function()
        my_function()

        # Get statistics
        profiler.print_stats()
    """

    def __init__(self):
        self.stats = {}

    def profile(self, func: Callable) -> Callable:
        """Decorator to profile a function and collect statistics."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Store statistics
                func_name = func.__name__
                if func_name not in self.stats:
                    self.stats[func_name] = []
                self.stats[func_name].append(execution_time)

                return result

            except Exception as e:
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Store failed execution time
                func_name = f"{func.__name__}_FAILED"
                if func_name not in self.stats:
                    self.stats[func_name] = []
                self.stats[func_name].append(execution_time)

                raise

        return wrapper

    def print_stats(self, unit: str = "seconds"):
        """Print timing statistics for all profiled functions."""
        if not self.stats:
            print("No profiling data available.")
            return

        print(f"\n{'=' * 60}")
        print("FUNCTION PROFILING STATISTICS")
        print(f"{'=' * 60}")

        for func_name, times in self.stats.items():
            if not times:
                continue

            # Convert times to requested unit
            if unit == "milliseconds":
                converted_times = [t * 1000 for t in times]
                unit_symbol = "ms"
            elif unit == "microseconds":
                converted_times = [t * 1_000_000 for t in times]
                unit_symbol = "μs"
            else:  # seconds
                converted_times = times
                unit_symbol = "s"

            avg_time = sum(converted_times) / len(converted_times)
            min_time = min(converted_times)
            max_time = max(converted_times)
            total_time = sum(converted_times)

            print(f"\nFunction: {func_name}")
            print(f"  Calls: {len(times)}")
            print(f"  Total: {total_time:.4f} {unit_symbol}")
            print(f"  Average: {avg_time:.4f} {unit_symbol}")
            print(f"  Min: {min_time:.4f} {unit_symbol}")
            print(f"  Max: {max_time:.4f} {unit_symbol}")

    def get_stats(self, func_name: str) -> dict:
        """Get timing statistics for a specific function."""
        times = self.stats.get(func_name, [])
        if not times:
            return {}

        return {
            "calls": len(times),
            "total": sum(times),
            "average": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "times": times.copy(),
        }

    def clear_stats(self):
        """Clear all collected statistics."""
        self.stats.clear()


# Example usage and test functions
if __name__ == "__main__":
    # Test the basic decorator
    @timeit
    def example_function():
        time.sleep(0.1)
        return "Hello, World!"

    @timeit(unit="milliseconds", precision=2)
    def another_example():
        time.sleep(0.05)
        return [i**2 for i in range(1000)]

    @timeit(return_time=True)
    def timed_function():
        time.sleep(0.02)
        return "Result with timing"

    print("Testing basic decorator:")
    result1 = example_function()
    print(f"Result: {result1}\n")

    print("Testing with custom units:")
    result2 = another_example()
    print(f"Result length: {len(result2)}\n")

    print("Testing return_time option:")
    result3, exec_time = timed_function()
    print(f"Result: {result3}, Execution time: {exec_time:.6f} seconds\n")

    # Test context manager
    print("Testing context manager:")
    with TimeitContext("List comprehension", unit="milliseconds"):
        squares = [i**2 for i in range(10000)]
    print(f"Generated {len(squares)} squares\n")

    # Test profiler
    print("Testing function profiler:")
    profiler = FunctionProfiler()

    @profiler.profile
    def test_function(n):
        return sum(range(n))

    # Call multiple times
    for i in range(5):
        test_function(1000 * (i + 1))

    profiler.print_stats(unit="milliseconds")
