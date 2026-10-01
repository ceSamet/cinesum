import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional


@contextmanager
def measured_stage(
    on_complete: Optional[Callable[[float], None]] = None,
) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        if on_complete:
            on_complete(elapsed)

