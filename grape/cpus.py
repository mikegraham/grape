"""How many threads grape's CPU math gets. No heavy imports."""

import os


def cpu_threads() -> int:
    """Three quarters of the usable CPUs.

    Every CPU (torch's default) was the slowest setting for a full run on
    both a hybrid 8-core laptop and a 24-core server: grape's image-decode
    threads need CPU at the same time.
    """
    cpus: int | None
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        cpus = len(getaffinity(0))  # respects taskset and containers
    else:
        cpus = os.cpu_count()
    if cpus is None:  # the OS can't say
        return 1
    return max(1, cpus * 3 // 4)
