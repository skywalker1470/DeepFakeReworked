import time
from collections import defaultdict, deque

WINDOW_SECONDS = 60
MAX_REQUESTS = 5

_requests: dict[str, deque] = defaultdict(deque)


def is_rate_limited(ip: str) -> bool:
    now = time.monotonic()
    timestamps = _requests[ip]

    while timestamps and now - timestamps[0] > WINDOW_SECONDS:
        timestamps.popleft()

    if len(timestamps) >= MAX_REQUESTS:
        return True

    timestamps.append(now)
    return False
