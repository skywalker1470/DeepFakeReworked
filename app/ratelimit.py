from ratelimit import is_rate_limited, WINDOW_SECONDS, MAX_REQUESTS


def check_rate_limit(request):
    from fastapi import HTTPException

    ip = request.client.host if request.client else "unknown"
    if is_rate_limited(ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {MAX_REQUESTS} requests per {WINDOW_SECONDS}s",
        )
