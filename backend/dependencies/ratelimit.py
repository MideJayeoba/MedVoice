"""Simple in-memory rate limiting for sensitive endpoints.

Sliding-window counter per client IP. In-memory is appropriate here: the
service runs as a single Render instance (WEB_CONCURRENCY=1); swap for a
Redis-backed limiter if it ever scales horizontally.
"""

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:  # Render terminates TLS and forwards the real client IP first
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(max_calls: int, window_seconds: int, scope: str):
    """Dependency factory: at most `max_calls` per `window_seconds` per IP."""

    def dependency(request: Request) -> None:
        now = time.time()
        key = f"{scope}:{_client_ip(request)}"
        q = _hits[key]
        while q and q[0] <= now - window_seconds:
            q.popleft()
        if len(q) >= max_calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many attempts — please wait a minute and try again.",
                headers={"Retry-After": str(window_seconds)},
            )
        q.append(now)

    return dependency


# Shared limiters
login_limiter = rate_limit(max_calls=8, window_seconds=60, scope="login")
register_limiter = rate_limit(max_calls=5, window_seconds=60, scope="register")
