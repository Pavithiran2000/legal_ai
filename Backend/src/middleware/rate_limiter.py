"""
Rate Limiter Middleware - Simple in-memory rate limiting.
"""
import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Simple token bucket rate limiter."""

    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/api/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip]
            if now - t < self._window
        ]

        if len(self._requests[client_ip]) >= self._max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )

        self._requests[client_ip].append(now)
        response = await call_next(request)
        return response
