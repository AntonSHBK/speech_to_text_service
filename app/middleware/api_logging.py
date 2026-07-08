from time import perf_counter
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.utils.logging import get_logger


api_logger = get_logger("api.requests")
get_logger("uvicorn.error", log_file="api.log")


def _short_user_agent(user_agent: str) -> str:
    if not user_agent:
        return "-"
    return user_agent[:120]


class ApiLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started_at = perf_counter()
        request_id = request.headers.get("x-request-id") or uuid4().hex[:12]
        request.state.request_id = request_id
        client_host = request.client.host if request.client else "-"
        method = request.method
        path = request.url.path
        user_agent = _short_user_agent(request.headers.get("user-agent", "-"))

        api_logger.info(
            "request started | request_id=%s | method=%s | path=%s | client=%s | user_agent=%s",
            request_id,
            method,
            path,
            client_host,
            user_agent,
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started_at) * 1000
            api_logger.exception(
                "request failed | request_id=%s | method=%s | path=%s | status=500 | duration_ms=%.2f | client=%s | user_agent=%s",
                request_id,
                method,
                path,
                duration_ms,
                client_host,
                user_agent,
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1000
        response.headers["X-Request-ID"] = request_id

        if response.status_code >= 400:
            api_logger.warning(
                "request completed | request_id=%s | method=%s | path=%s | status=%s | duration_ms=%.2f | client=%s | user_agent=%s",
                request_id,
                method,
                path,
                response.status_code,
                duration_ms,
                client_host,
                user_agent,
            )
            return response

        api_logger.info(
            "request completed | request_id=%s | method=%s | path=%s | status=%s | duration_ms=%.2f | client=%s",
            request_id,
            method,
            path,
            response.status_code,
            duration_ms,
            client_host,
        )
        return response
