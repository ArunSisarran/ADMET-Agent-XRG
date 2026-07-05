"""Shared retry/backoff helper for Gemini calls.

The free tier's RPM limit is low enough (observed: 5 requests/minute for
gemini-2.5-flash) that 429s are routine during a discovery run touching more
than a handful of candidates, not an edge case. Google's error message
usually includes a suggested retry delay ("Please retry in 14.8s") — use it
when present instead of guessing.
"""

import asyncio
import re

RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _extract_retry_delay(error_message: str) -> float | None:
    match = RETRY_DELAY_RE.search(error_message)
    return float(match.group(1)) if match else None


async def ainvoke_with_backoff(model, messages, max_retries: int = 6, default_delay: float = 15.0):
    delay = default_delay
    for attempt in range(max_retries):
        try:
            return await model.ainvoke(messages)
        except Exception as e:
            message = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in message or "429" in message
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            wait = (_extract_retry_delay(message) or delay) + 1
            print(f"    Gemini rate limit hit, waiting {wait:.0f}s (retry {attempt + 1}/{max_retries})...")
            await asyncio.sleep(wait)
            delay *= 1.5
