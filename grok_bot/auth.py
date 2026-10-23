"""Password hashing (scrypt) and a small in-memory sliding-window rate limiter."""

import hashlib
import hmac
import re
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD = 8


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, digest = stored.split("$")
        if algo != "scrypt":
            return False
        calc = hashlib.scrypt(
            password.encode(),
            salt=bytes.fromhex(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(digest) // 2,
        )
    except ValueError:
        return False
    return hmac.compare_digest(calc.hex(), digest)


# Verified against when the username doesn't exist, so login timing doesn't leak usernames.
DUMMY_HASH = hash_password(secrets.token_hex(8))


class RateLimiter:
    """Allow `limit` hits per `window` seconds per key; `limit <= 0` disables limiting."""

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def hit(self, key: str) -> float:
        """Record a hit; return 0 if allowed, else seconds until the next hit is allowed."""
        if self.limit <= 0:
            return 0.0
        now = self.clock()
        hits = self._hits[key]
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return hits[0] + self.window - now
        hits.append(now)
        return 0.0
