from functools import lru_cache
from typing import Protocol


class _HexDigestable(Protocol):
    def hexdigest(self) -> str:
        ...


class _HasherProto(Protocol):
    def __call__(self, bytes) -> _HexDigestable:
        ...


@lru_cache
def hash_username(hasher: _HasherProto, username: str, salt: bytes = b"") -> str:
    """
    Hash a human readable username to produce a user identifier.

    This user identifier is meant to be transmitted to the OpenAI API in place
    of a human readable username.
    """
    return hasher(salt + username.encode()).hexdigest()
