from functools import lru_cache

from sqlalchemy import Engine, create_engine

from residual_room.settings import get_settings


@lru_cache
def get_engine() -> Engine:
    # pool_pre_ping: Neon suspends idle databases, which silently kills pooled connections.
    return create_engine(get_settings().database_url, pool_pre_ping=True)
