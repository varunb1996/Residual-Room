from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from residual_room.db.models import Base
from residual_room.settings import get_settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The URL comes from DATABASE_URL (via settings), not alembic.ini, so the secret
# lives only in .env. It's passed straight to SQLAlchemy rather than through
# config.set_main_option, because configparser chokes on %-encoded passwords.
url = get_settings().database_url

# Migrations are still hand-written; the metadata lets `alembic check` spot drift.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
