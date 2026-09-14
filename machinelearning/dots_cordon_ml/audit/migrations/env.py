from alembic import context

from dots_cordon_ml.audit.schema import metadata


connection = context.config.attributes.get("connection")
if connection is None:
    raise RuntimeError("Use dots-cordon-audit db upgrade to configure the connection.")
context.configure(
    connection=connection,
    target_metadata=metadata,
    render_as_batch=connection.dialect.name == "sqlite",
)
with context.begin_transaction():
    context.run_migrations()
