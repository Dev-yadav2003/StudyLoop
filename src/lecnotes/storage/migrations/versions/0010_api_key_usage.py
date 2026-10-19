"""api_key_usage: requests per (API key hash, model, Pacific day) for quota-aware routing

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-19 16:00:00.000000+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "api_key_usage",
        sa.Column("key_id", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("requests", sa.Integer(), server_default="0", nullable=False),
        sa.Column("tokens_in", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("tokens_out", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("exhausted", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("daily_limit", sa.Integer(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key_id", "model", "day", name=op.f("pk_api_key_usage")),
    )


def downgrade() -> None:
    op.drop_table("api_key_usage")
