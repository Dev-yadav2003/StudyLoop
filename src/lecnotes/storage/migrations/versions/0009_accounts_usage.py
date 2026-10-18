"""accounts: chapters.user_id, feedback.user_id, usage ledger

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-19 10:00:00.000000+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chapters", sa.Column("user_id", sa.Uuid(), nullable=True))
    op.create_index(op.f("ix_chapters_user_id"), "chapters", ["user_id"], unique=False)
    op.add_column("feedback", sa.Column("user_id", sa.Uuid(), nullable=True))
    op.create_table(
        "usage",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("ref", sa.Text(), nullable=False),
        sa.Column("video_minutes", sa.Float(), server_default="0", nullable=False),
        sa.Column("tokens", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("kind IN ('video', 'qa')", name=op.f("ck_usage_kind")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_usage")),
    )
    op.create_index(op.f("ix_usage_user_id"), "usage", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_usage_user_id"), table_name="usage")
    op.drop_table("usage")
    op.drop_column("feedback", "user_id")
    op.drop_index(op.f("ix_chapters_user_id"), table_name="chapters")
    op.drop_column("chapters", "user_id")
