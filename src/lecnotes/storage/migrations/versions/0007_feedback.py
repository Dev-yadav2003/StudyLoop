"""feedback: thumbs up/down on notes

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-18 13:00:00.000000+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("notes_id", sa.Uuid(), nullable=False),
        sa.Column("unit_key", sa.Text(), nullable=True),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("trace_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("rating IN (-1, 1)", name=op.f("ck_feedback_rating")),
        sa.ForeignKeyConstraint(
            ["notes_id"], ["notes.id"], name=op.f("fk_feedback_notes_id_notes"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feedback")),
    )
    op.create_index(op.f("ix_feedback_notes_id"), "feedback", ["notes_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_feedback_notes_id"), table_name="feedback")
    op.drop_table("feedback")
