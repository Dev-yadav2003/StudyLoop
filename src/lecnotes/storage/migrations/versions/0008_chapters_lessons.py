"""chapters and lessons

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-18 15:00:00.000000+00:00
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[Any]]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "chapters",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), server_default="0", nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chapters")),
    )
    op.create_table(
        "lessons",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chapter_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("position", sa.Integer(), server_default="0", nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["chapter_id"],
            ["chapters.id"],
            name=op.f("fk_lessons_chapter_id_chapters"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name=op.f("fk_lessons_job_id_jobs")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lessons")),
    )
    op.create_index(op.f("ix_lessons_chapter_id"), "lessons", ["chapter_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_lessons_chapter_id"), table_name="lessons")
    op.drop_table("lessons")
    op.drop_table("chapters")
