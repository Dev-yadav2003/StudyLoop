"""unit edits: units.edited_at, notes.revision

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-18 10:00:00.000000+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("units", sa.Column("edited_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("notes", sa.Column("revision", sa.Integer(), server_default="0", nullable=False))


def downgrade() -> None:
    op.drop_column("notes", "revision")
    op.drop_column("units", "edited_at")
