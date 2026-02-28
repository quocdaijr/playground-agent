"""add_session_metadata

Revision ID: 98f0f866daf5
Revises: c3eed1dc9e56
Create Date: 2026-02-28
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "98f0f866daf5"
down_revision = "c3eed1dc9e56"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add user_agent / arbitrary metadata to sessions
    op.add_column("sessions", sa.Column("meta", JSONB(), nullable=True))
    # Add token_count tracking to messages
    op.add_column("messages", sa.Column("token_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "token_count")
    op.drop_column("sessions", "meta")
