"""Initial migration: create scans and users tables

Revision ID: 001
Revises:
Create Date: 2026-05-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('username', sa.String(), unique=True, nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_users_username', 'users', ['username'], unique=True)

    op.create_table(
        'scans',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('scan_type', sa.String(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('disease_type', sa.String(), nullable=False),
        sa.Column('severity', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('spread_risk', sa.String(), nullable=True),
        sa.Column('treatment', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('full_results', sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('scans')
    op.drop_index('ix_users_username', 'users')
    op.drop_table('users')
