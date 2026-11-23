"""
Authentication endpoints — register, login, token refresh.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from ...core.auth import (
    Token,
    UserIn,
    UserOut,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from ...db.database import get_db
from ...db.models import User

router = APIRouter()


@router.post("/auth/register", response_model=UserOut)
async def register(user_in: UserIn, db=Depends(get_db)):
    """
    Register a new user. Returns 409 if username already taken.
    """
    existing = await db.execute(
        select(User).where(User.username == user_in.username)
    )
    if existing.scalars().first():
        raise HTTPException(status_code=409, detail="Username already taken.")

    user = User(
        username=user_in.username,
        hashed_password=hash_password(user_in.password),
    )
    db.add(user)
    await db.commit()
    return UserOut(username=user.username)


@router.post("/auth/token", response_model=Token)
async def login(user_in: UserIn, db=Depends(get_db)):
    """
    Authenticate with username + password, return a JWT Bearer token.
    """
    result = await db.execute(
        select(User).where(User.username == user_in.username)
    )
    user = result.scalars().first()

    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    access_token = create_access_token(data={"sub": user.username})
    return Token(access_token=access_token)


@router.get("/auth/me", response_model=UserOut)
async def get_me(username: str = Depends(get_current_user)):
    """Return the currently authenticated user."""
    return UserOut(username=username or "anonymous")
