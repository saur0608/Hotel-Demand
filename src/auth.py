"""
Authentication, Registration & Role-Based Access Control (RBAC) Module.

Provides:
1. SQLite storage in data/app.db: id, name, email (unique), password_hash (bcrypt),
   role, status (pending/approved/rejected), created_at.
2. Registration validation: email format, password >= 8 chars, confirm-password match,
   forced role='front_desk' and status='pending'.
3. Rate-limited login (5 failed attempts per minute per email/IP).
4. Status enforcement: only status='approved' accounts receive JWT tokens.
5. Manager-only admin workflows (approve/reject access requests).
6. Security dependencies: get_current_user (401) and require_role (403).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import re
import sqlite3
import bcrypt
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from src.config import (
    APP_DB_PATH,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
)

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

# In-memory sliding window rate limiter: key -> list of timestamp floats
_failed_login_attempts: Dict[str, List[float]] = {}


# ==============================================================================
# 1. DATABASE MANAGEMENT (SQLite: data/app.db)
# ==============================================================================
def get_db_connection() -> sqlite3.Connection:
    """Connect to SQLite database."""
    APP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db() -> None:
    """Initialize the users table schema if not already present."""
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'front_desk',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.close()


# ==============================================================================
# 2. PASSWORD HASHING (BCRYPT)
# ==============================================================================
def hash_password(password: str) -> str:
    """Hash password using bcrypt with a salt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verify plain password against bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


# ==============================================================================
# 3. VALIDATION HELPERS
# ==============================================================================
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email_format(email: str) -> bool:
    """Check standard RFC email pattern."""
    if not email:
        return False
    return bool(EMAIL_REGEX.match(email.strip()))


# ==============================================================================
# 4. USER OPERATIONS (REGISTRATION, RETRIEVAL, ADMIN APPROVAL)
# ==============================================================================
def register_user(
    name: str,
    email: str,
    password: str,
    confirm_password: str,
) -> Dict:
    """
    Public registration endpoint logic:
    - Validates email format.
    - Enforces password >= 8 characters.
    - Enforces password == confirm_password.
    - Forces role='front_desk' and status='pending'. The client can NEVER choose role!
    - Enforces unique email.
    """
    init_user_db()
    name_clean = name.strip()
    email_clean = email.strip().lower()

    if not name_clean:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    if not validate_email_format(email_clean):
        raise HTTPException(status_code=400, detail="Invalid email address format.")

    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")

    if password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    # Check email uniqueness
    existing = get_user_by_email(email_clean)
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    pw_hash = hash_password(password)
    # Strict server-side assignment: role is ALWAYS front_desk, status ALWAYS pending
    role_assigned = "front_desk"
    status_assigned = "pending"

    conn = get_db_connection()
    with conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role, status)
            VALUES (?, ?, ?, ?, ?)
        """, (name_clean, email_clean, pw_hash, role_assigned, status_assigned))
        user_id = cursor.lastrowid
    conn.close()

    logger.info("Registered new user '%s' (id=%d, role=%s, status=%s)", email_clean, user_id, role_assigned, status_assigned)
    return {
        "id": user_id,
        "name": name_clean,
        "email": email_clean,
        "role": role_assigned,
        "status": status_assigned,
        "message": "Registration submitted successfully. Your account is pending manager approval.",
    }


def save_seeded_user(
    name: str,
    email: str,
    plain_password: str,
    role: str = "front_desk",
    status: str = "approved",
) -> Dict:
    """Internal helper for scripts/seed_users.py to create demo accounts."""
    init_user_db()
    email_clean = email.strip().lower()
    pw_hash = hash_password(plain_password)

    conn = get_db_connection()
    with conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (name, email, password_hash, role, status)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name = excluded.name,
                password_hash = excluded.password_hash,
                role = excluded.role,
                status = excluded.status
        """, (name.strip(), email_clean, pw_hash, role.strip().lower(), status.strip().lower()))
        user_id = cursor.lastrowid
    conn.close()
    return {"id": user_id, "name": name, "email": email_clean, "role": role, "status": status}


def get_user_by_email(email: str) -> Optional[Dict]:
    """Fetch user by email."""
    init_user_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, password_hash, role, status, created_at FROM users WHERE email = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Fetch user by primary key ID."""
    init_user_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, password_hash, role, status, created_at FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_all_users() -> List[Dict]:
    """Retrieve all user accounts (for manager administration)."""
    init_user_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, role, status, created_at FROM users ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_user_status(user_id: int, new_status: str) -> Optional[Dict]:
    """Approve or reject a user access request."""
    init_user_db()
    valid_statuses = ["approved", "rejected", "pending"]
    if new_status not in valid_statuses:
        raise ValueError(f"Invalid status: {new_status}. Allowed: {valid_statuses}")

    conn = get_db_connection()
    with conn:
        conn.execute("UPDATE users SET status = ? WHERE id = ?", (new_status, user_id))
    conn.close()
    return get_user_by_id(user_id)


# ==============================================================================
# 5. RATE LIMITING (5 failed attempts per minute)
# ==============================================================================
def check_rate_limit(key: str, max_attempts: int = 5, window_seconds: int = 60) -> None:
    """
    Check if IP or email key has exceeded failed attempts threshold in the last window_seconds.
    Raises HTTP 429 Too Many Requests if exceeded.
    """
    now = datetime.now(timezone.utc).timestamp()
    attempts = _failed_login_attempts.get(key, [])
    # Keep attempts within the sliding window
    valid_attempts = [ts for ts in attempts if now - ts < window_seconds]
    _failed_login_attempts[key] = valid_attempts

    if len(valid_attempts) >= max_attempts:
        logger.warning("Rate limit exceeded for '%s' (%d failed attempts in %ds)", key, len(valid_attempts), window_seconds)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Please wait {window_seconds} seconds before trying again.",
        )


def record_failed_attempt(key: str) -> None:
    """Record a failed login attempt timestamp."""
    now = datetime.now(timezone.utc).timestamp()
    if key not in _failed_login_attempts:
        _failed_login_attempts[key] = []
    _failed_login_attempts[key].append(now)


def reset_failed_attempts(key: str) -> None:
    """Clear failed attempts on successful login."""
    if key in _failed_login_attempts:
        _failed_login_attempts.pop(key, None)


# ==============================================================================
# 6. JWT TOKENS & AUTHENTICATION
# ==============================================================================
def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create signed JWT access token with 60-minute expiration."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict:
    """Decode and validate a JWT access token."""
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token has expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def authenticate_user(email: str, plain_password: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Authenticate email and password.
    Returns:
    - (user, None) if password is valid AND status == 'approved'
    - (None, 'invalid_credentials') if user does not exist or password mismatch
    - (user, 'pending') if password is valid but account is still pending approval
    - (user, 'rejected') if password is valid but account was rejected
    """
    user = get_user_by_email(email)
    if not user:
        return None, "invalid_credentials"

    if not verify_password(plain_password, user["password_hash"]):
        return None, "invalid_credentials"

    if user["status"] == "pending":
        return user, "pending"

    if user["status"] == "rejected":
        return user, "rejected"

    return user, None


# ==============================================================================
# 7. FASTAPI DEPENDENCIES (JWT AUTH & RBAC)
# ==============================================================================
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Dict:
    """
    FastAPI dependency: extracts and validates the Bearer token.
    Enforces that user exists and is approved. Raises HTTP 401 Unauthorized otherwise.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_access_token(token)
    email = payload.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload: missing subject claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user["status"] != "approved":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is not approved (current status: '{user['status']}').",
        )

    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "status": user["status"],
    }


def require_role(allowed_roles: List[str]):
    """
    RBAC Dependency: checks user role against allowed roles.
    Raises HTTP 403 Forbidden if unauthorized.
    """
    def role_checker(current_user: Dict = Depends(get_current_user)) -> Dict:
        user_role = current_user.get("role", "").lower()
        allowed = [r.lower() for r in allowed_roles]
        if user_role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Operation requires role in {allowed_roles}. Current role: '{user_role}'.",
            )
        return current_user

    return role_checker
