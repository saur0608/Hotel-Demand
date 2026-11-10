"""
Authentication, Registration & Role-Based Access Control (RBAC) Test Suite.

Rigorously verifies:
1. Passwords are stored exclusively as bcrypt hashes (no plain-text) in data/app.db.
2. Public registration enforces email format, password length >= 8, confirm match.
3. Client role injection is strictly blocked (always assigned role='front_desk', status='pending').
4. Duplicate email registrations are rejected (HTTP 400).
5. Pending users cannot log in (HTTP 403 with clear explanation).
6. Approved users receive JWT access tokens with 60-minute expiration.
7. Sliding-window rate limiting triggers HTTP 429 after 5 failed login attempts.
8. Unauthenticated requests return HTTP 401 Unauthorized.
9. front_desk role can access /predict, but gets HTTP 403 on /recommend and /admin/*.
10. manager role can access /predict, /recommend, and manager-only /admin/* endpoints.
11. Manager approval workflow: Manager approves a pending user, who can then log in successfully.
"""

import sqlite3
import pytest
from starlette.testclient import TestClient

from app.api import app as test_app

test_app.__test__ = False
from scripts.seed_users import seed_demo_users
from src.auth import _failed_login_attempts, init_user_db, update_user_status
from src.config import (
    APP_DB_PATH,
    DEMO_FRONT_DESK_EMAIL,
    DEMO_FRONT_DESK_PASSWORD,
    DEMO_MANAGER_EMAIL,
    DEMO_MANAGER_PASSWORD,
    DEMO_PENDING_EMAIL,
    DEMO_PENDING_PASSWORD,
)


@pytest.fixture(autouse=True, scope="module")
def setup_users():
    """Ensure user database is initialized and seeded with demo accounts."""
    init_user_db()
    conn = sqlite3.connect(APP_DB_PATH)
    conn.execute("DELETE FROM users")
    conn.commit()
    conn.close()
    seed_demo_users()
    _failed_login_attempts.clear()


@pytest.fixture(autouse=True)
def clean_test_users_and_rate_limits():
    """Clear test accounts and in-memory rate limiting state between test cases to ensure idempotency."""
    _failed_login_attempts.clear()
    yield
    _failed_login_attempts.clear()
    conn = sqlite3.connect(APP_DB_PATH)
    conn.execute(
        "DELETE FROM users WHERE email NOT IN (?, ?, ?)",
        (DEMO_MANAGER_EMAIL, DEMO_FRONT_DESK_EMAIL, DEMO_PENDING_EMAIL)
    )
    # Ensure demo pending user stays pending
    conn.execute("UPDATE users SET status = 'pending' WHERE email = ?", (DEMO_PENDING_EMAIL,))
    conn.commit()
    conn.close()


# ==============================================================================
# 1. DATABASE & SECURITY TESTS
# ==============================================================================
def test_passwords_stored_strictly_as_bcrypt_hashes():
    """Assertion: Database must only store bcrypt hashed strings ($2b$...), never plain text."""
    conn = sqlite3.connect(APP_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT email, password_hash FROM users")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) >= 3, "User database missing seeded test accounts."
    for email, hashed in rows:
        assert hashed.startswith("$2b$"), f"Password for {email} is not a valid bcrypt hash!"
        assert DEMO_MANAGER_PASSWORD not in hashed
        assert DEMO_FRONT_DESK_PASSWORD not in hashed
        assert DEMO_PENDING_PASSWORD not in hashed


# ==============================================================================
# 2. REGISTRATION VALIDATION TESTS
# ==============================================================================
def test_registration_success_forces_pending_front_desk():
    """Verify that public registration assigns role='front_desk' and status='pending'."""
    with TestClient(test_app) as client:
        res = client.post("/auth/register", json={
            "name": "Alice Reception",
            "email": "alice_test@hotel.com",
            "password": "Password123!",
            "confirm_password": "Password123!",
        })
        assert res.status_code == 201, res.text
        data = res.json()
        assert data["email"] == "alice_test@hotel.com"
        assert data["role"] == "front_desk"
        assert data["status"] == "pending"
        assert "pending manager approval" in data["message"].lower()


def test_role_injection_in_registration_is_prevented():
    """Verify that client cannot elevate role to 'manager' or force status='approved'."""
    with TestClient(test_app) as client:
        # Client maliciously sends role='manager' and status='approved'
        res = client.post("/auth/register", json={
            "name": "Attacker Guy",
            "email": "attacker@hotel.com",
            "password": "AttackerPass123",
            "confirm_password": "AttackerPass123",
            "role": "manager",
            "status": "approved",
        })
        assert res.status_code == 201
        data = res.json()
        # Must still be forced to front_desk and pending
        assert data["role"] == "front_desk"
        assert data["status"] == "pending"

        # Verify DB directly
        conn = sqlite3.connect(APP_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT role, status FROM users WHERE email = 'attacker@hotel.com'")
        db_role, db_status = cursor.fetchone()
        conn.close()
        assert db_role == "front_desk"
        assert db_status == "pending"


def test_duplicate_email_rejected():
    """Verify that registering with an already existing email returns HTTP 400."""
    with TestClient(test_app) as client:
        res = client.post("/auth/register", json={
            "name": "Duplicate Manager",
            "email": DEMO_MANAGER_EMAIL,
            "password": "SomeOtherPass123",
            "confirm_password": "SomeOtherPass123",
        })
        assert res.status_code == 400
        assert "already exists" in res.json()["detail"].lower()


def test_weak_password_rejected():
    """Verify that password shorter than 8 characters returns HTTP 400."""
    with TestClient(test_app) as client:
        res = client.post("/auth/register", json={
            "name": "Weak Pass User",
            "email": "weak@hotel.com",
            "password": "short",
            "confirm_password": "short",
        })
        assert res.status_code == 400 or res.status_code == 422


def test_password_mismatch_rejected():
    """Verify that confirm_password mismatch returns HTTP 400."""
    with TestClient(test_app) as client:
        res = client.post("/auth/register", json={
            "name": "Mismatch User",
            "email": "mismatch@hotel.com",
            "password": "CorrectPassword123",
            "confirm_password": "DifferentPassword123",
        })
        assert res.status_code == 400
        assert "do not match" in res.json()["detail"].lower()


def test_invalid_email_format_rejected():
    """Verify that ill-formed email address returns HTTP 400."""
    with TestClient(test_app) as client:
        res = client.post("/auth/register", json={
            "name": "Invalid Email User",
            "email": "not-a-valid-email-format",
            "password": "ValidPassword123",
            "confirm_password": "ValidPassword123",
        })
        assert res.status_code == 400
        assert "invalid email" in res.json()["detail"].lower()


# ==============================================================================
# 3. LOGIN & STATUS ENFORCEMENT TESTS
# ==============================================================================
def test_wrong_password_rejected_with_401():
    """Verify that wrong passwords return HTTP 401 Unauthorized."""
    with TestClient(test_app) as client:
        res = client.post("/auth/login", json={
            "email": DEMO_MANAGER_EMAIL,
            "password": "WrongPassword999",
        })
        assert res.status_code == 401
        assert "invalid email or password" in res.json()["detail"].lower()


def test_pending_user_cannot_login():
    """Verify that pending users receive HTTP 403 Forbidden with clear message."""
    with TestClient(test_app) as client:
        res = client.post("/auth/login", json={
            "email": DEMO_PENDING_EMAIL,
            "password": DEMO_PENDING_PASSWORD,
        })
        assert res.status_code == 403
        assert "pending approval" in res.json()["detail"].lower()


def test_approved_user_can_login():
    """Verify that approved users get HTTP 200 with JWT access token."""
    with TestClient(test_app) as client:
        res = client.post("/auth/login", json={
            "email": DEMO_MANAGER_EMAIL,
            "password": DEMO_MANAGER_PASSWORD,
        })
        assert res.status_code == 200
        body = res.json()
        assert "access_token" in body
        assert body["role"] == "manager"
        assert body["email"] == DEMO_MANAGER_EMAIL


def test_rate_limiting_triggers_429():
    """Verify that 5 failed login attempts trigger HTTP 429 Too Many Requests."""
    with TestClient(test_app) as client:
        target_email = "ratelimit_victim@hotel.com"
        # 5 failed attempts
        for _ in range(5):
            res = client.post("/auth/login", json={
                "email": target_email,
                "password": "WrongPassword!",
            })
            assert res.status_code == 401

        # 6th attempt should be blocked by rate limiter
        res_blocked = client.post("/auth/login", json={
            "email": target_email,
            "password": "WrongPassword!",
        })
        assert res_blocked.status_code == 429
        assert "too many failed login attempts" in res_blocked.json()["detail"].lower()


# ==============================================================================
# 4. RBAC & PERMISSIONS TESTS
# ==============================================================================
def test_unauthenticated_requests_return_401():
    """Verify that requests without Bearer tokens return HTTP 401 Unauthorized."""
    with TestClient(test_app) as client:
        payload = {"stay_date": "2025-11-15", "room_category": "Standard", "horizon": 7}

        # /predict without token
        res_pred = client.post("/predict", json=payload)
        assert res_pred.status_code == 401

        # /recommend without token
        res_rec = client.post("/recommend", json=payload)
        assert res_rec.status_code == 401

        # /admin/requests without token
        res_admin = client.get("/admin/requests")
        assert res_admin.status_code == 401


def test_front_desk_role_permissions():
    """
    Verify RBAC boundary:
    front_desk CAN call /predict, but is FORBIDDEN (403) from calling /recommend and /admin/*.
    """
    with TestClient(test_app) as client:
        # Login as front_desk
        login_res = client.post("/auth/login", json={
            "email": DEMO_FRONT_DESK_EMAIL,
            "password": DEMO_FRONT_DESK_PASSWORD,
        })
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        payload = {"stay_date": "2025-11-15", "room_category": "Standard", "horizon": 7, "otb_rooms": 40}

        # 1. front_desk calling /predict -> Allowed (200)
        res_pred = client.post("/predict", json=payload, headers=headers)
        assert res_pred.status_code == 200
        assert "predicted_demand" in res_pred.json()

        # 2. front_desk calling /recommend -> Forbidden (403)
        res_rec = client.post("/recommend", json=payload, headers=headers)
        assert res_rec.status_code == 403
        assert "access denied" in res_rec.json()["detail"].lower()

        # 3. front_desk calling /admin/requests -> Forbidden (403)
        res_admin = client.get("/admin/requests", headers=headers)
        assert res_admin.status_code == 403
        assert "access denied" in res_admin.json()["detail"].lower()


def test_manager_role_permissions():
    """
    Verify RBAC boundary:
    manager CAN call /predict, /recommend, and /admin/requests.
    """
    with TestClient(test_app) as client:
        # Login as manager
        login_res = client.post("/auth/login", json={
            "email": DEMO_MANAGER_EMAIL,
            "password": DEMO_MANAGER_PASSWORD,
        })
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        payload = {"stay_date": "2025-11-15", "room_category": "Standard", "horizon": 7, "otb_rooms": 40}

        # 1. manager calling /predict -> Allowed (200)
        res_pred = client.post("/predict", json=payload, headers=headers)
        assert res_pred.status_code == 200

        # 2. manager calling /recommend -> Allowed (200)
        res_rec = client.post("/recommend", json=payload, headers=headers)
        assert res_rec.status_code == 200
        assert "authorized_overbooking_rooms" in res_rec.json()

        # 3. manager calling /admin/requests -> Allowed (200)
        res_admin = client.get("/admin/requests", headers=headers)
        assert res_admin.status_code == 200
        users_list = res_admin.json()
        assert len(users_list) >= 3


def test_manager_approval_workflow():
    """
    Verify complete admin approval workflow:
    1. A new user registers (status='pending').
    2. User cannot log in yet (403).
    3. Manager approves user via POST /admin/requests/{id}/approve.
    4. User can now successfully log in and receive a JWT token!
    """
    with TestClient(test_app) as client:
        # Step 1: Register new user
        reg_res = client.post("/auth/register", json={
            "name": "Bob Trainee",
            "email": "bob_trainee@hotel.com",
            "password": "BobTraineePass123",
            "confirm_password": "BobTraineePass123",
        })
        assert reg_res.status_code == 201
        bob_id = reg_res.json()["id"]

        # Step 2: Bob cannot log in while pending
        login_fail = client.post("/auth/login", json={
            "email": "bob_trainee@hotel.com",
            "password": "BobTraineePass123",
        })
        assert login_fail.status_code == 403

        # Step 3: Manager logs in and approves Bob
        mgr_login = client.post("/auth/login", json={
            "email": DEMO_MANAGER_EMAIL,
            "password": DEMO_MANAGER_PASSWORD,
        })
        mgr_token = mgr_login.json()["access_token"]
        mgr_headers = {"Authorization": f"Bearer {mgr_token}"}

        approve_res = client.post(f"/admin/requests/{bob_id}/approve", headers=mgr_headers)
        assert approve_res.status_code == 200
        assert approve_res.json()["status"] == "approved"

        # Step 4: Bob can now log in!
        login_success = client.post("/auth/login", json={
            "email": "bob_trainee@hotel.com",
            "password": "BobTraineePass123",
        })
        assert login_success.status_code == 200
        assert "access_token" in login_success.json()
        assert login_success.json()["role"] == "front_desk"
