"""
Demo User Seeding Script.

Initializes the users database in data/app.db and creates three demo users:
1. manager (Role: 'manager', Status: 'approved')
2. front_desk (Role: 'front_desk', Status: 'approved')
3. pending_trainee (Role: 'front_desk', Status: 'pending' - for video demo approval flow)

Passwords are read from environment variables or src.config defaults,
and stored strictly as bcrypt hashes in SQLite.
"""

from pathlib import Path
import sys

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auth import init_user_db, save_seeded_user
from src.config import (
    APP_DB_PATH,
    DEMO_FRONT_DESK_EMAIL,
    DEMO_FRONT_DESK_NAME,
    DEMO_FRONT_DESK_PASSWORD,
    DEMO_MANAGER_EMAIL,
    DEMO_MANAGER_NAME,
    DEMO_MANAGER_PASSWORD,
    DEMO_PENDING_EMAIL,
    DEMO_PENDING_NAME,
    DEMO_PENDING_PASSWORD,
)


def seed_demo_users():
    """Seed manager, front_desk, and pending trainee accounts with bcrypt password hashes."""
    print("Initializing user database schema in data/app.db...")
    init_user_db()

    print(f"Creating approved manager: '{DEMO_MANAGER_EMAIL}'...")
    save_seeded_user(
        name=DEMO_MANAGER_NAME,
        email=DEMO_MANAGER_EMAIL,
        plain_password=DEMO_MANAGER_PASSWORD,
        role="manager",
        status="approved",
    )

    print(f"Creating approved front desk: '{DEMO_FRONT_DESK_EMAIL}'...")
    save_seeded_user(
        name=DEMO_FRONT_DESK_NAME,
        email=DEMO_FRONT_DESK_EMAIL,
        plain_password=DEMO_FRONT_DESK_PASSWORD,
        role="front_desk",
        status="approved",
    )

    print(f"Creating pending user for approval demo: '{DEMO_PENDING_EMAIL}' (status='pending')...")
    save_seeded_user(
        name=DEMO_PENDING_NAME,
        email=DEMO_PENDING_EMAIL,
        plain_password=DEMO_PENDING_PASSWORD,
        role="front_desk",
        status="pending",
    )

    print(f"\nUser seeding complete! Database stored at: {APP_DB_PATH}")
    print("================================================================================")
    print("DEMO CREDENTIALS FOR TESTING & VIDEO WALKTHROUGH:")
    print("--------------------------------------------------------------------------------")
    print(f"1. Manager (Approved):    Email='{DEMO_MANAGER_EMAIL}', Password='{DEMO_MANAGER_PASSWORD}' [Role: manager, Status: approved]")
    print(f"2. Front Desk (Approved): Email='{DEMO_FRONT_DESK_EMAIL}', Password='{DEMO_FRONT_DESK_PASSWORD}' [Role: front_desk, Status: approved]")
    print(f"3. Trainee (Pending):     Email='{DEMO_PENDING_EMAIL}', Password='{DEMO_PENDING_PASSWORD}' [Role: front_desk, Status: pending]")
    print("================================================================================")


if __name__ == "__main__":
    seed_demo_users()
