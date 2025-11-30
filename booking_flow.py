"""
app/booking_flow.py

Slot-filling booking flow with strict validations, DB persistence and email confirmation.
Adheres to assignment requirements (validation, short-term memory, DB/email error handling).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.database import SessionLocal, engine, Base
from db.models import Customer, Booking

import re
from typing import Tuple, Dict, Any, Optional
import datetime
import html
import string

import streamlit as st
from dateutil import parser as dateparser
from email.utils import parseaddr
from db.database import SessionLocal, engine
from db.models import Base, Customer, Booking

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

# ---------------- CONFIG / CONSTANTS ----------------
REQUIRED_SLOTS = ["name", "email", "phone", "destination", "date", "time"]
MAX_NAME_LEN = 100
MIN_NAME_LEN = 3
MAX_TEXT_LEN = 200
MIN_PHONE_DIGITS = 7
MAX_PHONE_DIGITS = 15

EMAIL_RE = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
PHONE_RE = re.compile(r"^\+?[\d\-\s\(\)]+$")


# ---------------- Utility validators ----------------
def _sanitize_text(s: str, max_len: int = MAX_TEXT_LEN) -> str:
    """Strip, collapse whitespace, remove control chars, limit length."""
    if not s:
        return ""
    s = "".join(ch for ch in s if ch in string.printable)
    s = " ".join(s.split())
    s = html.escape(s)
    if len(s) > max_len:
        s = s[:max_len]
    return s.strip()


def validate_name(name: str) -> Tuple[bool, str]:
    """Validate full name."""
    if not name or not name.strip():
        return False, "Name cannot be empty."
    name_clean = _sanitize_text(name, max_len=MAX_NAME_LEN)
    if len(name_clean) < MIN_NAME_LEN:
        return False, f"Name too short (min {MIN_NAME_LEN} characters)."
    return True, name_clean


def validate_email(email: str) -> Tuple[bool, str]:
    """Validate and normalize email."""
    if not email or not email.strip():
        return False, "Email cannot be empty."
    email = email.strip()
    if EMAIL_RE.match(email):
        return True, email.lower()
    parsed = parseaddr(email)[1]
    if parsed and EMAIL_RE.match(parsed):
        return True, parsed.lower()
    return False, "Invalid email format. Please provide a valid email (e.g., name@example.com)."


def validate_phone(phone: str) -> Tuple[bool, str]:
    """Normalize phone and validate digit counts."""
    if not phone or not phone.strip():
        return False, "Phone number cannot be empty."
    raw = phone.strip()
    plus = "+" if raw.startswith("+") else ""
    digits = re.sub(r"\D", "", raw)
    if len(digits) < MIN_PHONE_DIGITS or len(digits) > MAX_PHONE_DIGITS:
        return False, f"Phone number must have between {MIN_PHONE_DIGITS} and {MAX_PHONE_DIGITS} digits."
    normalized = plus + digits
    return True, normalized


def validate_destination(dest: str) -> Tuple[bool, str]:
    """Simple validation on destination/service type."""
    if not dest or not dest.strip():
        return False, "Booking/service type cannot be empty."
    clean = _sanitize_text(dest, max_len=MAX_TEXT_LEN)
    if len(clean) < 2:
        return False, "Provide a more descriptive destination or package name."
    return True, clean


def validate_date(date_text: str) -> Tuple[bool, str]:
    """Parse date and normalize to YYYY-MM-DD."""
    if not date_text or not date_text.strip():
        return False, "Date cannot be empty. Please enter as YYYY-MM-DD."
    date_text_str = date_text.strip()
    iso_match = re.match(r"^\d{4}-\d{2}-\d{2}$", date_text_str)
    try:
        if iso_match:
            dt = datetime.date.fromisoformat(date_text_str)
        else:
            dt = dateparser.parse(date_text_str, fuzzy=True).date()
    except Exception:
        return False, "Invalid date. Please enter date as YYYY-MM-DD (e.g., 2025-12-31)."
    today = datetime.date.today()
    if dt < today:
        return False, "Date cannot be in the past. Please enter a current or future date (YYYY-MM-DD)."
    return True, dt.isoformat()


def validate_time(time_text: str) -> Tuple[bool, str]:
    """Parse and normalize time to HH:MM 24-hour format."""
    if not time_text or not time_text.strip():
        return False, "Time cannot be empty. Please enter time as HH:MM (24-hour) or '2:30 PM'."
    t = time_text.strip()
    hhmm = re.match(r"^([01]?\d|2[0-3]):([0-5]\d)$", t)
    if hhmm:
        hh = int(hhmm.group(1))
        mm = int(hhmm.group(2))
        return True, f"{hh:02d}:{mm:02d}"
    try:
        dt = dateparser.parse(t, fuzzy=True)
        if dt is None:
            return False, "Could not parse time. Use HH:MM (24-hour) format."
        return True, dt.time().strftime("%H:%M")
    except Exception:
        return False, "Could not parse time. Use HH:MM (24-hour) format."


# ---------------- DB helpers ----------------
def _save_booking_to_db(slots: Dict[str, str]) -> Tuple[bool, str]:
    """Save booking to sqlite DB using SQLAlchemy."""
    db = SessionLocal()
    try:
        customer = None
        if slots.get("email"):
            customer = db.query(Customer).filter(Customer.email == slots["email"]).first()
        if not customer and slots.get("phone"):
            customer = db.query(Customer).filter(Customer.phone == slots["phone"]).first()

        if not customer:
            customer = Customer(
                name=slots.get("name"),
                email=slots.get("email"),
                phone=slots.get("phone"),
            )
            db.add(customer)
            db.flush()

        booking = Booking(
            customer_id=customer.customer_id,
            booking_type=slots.get("destination", "Travel Consultation"),
            date=slots.get("date"),
            time=slots.get("time"),
            status="CONFIRMED",
        )
        db.add(booking)
        db.commit()
        return True, str(booking.id)
    except Exception as e:
        db.rollback()
        print(f"[booking_flow] DB error: {e}")
        return False, f"Database error: {e}"
    finally:
        db.close()


# ---------------- Email helper ----------------
def _send_confirmation_email(to_email: str, slots: Dict[str, str]) -> Tuple[bool, str]:
    """Send confirmation email if SMTP configured."""
    try:
        from app.config import get_smtp_config
        smtp_conf = get_smtp_config()
    except Exception:
        smtp_conf = {}

    smtp_server = smtp_conf.get("smtp_server") or st.secrets.get("smtp", {}).get("server")
    smtp_port = int(smtp_conf.get("smtp_port") or st.secrets.get("smtp", {}).get("port", 465))
    username = smtp_conf.get("username") or st.secrets.get("smtp", {}).get("username")
    password = smtp_conf.get("password") or st.secrets.get("smtp", {}).get("password")
    from_email = smtp_conf.get("from_email") or st.secrets.get("smtp", {}).get("from_email")

    if not smtp_server or not username or not password:
        return False, "SMTP not configured."

    try:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = f"Booking Confirmation — {slots.get('destination')} on {slots.get('date')} {slots.get('time')}"
        msg["From"] = from_email or username
        msg["To"] = to_email
        body = (
            f"Hello {slots.get('name')},\n\n"
            f"Your booking has been confirmed.\n\n"
            f"Booking ID: {slots.get('_booking_id', 'N/A')}\n"
            f"Booking Type: {slots.get('destination')}\n"
            f"Date: {slots.get('date')}\n"
            f"Time: {slots.get('time')}\n\n"
            "If you have any queries, reply to this email.\n\n"
            "Regards,\nOceania Travel Team"
        )
        msg.set_content(body)
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp_conn:
            smtp_conn.login(username, password)
            smtp_conn.send_message(msg)
        return True, "Email sent."
    except Exception as e:
        print(f"[booking_flow] Email send error: {e}")
        return False, f"Email send failed: {e}"


# ---------------- Booking flow (main) ----------------
def _next_missing_slot(slots: Dict[str, str]) -> Optional[str]:
    for s in REQUIRED_SLOTS:
        if not slots.get(s):
            return s
    return None


def slot_prompt(slot_name: str) -> str:
    prompts = {
        "destination": "Which package would you like to book: Australia or New Zealand?",
        "name": "Please provide your full name (3–100 characters).",
        "email": "Please enter your email address (e.g., name@example.com).",
        "phone": "Please provide your phone number (include country code if outside your country).",
        "date": "Preferred date (YYYY-MM-DD).",
        "time": "Preferred time (HH:MM, 24-hour).",
    }
    return prompts.get(slot_name, f"Please provide {slot_name}.")


def confirm_summary(slots: Dict[str, str]) -> str:
    s = "Please confirm your booking details:\n\n"
    for k in REQUIRED_SLOTS:
        s += f"{k.capitalize()}: {slots.get(k, '')}\n"
    s += "\nReply 'yes' to confirm, 'no' to cancel, or say 'change <field>' to edit (e.g., 'change date')."
    return s


def booking_step(user_text: str, session_state: Dict[str, Any]) -> str:
    """
    Main slot-filling function. Returns a user-facing prompt/response.
    Strict multi-turn dialogue:
    - Ask if user wants to book.
    - Then destination (Australia/New Zealand), name, email, phone, date, time.
    """
    slots = session_state.setdefault("booking_slots", {})
    text = (user_text or "").strip()
    lower = text.lower()

    # ---- Initial confirmation: start booking? ----
    started_flag = session_state.get("booking_started", False)

    if not started_flag:
        if any(k in lower for k in ["book", "appointment", "reserve", "schedule", "consultation", "meeting", "trip"]):
            session_state["booking_started"] = "asked"
            return "You mentioned booking a trip. Do you want to start a booking now? Please reply 'yes' or 'no'."

    if session_state.get("booking_started") == "asked":
        if lower in ("yes", "y"):
            session_state["booking_started"] = True
            return slot_prompt("destination")
        elif lower in ("no", "n", "cancel", "stop"):
            session_state["booking_started"] = False
            session_state["booking_slots"] = {}
            return "Okay, booking cancelled. You can ask travel questions or say 'I want to book' any time."
        else:
            return "Please reply 'yes' to start a booking or 'no' to cancel."

    # If booking not started, let caller handle other intents
    if not session_state.get("booking_started"):
        return "If you would like to book a package, please say something like 'I want to book a trip'."

    # ---- Active booking: strict ordered slots ----

    # Allow user to change a field
    m = re.search(r"change\s+(\w+)", lower)
    if m:
        field = m.group(1)
        if field in REQUIRED_SLOTS:
            slots.pop(field, None)
            session_state["booking_slots"] = slots
            return slot_prompt(field)
        else:
            return f"Unknown field '{field}'. You can change: {', '.join(REQUIRED_SLOTS)}."

    # 1) Destination (Australia / New Zealand only)
    if not slots.get("destination"):
        dest = None
        if "australia" in lower:
            dest = "Australia"
        elif "new zealand" in lower or "newzealand" in lower:
            dest = "New Zealand"

        if dest:
            ok, val = validate_destination(dest)
            if not ok:
                return val
            slots["destination"] = val
        else:
            session_state["booking_slots"] = slots
            return slot_prompt("destination")

        # after setting destination, next ask for name
        session_state["booking_slots"] = slots
        return slot_prompt("name")

    # 2) Name
    if not slots.get("name"):
        ok, val = validate_name(text)
        if not ok:
            return slot_prompt("name")
        slots["name"] = val
        session_state["booking_slots"] = slots
        return slot_prompt("email")

    # 3) Email
    if not slots.get("email"):
        ok, val = validate_email(text)
        if not ok:
            return val
        slots["email"] = val
        session_state["booking_slots"] = slots
        return slot_prompt("phone")

    # 4) Phone
    if not slots.get("phone"):
        ok, val = validate_phone(text)
        if not ok:
            return val
        slots["phone"] = val
        session_state["booking_slots"] = slots
        return slot_prompt("date")

    # 5) Date
    if not slots.get("date"):
        ok, val = validate_date(text)
        if not ok:
            return val
        slots["date"] = val
        session_state["booking_slots"] = slots
        return slot_prompt("time")

    # 6) Time
    if not slots.get("time"):
        ok, val = validate_time(text)
        if not ok:
            return val
        slots["time"] = val
        session_state["booking_slots"] = slots
        return confirm_summary(slots)

    # ---- Confirmation / cancellation after summary ----
    if lower in ("yes", "y", "confirm", "confirmed"):
        validators = {
            "name": validate_name,
            "email": validate_email,
            "phone": validate_phone,
            "destination": validate_destination,
            "date": validate_date,
            "time": validate_time,
        }
        for k, fn in validators.items():
            ok, val = fn(slots.get(k))
            if not ok:
                return f"Invalid {k}: {val}"
            slots[k] = val

        ok, res = _save_booking_to_db(slots)
        if not ok:
            return f"Failed to save booking: {res}"

        slots["_booking_id"] = res
        email_ok, email_msg = _send_confirmation_email(slots["email"], slots)
        session_state["booking_slots"] = {}
        session_state["booking_started"] = False

        if email_ok:
            return f"Booking confirmed (ID: {res}). Confirmation email sent to {slots['email']}."
        else:
            return f"Booking confirmed (ID: {res}). Email not sent: {email_msg}. The booking is saved."

    if lower in ("no", "cancel", "stop"):
        session_state["booking_slots"] = {}
        session_state["booking_started"] = False
        return "Booking cancelled. Let me know if you'd like to start again."

    # If user says something else after summary, show summary again
    session_state["booking_slots"] = slots
    return confirm_summary(slots)
