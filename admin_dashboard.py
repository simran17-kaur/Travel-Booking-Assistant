# app/admin_dashboard.py
"""
Streamlit admin dashboard for the Travel Booking Assistant.


Features:
- Summary counts (customers, bookings)
- Paginated, filterable booking table (joined with customer info)
- Delete booking (with confirmation)
- Export bookings to CSV
- Defensive DB handling and helpful error messages
"""


from typing import List
import io
import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session
from sqlalchemy import select


from db.database import SessionLocal, engine
from db.models import Customer, Booking


# small helper: open DB session
def get_db() -> Session:
    return SessionLocal()


def fetch_summary(db: Session):
    try:
        customer_count = db.query(Customer).count()
        booking_count = db.query(Booking).count()
    except Exception as e:
        st.error(f"DB error while fetching counts: {e}")
        return 0, 0
    return customer_count, booking_count


def fetch_bookings(db: Session, search: str = "", status: str = "", limit: int = 200):
    """
    Returns list of dict rows joining bookings <> customers
    """
    try:
        q = db.query(Booking, Customer).join(Customer, Booking.customer_id == Customer.customer_id)
        if status:
            q = q.filter(Booking.status == status)
        if search:
            like = f"%{search}%"
            q = q.filter(
                (Customer.name.ilike(like)) |
                (Customer.email.ilike(like)) |
                (Customer.phone.ilike(like)) |
                (Booking.booking_type.ilike(like))
            )
        q = q.order_by(Booking.id.desc()).limit(limit)
        rows = []
        for b, c in q.all():
            rows.append({
                "booking_id": b.id,
                "customer_id": c.customer_id,
                "name": c.name,
                "email": c.email,
                "phone": c.phone,
                "booking_type": b.booking_type,
                "date": b.date,
                "time": b.time,
                "status": b.status
            })
        return rows
    except Exception as e:
        st.error(f"DB error while fetching bookings: {e}")
        return []


def delete_booking(db: Session, booking_id: int) -> bool:
    try:
        booking = db.query(Booking).filter(Booking.id == booking_id).first()
        if not booking:
            return False
        db.delete(booking)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        st.error(f"DB error deleting booking: {e}")
        return False


def bookings_to_csv(rows: List[dict]) -> bytes:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def admin_page():
    st.title("🔒 Admin Dashboard")
    st.write("Overview of customers and bookings. Actions: delete bookings, export CSV.")

    db = get_db()

    # summary
    customer_count, booking_count = fetch_summary(db)
    col1, col2 = st.columns(2)
    col1.metric("Customers", customer_count)
    col2.metric("Bookings", booking_count)

    st.markdown("---")

    # Filters
    with st.form("filters", clear_on_submit=False):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            search = st.text_input("Search (name, email, phone, booking type)", value="")
        with c2:
            status = st.selectbox("Status", options=["", "CONFIRMED", "CANCELLED", "PENDING"])
        with c3:
            limit = st.number_input("Max rows", min_value=10, max_value=2000, value=200, step=10)
        submitted = st.form_submit_button("Apply")

    # fetch rows
    rows = fetch_bookings(db, search=search, status=status, limit=int(limit))

    st.write(f"Showing {len(rows)} booking(s).")
    if not rows:
        st.info("No bookings found for the selected filters.")
    else:
        df = pd.DataFrame(rows)
        # show table
        st.dataframe(df, use_container_width=True)

        # row actions
        st.markdown("### Actions")
        c1, c2 = st.columns([2, 2])
        with c1:
            # delete booking
            booking_to_delete = st.number_input("Booking ID to delete", min_value=0, value=0, step=1)
            if st.button("Delete booking"):
                if booking_to_delete <= 0:
                    st.warning("Enter a valid booking id to delete.")
                else:
                    # FIX: capture checkbox result in a variable instead of using walrus on st.confirm
                    confirm_delete = st.checkbox(f"Confirm delete booking {booking_to_delete}")
                    if confirm_delete:
                        ok = delete_booking(db, int(booking_to_delete))
                        if ok:
                            st.success(f"Booking {booking_to_delete} deleted.")
                            # refresh
                            rows = fetch_bookings(db, search=search, status=status, limit=int(limit))
                        else:
                            st.error(f"Could not delete booking {booking_to_delete}.")
        with c2:
            csv_bytes = bookings_to_csv(rows)
            st.download_button("Export CSV", data=csv_bytes, file_name="bookings_export.csv", mime="text/csv")

    # Close DB session
    try:
        db.close()
    except Exception:
        pass


if __name__ == "__main__":
    # allow running directly for quick checks
    admin_page()