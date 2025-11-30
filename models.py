# db/models.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from db.database import Base  # import Base from database.py (no circular import)

class Customer(Base):
    __tablename__ = "customers"

    # booking_flow expects customer.customer_id
    customer_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=True)
    email = Column(String(200), unique=True, index=True, nullable=True)
    phone = Column(String(50), unique=False, index=True, nullable=True)

    bookings = relationship("Booking", back_populates="customer")


class Booking(Base):
    __tablename__ = "bookings"

    # booking_flow expects booking.id
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), nullable=False)
    booking_type = Column(String(200), nullable=True)   # e.g., "Australia 7-day"
    date = Column(String(20), nullable=True)            # stored as ISO YYYY-MM-DD
    time = Column(String(10), nullable=True)            # stored as HH:MM
    status = Column(String(50), nullable=True)          # e.g., "CONFIRMED"

    customer = relationship("Customer", back_populates="bookings")
