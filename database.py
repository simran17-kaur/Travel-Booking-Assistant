# db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite local file relative to project root
SQLITE_URL = "sqlite:///./app.db"

# For sqlite on Windows, check_same_thread must be False
engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# shared Base for models
Base = declarative_base()
