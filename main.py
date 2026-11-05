
import os
import uuid
import hashlib
import base64
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    create_engine, Column, String, DateTime, Boolean,
    ForeignKey, Integer, select, delete, update
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from starlette.responses import JSONResponse


# CONFIGURATION

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./clinic.db")
API_KEY = os.getenv("API_KEY", "clinic-dev-key-change-me")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


# DATABASE TABLES

class Slot(Base):
    __tablename__ = "slots"

    id = Column(String, primary_key=True)
    provider = Column(String, nullable=False)
    start = Column(String, nullable=False)
    end = Column(String, nullable=False)
    available = Column(Boolean, default=True, nullable=False)


class Booking(Base):
    __tablename__ = "bookings"

    id = Column(String, primary_key=True)
    slot_id = Column(String, ForeignKey("slots.id"), nullable=False)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, index=True)
    status = Column(String, default="confirmed", nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"

    key = Column(String, primary_key=True)
    request_hash = Column(String, nullable=False)
    booking_id = Column(String, nullable=False)


class BookingAttempt(Base):
    __tablename__ = "booking_attempts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, nullable=False, index=True)
    attempted_at = Column(DateTime, nullable=False)


Base.metadata.create_all(bind=engine)


# FASTAPI APP AND ERROR HANDLING

app = FastAPI(title="Koya Clinic Booking API")


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)},
        headers=exc.headers
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"error": "missing or invalid details"}
    )


# DATABASE AND SECURITY HELPERS

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_api_key(x_api_key: str = Header(default=None)):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API key"
        )


def booking_data(booking):
    return {
        "id": booking.id,
        "slot_id": booking.slot_id,
        "name": booking.name,
        "email": booking.email,
        "status": booking.status,
        "created_at": booking.created_at.replace(
            tzinfo=timezone.utc
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    }


def parse_slot_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# REQUEST MODELS

class NewSlot(BaseModel):
    id: str
    provider: str
    start: str
    end: str


class ResetRequest(BaseModel):
    slots: list[NewSlot]


class NewBooking(BaseModel):
    slot_id: str
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr


# BASIC ROUTES

@app.get("/health")
def health():
    return {"status": "ok"}


# LEVEL 1: RESET SLOTS

@app.post("/reset", dependencies=[Depends(require_api_key)])
def reset_slots(data: ResetRequest, db: Session = Depends(get_db)):
    try:
        db.execute(delete(IdempotencyRecord))
        db.execute(delete(BookingAttempt))
        db.execute(delete(Booking))
        db.execute(delete(Slot))

        for item in data.slots:
            db.add(Slot(
                id=item.id,
                provider=item.provider,
                start=item.start,
                end=item.end,
                available=True
            ))

        db.commit()
        return {"message": "Done"}

    except Exception:
        db.rollback()
        raise HTTPException(status_code=400, detail="could not reset slots")


# LEVEL 1: LIST SLOTS

@app.get("/slots", dependencies=[Depends(require_api_key)])
def list_slots(
    provider: str = None,
    db: Session = Depends(get_db)
):
    query = select(Slot)

    if provider is not None:
        query = query.where(Slot.provider == provider)

    slots = db.execute(
        query.order_by(Slot.start.asc(), Slot.id.asc())
    ).scalars().all()

    return {
        "slots": [
            {
                "id": slot.id,
                "provider": slot.provider,
                "start": slot.start,
                "end": slot.end,
                "available": slot.available
            }
            for slot in slots
        ]
    }


# LEVEL 5: RATE LIMITING

def check_rate_limit(email, db):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(minutes=1)

    db.execute(
        delete(BookingAttempt).where(
            BookingAttempt.attempted_at < cutoff
        )
    )

    attempts = db.execute(
        select(BookingAttempt).where(
            BookingAttempt.email == email,
            BookingAttempt.attempted_at >= cutoff
        )
    ).scalars().all()

    if len(attempts) >= 5:
        oldest = min(item.attempted_at for item in attempts)
        retry_after = max(
            1, int(60 - (now - oldest).total_seconds())
        )
        db.rollback()

        raise HTTPException(
            status_code=429,
            detail="too many attempts for this email",
            headers={"Retry-After": str(retry_after)}
        )

    db.add(BookingAttempt(email=email, attempted_at=now))
    db.commit()


# LEVELS 1-5: CREATE BOOKING

@app.post(
    "/bookings",
    dependencies=[Depends(require_api_key)],
    status_code=201
)
def create_booking(
    data: NewBooking,
    idempotency_key: str = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db)
):
    email = str(data.email).strip().lower()
    name = data.name.strip()
    slot_id = data.slot_id.strip()

    if not name or len(name) < 2 or not slot_id:
        raise HTTPException(status_code=400, detail="missing or invalid details")

    request_hash = hashlib.sha256(
        f"{slot_id}|{name}|{email}".encode()
    ).hexdigest()

    # Check previous idempotency requests.
    if idempotency_key:
        previous = db.get(IdempotencyRecord, idempotency_key)

        if previous:
            if previous.request_hash != request_hash:
                raise HTTPException(
                    status_code=422,
                    detail="idempotency key reused with different request"
                )

            existing = db.get(Booking, previous.booking_id)

            if existing:
                return booking_data(existing)

    # Check booking attempts.
    check_rate_limit(email, db)

    # Check whether slot exists.
    slot = db.get(Slot, slot_id)

    if not slot:
        raise HTTPException(status_code=404, detail="slot not found")

    # Reject slots that have already started.
    try:
        if parse_slot_time(slot.start) <= datetime.now(timezone.utc):
            raise HTTPException(
                status_code=400,
                detail="slot has already started"
            )
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid slot date")

    # Atomically reserve the slot.
    result = db.execute(
        update(Slot)
        .where(Slot.id == slot_id, Slot.available == True)
        .values(available=False)
    )

    if result.rowcount != 1:
        db.rollback()
        raise HTTPException(status_code=409, detail="slot already booked")

    booking = Booking(
        id=str(uuid.uuid4()),
        slot_id=slot_id,
        name=name,
        email=email,
        status="confirmed",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None)
    )

    db.add(booking)

    if idempotency_key:
        db.add(IdempotencyRecord(
            key=idempotency_key,
            request_hash=request_hash,
            booking_id=booking.id
        ))

    try:
        db.commit()
        db.refresh(booking)
        return booking_data(booking)

    except IntegrityError:
        db.rollback()

        if idempotency_key:
            previous = db.get(IdempotencyRecord, idempotency_key)

            if previous:
                if previous.request_hash != request_hash:
                    raise HTTPException(
                        status_code=422,
                        detail="idempotency key reused with different request"
                    )

                existing = db.get(Booking, previous.booking_id)

                if existing:
                    return booking_data(existing)

        raise HTTPException(status_code=409, detail="slot already booked")


# LEVEL 1: LOOK UP BOOKING

@app.get(
    "/bookings/{booking_id}",
    dependencies=[Depends(require_api_key)]
)
def get_booking(booking_id: str, db: Session = Depends(get_db)):
    booking = db.get(Booking, booking_id)

    if not booking:
        raise HTTPException(status_code=404, detail="booking not found")

    return booking_data(booking)


# LEVEL 3: CANCEL BOOKING

@app.delete(
    "/bookings/{booking_id}",
    dependencies=[Depends(require_api_key)]
)
def cancel_booking(booking_id: str, db: Session = Depends(get_db)):
    booking = db.get(Booking, booking_id)

    if not booking:
        raise HTTPException(status_code=404, detail="booking not found")

    if booking.status == "cancelled":
        raise HTTPException(status_code=409, detail="booking already cancelled")

    booking.status = "cancelled"

    db.execute(
        update(Slot)
        .where(Slot.id == booking.slot_id)
        .values(available=True)
    )

    db.commit()
    db.refresh(booking)

    return booking_data(booking)


# LEVEL 4: PATIENT BOOKINGS AND PAGINATION

@app.get(
    "/bookings",
    dependencies=[Depends(require_api_key)]
)
def list_patient_bookings(
    email: EmailStr,
    limit: int = Query(default=10),
    cursor: str = None,
    db: Session = Depends(get_db)
):
    if limit < 1 or limit > 50:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 50"
        )

    normalized_email = str(email).strip().lower()

    query = select(Booking).where(
        Booking.email == normalized_email
    )

    if cursor:
        try:
            decoded = base64.urlsafe_b64decode(
                cursor.encode()
            ).decode()

            cursor_time, cursor_id = decoded.split("|", 1)
            parsed_time = datetime.fromisoformat(cursor_time)

            query = query.where(
                (Booking.created_at > parsed_time)
                | (
                    (Booking.created_at == parsed_time)
                    & (Booking.id > cursor_id)
                )
            )

        except Exception:
            raise HTTPException(status_code=400, detail="invalid cursor")

    results = db.execute(
        query.order_by(
            Booking.created_at.asc(),
            Booking.id.asc()
        ).limit(limit + 1)
    ).scalars().all()

    has_more = len(results) > limit
    page = results[:limit]

    next_cursor = None

    if has_more and page:
        last = page[-1]

        cursor_value = f"{last.created_at.isoformat()}|{last.id}"

        next_cursor = base64.urlsafe_b64encode(
            cursor_value.encode()
        ).decode()

    return {
        "bookings": [booking_data(item) for item in page],
        "next_cursor": next_cursor
    }


# SIMPLE WEB PAGE

@app.get("/", include_in_schema=False)
def web_page():
    return FileResponse("static/index.html")
