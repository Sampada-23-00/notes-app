import os
from datetime import datetime, timedelta, timezone
from typing import List
import uuid

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, String, Text, DateTime,
    ForeignKey, Integer, or_,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./notes.db")

# ── Database ──────────────────────────────────────────────────────────────────
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ── ORM Models ────────────────────────────────────────────────────────────────
class NoteShareLink(Base):
    __tablename__ = "note_shares"
    note_id = Column(String, ForeignKey("notes.id", ondelete="CASCADE"), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    owned_notes = relationship("Note", back_populates="owner", cascade="all, delete-orphan")


class Note(Base):
    __tablename__ = "notes"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False, default="")
    owner_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    owner = relationship("User", back_populates="owned_notes")
    versions = relationship(
        "NoteVersion", back_populates="note",
        order_by="NoteVersion.version_number",
        cascade="all, delete-orphan",
    )


class NoteVersion(Base):
    """Stores a snapshot of a note taken before each update."""
    __tablename__ = "note_versions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    note_id = Column(String, ForeignKey("notes.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    version_number = Column(Integer, nullable=False)
    saved_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    note = relationship("Note", back_populates="versions")


Base.metadata.create_all(bind=engine)

# ── Auth helpers ──────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(user_id: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": user_id, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub", "")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class RegisterIn(BaseModel):
    email: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


class NoteIn(BaseModel):
    title: str
    content: str


class ShareIn(BaseModel):
    share_with_email: str


class NoteOut(BaseModel):
    id: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class VersionOut(BaseModel):
    id: str
    note_id: str
    title: str
    content: str
    version_number: int
    saved_at: datetime

    class Config:
        from_attributes = True


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Notes API",
    description=(
        "Multi-user notes service with JWT auth, note sharing, "
        "full-text search, and automatic version history."
    ),
    version="1.0.0",
)


# ── Utility ───────────────────────────────────────────────────────────────────
def _require_access(note_id: str, user: User, db: Session) -> Note:
    """Return note if user owns it or it has been shared with them."""
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    shared = db.query(NoteShareLink).filter(
        NoteShareLink.note_id == note_id,
        NoteShareLink.user_id == user.id,
    ).first()
    if note.owner_id != user.id and not shared:
        raise HTTPException(status_code=403, detail="Access denied")
    return note


def _save_version(note: Note, db: Session):
    next_num = (
        db.query(NoteVersion)
        .filter(NoteVersion.note_id == note.id)
        .count()
    ) + 1
    db.add(NoteVersion(
        note_id=note.id,
        title=note.title,
        content=note.content,
        version_number=next_num,
    ))


# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/register", status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if not body.email or "@" not in body.email or "." not in body.email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if db.query(User).filter(User.email == body.email.lower().strip()).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(email=body.email.lower().strip(), password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return {"message": "User registered successfully"}


@app.post("/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    if not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    user = db.query(User).filter(User.email == body.email.lower().strip()).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"access_token": create_token(user.id)}


# ── Notes CRUD ────────────────────────────────────────────────────────────────
@app.get("/notes", response_model=List[NoteOut])
def list_notes(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    per_page: int = Query(20, ge=1, le=100, description="Results per page"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all notes owned by or shared with the authenticated user (paginated)."""
    shared_ids = (
        db.query(NoteShareLink.note_id)
        .filter(NoteShareLink.user_id == user.id)
        .scalar_subquery()
    )
    notes = (
        db.query(Note)
        .filter(or_(Note.owner_id == user.id, Note.id.in_(shared_ids)))
        .order_by(Note.updated_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    return notes


@app.get("/notes/{note_id}", response_model=NoteOut)
def get_note(
    note_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return _require_access(note_id, user, db)


@app.post("/notes", status_code=201, response_model=NoteOut)
def create_note(
    body: NoteIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not body.title or not body.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    note = Note(title=body.title.strip(), content=body.content, owner_id=user.id)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@app.put("/notes/{note_id}", response_model=NoteOut)
def update_note(
    note_id: str,
    body: NoteIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    note = db.query(Note).filter(Note.id == note_id, Note.owner_id == user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found or you are not the owner")
    if not body.title or not body.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    _save_version(note, db)  # snapshot before overwrite

    note.title = body.title.strip()
    note.content = body.content
    note.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(note)
    return note


@app.delete("/notes/{note_id}", status_code=204)
def delete_note(
    note_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    note = db.query(Note).filter(Note.id == note_id, Note.owner_id == user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found or you are not the owner")
    db.delete(note)
    db.commit()


# ── Sharing ───────────────────────────────────────────────────────────────────
@app.post("/notes/{note_id}/share")
def share_note(
    note_id: str,
    body: ShareIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    note = db.query(Note).filter(Note.id == note_id, Note.owner_id == user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found or you are not the owner")

    if not body.share_with_email:
        raise HTTPException(status_code=400, detail="share_with_email is required")

    target = db.query(User).filter(User.email == body.share_with_email.lower().strip()).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.id == user.id:
        raise HTTPException(status_code=400, detail="Cannot share a note with yourself")

    already = db.query(NoteShareLink).filter(
        NoteShareLink.note_id == note_id,
        NoteShareLink.user_id == target.id,
    ).first()
    if not already:
        db.add(NoteShareLink(note_id=note_id, user_id=target.id))
        db.commit()

    return {"message": f"Note shared with {body.share_with_email}"}


# ── Search (stretch goal) ─────────────────────────────────────────────────────
@app.get("/search", response_model=List[NoteOut])
def search_notes(
    q: str = Query(..., min_length=1, description="Search keyword"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Full-text search across titles and content of accessible notes."""
    shared_ids = (
        db.query(NoteShareLink.note_id)
        .filter(NoteShareLink.user_id == user.id)
        .scalar_subquery()
    )
    pattern = f"%{q}%"
    notes = (
        db.query(Note)
        .filter(
            or_(Note.owner_id == user.id, Note.id.in_(shared_ids)),
            or_(Note.title.ilike(pattern), Note.content.ilike(pattern)),
        )
        .order_by(Note.updated_at.desc())
        .all()
    )
    return notes


# ── Version history (custom feature) ─────────────────────────────────────────
@app.get("/notes/{note_id}/history", response_model=List[VersionOut])
def get_note_history(
    note_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return all saved versions of a note (owner only).

    A new version is automatically captured before every PUT update, giving
    users a complete audit trail and the ability to undo unwanted changes.
    """
    note = db.query(Note).filter(Note.id == note_id, Note.owner_id == user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found or you are not the owner")
    return note.versions


@app.post("/notes/{note_id}/revert/{version_number}", response_model=NoteOut)
def revert_note(
    note_id: str,
    version_number: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Restore a note to a previously saved version (owner only).

    The current state is saved as a new version before reverting,
    so the revert itself is also undoable.
    """
    note = db.query(Note).filter(Note.id == note_id, Note.owner_id == user.id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found or you are not the owner")

    target = db.query(NoteVersion).filter(
        NoteVersion.note_id == note_id,
        NoteVersion.version_number == version_number,
    ).first()
    if not target:
        raise HTTPException(status_code=404, detail=f"Version {version_number} not found")

    _save_version(note, db)  # snapshot current state before reverting

    note.title = target.title
    note.content = target.content
    note.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(note)
    return note


# ── Meta endpoints ────────────────────────────────────────────────────────────
@app.get("/about")
def about():
    return {
        "name": "Sampada Waghode",
        "email": "sampada.waghode@gmail.com",
        "my features": {
            "Note Version History": (
                "Every PUT /notes/{id} automatically snapshots the previous content before "
                "overwriting it. Users can retrieve the full edit history via "
                "GET /notes/{id}/history and restore any past version via "
                "POST /notes/{id}/revert/{version}. The revert itself is also snapshotted, "
                "making every change undoable. This prevents accidental data loss and lets "
                "users track how their ideas evolved over time — a critical feature for a "
                "notes app where users often refine drafts iteratively."
            ),
        },
    }


@app.get("/openapi.json", include_in_schema=False)
def openapi_schema():
    return app.openapi()
