import uuid
from sqlalchemy.orm import Session
from . import models, schemas

def create_document(db: Session, doc_id: str, filename: str, extracted_text: str, file_location: str):
    db_doc = models.Document(doc_id=doc_id, filename=filename, extracted_text=extracted_text, file_location=file_location)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc

def get_document(db: Session, doc_id: str):
    return db.query(models.Document).filter(models.Document.doc_id == doc_id).first()

def get_documents(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.Document).offset(skip).limit(limit).all()
