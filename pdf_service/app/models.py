from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func
from .database import Base

class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    upload_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    extracted_text = Column(String, nullable=False)
    file_location = Column(String, nullable=False)  # 'local:path' or 's3:key'
