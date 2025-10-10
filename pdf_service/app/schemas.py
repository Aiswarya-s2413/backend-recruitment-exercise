from pydantic import BaseModel
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str

class DocumentCreate(DocumentBase):
    extracted_text: str

class DocumentResponse(DocumentBase):
    doc_id: str
    upload_timestamp: datetime
    extracted_text: str
    file_location: str

    class Config:
        orm_mode = True
