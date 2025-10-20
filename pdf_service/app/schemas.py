from pydantic import BaseModel, ConfigDict
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

    model_config = ConfigDict(from_attributes=True)
