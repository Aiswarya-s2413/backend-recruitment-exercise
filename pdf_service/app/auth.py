from fastapi import HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import os

# Security scheme - switched to HTTP Basic for simpler manual testing
security = HTTPBasic()

# For simplicity, we'll use a dummy user. In production, you'd have a user database.
def authenticate_user(username: str, password: str):
    # Dummy authentication - replace with real user lookup
    if username == "admin" and password == "password":
        return {"username": username}
    return False

def verify_token(credentials: HTTPBasicCredentials = Depends(security)):
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user["username"]