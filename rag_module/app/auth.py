from fastapi import HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import os
from .logger import get_logger
from .exceptions import AuthenticationError, AuthorizationError

# Logger
logger = get_logger(__name__)

# Security scheme - switched to HTTP Basic for simpler manual testing
security = HTTPBasic()

# For simplicity, we'll use a dummy user. In production, you'd have a user database.
def authenticate_user(username: str, password: str):
    logger.info(f"Attempting authentication for user: {username}")
    # Dummy authentication - replace with real user lookup
    if username == "admin" and password == "password":
        logger.info(f"Authentication successful for user: {username}")
        return {"username": username}
    logger.warning(f"Authentication failed for user: {username}")
    return False

def verify_token(credentials: HTTPBasicCredentials = Depends(security)):
    logger.info("Verifying authentication credentials")
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        logger.warning("Authentication failed")
        raise AuthenticationError("Invalid authentication credentials")
    logger.info(f"Authentication successful for user: {user['username']}")
    return user["username"]