from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import os

# Set auto_error=False to handle missing token manually and return 401
http_bearer_auth = HTTPBearer(auto_error=False)

async def verify_token(creds: HTTPAuthorizationCredentials = Depends(http_bearer_auth)):
    """
    Verify the provided service token.
    """
    # Get SERVICE_TOKEN inside the function to ensure we get the latest value
    SERVICE_TOKEN = os.getenv("SERVICE_TOKEN")
    
    # If no service token is configured, allow access (for local development)
    if not SERVICE_TOKEN:
        return "anonymous"

    # If SERVICE_TOKEN is configured, credentials are REQUIRED
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing service token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify the token matches using constant-time comparison
    if not secrets.compare_digest(creds.credentials, SERVICE_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing service token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return "service"