#!/usr/bin/env python3
"""
SupportGraph AI — Groq API Connection Verification Script

Safely tests connectivity to the Groq API using GROQ_API_KEY from backend/.env.
Never logs or displays the actual API key.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def main() -> int:
    # 1. Resolve backend/.env path
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent
    env_path = backend_dir / ".env"

    if not env_path.is_file():
        print("Groq API connection failed: backend/.env file not found.")
        return 1

    # Load environment variables from backend/.env explicitly
    load_dotenv(dotenv_path=env_path, override=True)

    # 2. Check whether GROQ_API_KEY is set and non-empty
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("Groq API connection failed: GROQ_API_KEY is not set or is empty in backend/.env")
        return 1

    # 3. Connect to Groq safely and perform minimal check
    try:
        from groq import Groq, AuthenticationError, APIConnectionError, APIStatusError
    except ImportError:
        print("Groq API connection failed: The 'groq' Python package is not installed.")
        return 1

    try:
        client = Groq(api_key=api_key)
        # Minimal connectivity check: list available models
        client.models.list()
        print("Groq API connection successful.")
        return 0
    except AuthenticationError:
        print("Groq API connection failed: Invalid API key or unauthorized access.")
        return 1
    except APIConnectionError:
        print("Groq API connection failed: Unable to reach Groq API servers. Please check network connectivity.")
        return 1
    except APIStatusError as e:
        # Sanitize any unexpected error message to ensure the key is never leaked
        safe_msg = str(e.message) if hasattr(e, "message") else "HTTP error from Groq API"
        if api_key in safe_msg:
            safe_msg = safe_msg.replace(api_key, "[REDACTED]")
        print(f"Groq API connection failed: {safe_msg}")
        return 1
    except Exception as e:
        safe_msg = str(e)
        if api_key in safe_msg:
            safe_msg = safe_msg.replace(api_key, "[REDACTED]")
        print(f"Groq API connection failed: {safe_msg}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
