import os
import subprocess
import pytest
from pydantic import ValidationError
from starlette.middleware.sessions import SessionMiddleware
from starlette.testclient import TestClient
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings, INSECURE_DEFAULT_SECRETS, DEV_FALLBACK_SECRET
from app.main import app


def test_production_config_rejects_missing_secret():
    """Verify production environment strictly rejects missing or blank SESSION_SECRET_KEY."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        Settings(app_env="production", session_secret_key=None)
    assert "SESSION_SECRET_KEY is required in production" in str(exc_info.value)

    with pytest.raises((ValidationError, ValueError)) as exc_info_empty:
        Settings(app_env="production", session_secret_key="   ")
    assert "SESSION_SECRET_KEY is required in production" in str(exc_info_empty.value)


def test_production_config_rejects_hardcoded_or_insecure_defaults():
    """Verify that previously hardcoded defaults or placeholders are rejected in production."""
    for bad_secret in INSECURE_DEFAULT_SECRETS:
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            Settings(app_env="production", session_secret_key=bad_secret)
        assert "known insecure default" in str(exc_info.value)


def test_production_config_rejects_short_secret():
    """Verify that production rejects secrets shorter than 32 characters."""
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        Settings(app_env="production", session_secret_key="short_secret_under_32_chars")
    assert "must be at least 32 characters long" in str(exc_info.value)


def test_development_and_testing_config_remains_usable():
    """Verify that development and testing environments retain usable fallbacks."""
    dev_settings = Settings(app_env="development", session_secret_key=None)
    assert dev_settings.session_secret_key == DEV_FALLBACK_SECRET
    assert dev_settings.is_cookie_secure is False

    test_settings = Settings(app_env="testing", session_secret_key=None)
    assert test_settings.session_secret_key == DEV_FALLBACK_SECRET
    assert test_settings.is_cookie_secure is False

    # Explicit key in dev/testing should be respected
    custom_key = "a_custom_local_dev_key_1234567890"
    custom_dev = Settings(app_env="development", session_secret_key=custom_key)
    assert custom_dev.session_secret_key == custom_key


def test_production_cookie_security_attributes():
    """Verify cookie security attributes (Secure, HttpOnly, SameSite) in production."""
    prod_key = "a" * 32
    prod_settings = Settings(app_env="production", session_secret_key=prod_key)
    assert prod_settings.is_cookie_secure is True
    assert prod_settings.session_cookie_same_site == "lax"

    # Test that SessionMiddleware configured with production settings emits Secure and SameSite=lax
    test_app = FastAPI()
    test_app.add_middleware(
        SessionMiddleware,
        secret_key=prod_settings.session_secret_key,
        max_age=prod_settings.session_max_age,
        same_site=prod_settings.session_cookie_same_site,
        https_only=prod_settings.is_cookie_secure
    )

    @test_app.get("/set_cookie")
    def set_cookie(request: Request):
        request.session["test"] = "val"
        return JSONResponse({"status": "ok"})

    client = TestClient(test_app)
    response = client.get("/set_cookie")
    assert response.status_code == 200
    set_cookie_header = response.headers.get("set-cookie", "")
    assert "session=" in set_cookie_header
    assert "secure" in set_cookie_header.lower()
    assert "httponly" in set_cookie_header.lower()
    assert "samesite=lax" in set_cookie_header.lower()


def test_env_example_contains_no_real_credentials():
    """Verify that .env.example contains only placeholders and no live credentials."""
    env_example_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env.example")
    assert os.path.exists(env_example_path), ".env.example must exist"
    
    with open(env_example_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Must contain placeholder directives
    assert "SESSION_SECRET_KEY=" in content
    assert "GROQ_API_KEY=" in content
    assert "GEMINI_API_KEY=" in content

    # Must NOT contain actual live keys
    assert "gsk_" not in content, "Real Groq key found in .env.example!"
    assert "AQ." not in content, "Real Gemini key found in .env.example!"


def test_provider_test_scripts_contain_no_hardcoded_keys():
    """Verify test scripts do not contain hardcoded credentials and require env vars."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_scripts = ["test_groq_models.py", "scratch_test_gemini.py", "test_groq.py"]

    for script_name in test_scripts:
        script_path = os.path.join(root_dir, script_name)
        if os.path.exists(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                code = f.read()
            assert "gsk_" not in code, f"Hardcoded Groq key found in {script_name}"
            assert "AQ." not in code, f"Hardcoded Gemini key found in {script_name}"
            assert "os.getenv" in code, f"{script_name} must use os.getenv to read credentials"


def test_provider_scripts_fail_gracefully_when_env_missing():
    """Verify provider test scripts exit with an error code when environment variable is unset."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(root_dir, "test_groq_models.py")

    # Run with empty environment for GROQ_API_KEY
    clean_env = os.environ.copy()
    clean_env.pop("GROQ_API_KEY", None)
    clean_env["GROQ_API_KEY"] = ""

    result = subprocess.run(
        ["python", script_path],
        cwd=root_dir,
        capture_output=True,
        text=True,
        env=clean_env
    )
    assert result.returncode != 0
    assert "GROQ_API_KEY environment variable is missing" in (result.stdout + result.stderr)


def test_gitignore_covers_sensitive_and_runtime_files():
    """Verify that .gitignore properly covers sensitive files and runtime directories."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gitignore_path = os.path.join(root_dir, ".gitignore")
    assert os.path.exists(gitignore_path), ".gitignore must exist"

    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert ".env" in content
    assert "*.db" in content
    assert "data/documents/" in content
    assert "results/" in content
    assert "__pycache__/" in content


def test_auth_workflow_works_end_to_end_with_configured_secret():
    """Verify that registration, login, session cookies, and logout continue working."""
    client = TestClient(app)
    username = "p2_2_user"
    password = "password123"

    # 1. Register
    reg_res = client.post("/register", data={"username": username, "password": password}, follow_redirects=False)
    assert reg_res.status_code == 302
    assert "session" in reg_res.cookies

    # 2. Access dashboard with session
    dash_res = client.get("/")
    assert dash_res.status_code == 200

    # 3. Logout
    logout_res = client.get("/logout", follow_redirects=False)
    assert logout_res.status_code == 302
    assert logout_res.headers["location"] == "/login"
