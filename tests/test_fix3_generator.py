import pytest
import requests
from unittest.mock import patch, MagicMock
from app.rag.generator import generator, log_retry
from app.core.config import settings
import os

@pytest.fixture(autouse=True)
def setup_generator_env():
    # Use ollama for testing since it's the default
    settings.llm_provider = "ollama"
    yield

@patch("app.rag.generator.requests.post")
@patch("app.rag.generator.wait_exponential") # mock wait to avoid actual delays
def test_retry_on_429_success(mock_wait, mock_post):
    mock_wait.return_value = lambda x: 0 # return wait 0s
    
    resp_429 = MagicMock()
    resp_429.status_code = 429
    exc_429 = requests.exceptions.HTTPError("429 Too Many Requests")
    exc_429.response = resp_429
    
    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"response": "Success_Ans", "eval_count": 10}
    
    mock_post.side_effect = [exc_429, resp_200]
    
    result = generator.generate("query", "context")
    
    assert mock_post.call_count == 2
    assert result["answer"] == "Success_Ans"

@patch("app.rag.generator.requests.post")
@patch("app.rag.generator.wait_exponential")
def test_repeated_429_stops(mock_wait, mock_post):
    mock_wait.return_value = lambda x: 0
    
    resp_429 = MagicMock()
    resp_429.status_code = 429
    exc_429 = requests.exceptions.HTTPError("429 Too Many Requests")
    exc_429.response = resp_429
    
    mock_post.side_effect = exc_429
    
    result = generator.generate("query", "context")
    
    assert mock_post.call_count == 4 # 1 initial + 3 retries
    assert result["answer"] == "Failed to generate answer due to LLM error."

@patch("app.rag.generator.requests.post")
@patch("app.rag.generator.wait_exponential")
def test_transient_server_error(mock_wait, mock_post):
    mock_wait.return_value = lambda x: 0
    
    resp_503 = MagicMock()
    resp_503.status_code = 503
    exc_503 = requests.exceptions.HTTPError("503 Service Unavailable")
    exc_503.response = resp_503
    
    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"response": "Recovered", "eval_count": 10}
    
    mock_post.side_effect = [exc_503, resp_200]
    
    result = generator.generate("query", "context")
    
    assert mock_post.call_count == 2
    assert result["answer"] == "Recovered"

@patch("app.rag.generator.requests.post")
@patch("app.rag.generator.wait_exponential")
def test_permanent_error_no_retry(mock_wait, mock_post):
    mock_wait.return_value = lambda x: 0
    
    resp_401 = MagicMock()
    resp_401.status_code = 401
    exc_401 = requests.exceptions.HTTPError("401 Unauthorized")
    exc_401.response = resp_401
    
    mock_post.side_effect = exc_401
    
    result = generator.generate("query", "context")
    
    assert mock_post.call_count == 1 # No retries
    assert result["answer"] == "Failed to generate answer due to LLM error."

@patch("app.rag.generator.requests.post")
@patch("app.rag.generator.wait_exponential")
def test_timeout_retry(mock_wait, mock_post):
    mock_wait.return_value = lambda x: 0
    
    exc_timeout = requests.exceptions.Timeout("Read timeout")
    
    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"response": "Timeout_Recovered", "eval_count": 10}
    
    mock_post.side_effect = [exc_timeout, resp_200]
    
    result = generator.generate("query", "context")
    
    assert mock_post.call_count == 2
    assert result["answer"] == "Timeout_Recovered"
