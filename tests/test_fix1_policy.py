import pytest
from app.policy.policy_engine import policy_engine
from app.core.config import settings
from unittest.mock import patch

def mock_get_active_config(workspace_id):
    return {
        "top_k": settings.top_k,
        "dense_weight": settings.dense_weight,
        "bm25_weight": settings.bm25_weight
    }

@patch('app.policy.policy_engine.config_manager.get_active_config', side_effect=mock_get_active_config)
def test_increase_top_k(mock_config):
    settings.top_k = 5
    result = policy_engine.recommend("legacy", {"suggested_actions": ["INCREASE_TOP_K"]})
    assert result["candidate_config"]["top_k"] == 10

@patch('app.policy.policy_engine.config_manager.get_active_config', side_effect=mock_get_active_config)
def test_increase_top_k_near_max(mock_config):
    settings.top_k = 18
    result = policy_engine.recommend("legacy", {"suggested_actions": ["INCREASE_TOP_K"]})
    assert result["candidate_config"]["top_k"] == 20

@patch('app.policy.policy_engine.config_manager.get_active_config', side_effect=mock_get_active_config)
def test_repeated_increases(mock_config):
    settings.top_k = 20
    result = policy_engine.recommend("legacy", {"suggested_actions": ["INCREASE_TOP_K"]})
    assert result["candidate_config"]["top_k"] == 20

@patch('app.policy.policy_engine.config_manager.get_active_config', side_effect=mock_get_active_config)
def test_decrease_top_k_bounds(mock_config):
    settings.top_k = 2
    result = policy_engine.recommend("legacy", {"suggested_actions": ["DECREASE_TOP_K"]})
    assert result["candidate_config"]["top_k"] == 1

@patch('app.policy.policy_engine.config_manager.get_active_config', side_effect=mock_get_active_config)
def test_invalid_baseline_too_high(mock_config):
    settings.top_k = 100
    result = policy_engine.recommend("legacy", {"suggested_actions": ["DECREASE_TOP_K"]})
    assert result["current_config"]["top_k"] == 20
    assert result["candidate_config"]["top_k"] == 17

@patch('app.policy.policy_engine.config_manager.get_active_config', side_effect=mock_get_active_config)
def test_invalid_baseline_too_low(mock_config):
    settings.top_k = -5
    result = policy_engine.recommend("legacy", {"suggested_actions": ["INCREASE_TOP_K"]})
    assert result["current_config"]["top_k"] == 1
    assert result["candidate_config"]["top_k"] == 6

@patch('app.policy.policy_engine.config_manager.get_active_config', side_effect=mock_get_active_config)
def test_other_fields_unchanged(mock_config):
    settings.top_k = 5
    settings.dense_weight = 0.8
    result = policy_engine.recommend("legacy", {"suggested_actions": ["INCREASE_TOP_K"]})
    assert result["candidate_config"]["dense_weight"] == 0.8
