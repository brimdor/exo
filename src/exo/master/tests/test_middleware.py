import pytest
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from exo.master.api import API
from exo.shared.types.common import NodeId, SessionId

# Mock find_dashboard to avoid IO error during invalid path mount
# removed manual patch

from typing import Generator
@pytest.fixture
def api_setup() -> Generator[API, None, None]:
    node_id = NodeId("node1")
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    mock_receiver = MagicMock()
    mock_receiver.__enter__ = MagicMock()
    mock_receiver.__exit__ = MagicMock()
    mock_sender = MagicMock()
    
    # Patch find_dashboard where it is defined/used to ensure it is intercepted
    with unittest.mock.patch("exo.master.api.find_dashboard", return_value=Path("/tmp")):
        api = API(
            node_id=node_id,
            session_id=session_id,
            port=8080,
            global_event_receiver=mock_receiver,
            command_sender=mock_sender,
            election_receiver=mock_receiver
        )
        yield api

def test_master_access(api_setup: API):
    api = api_setup
    client = TestClient(api.app)
    # Master access (node_id == master_node_id)
    resp = client.get("/node_id")
    # Should be 200 or at least not 403
    assert resp.status_code != 403

def test_worker_blocked(api_setup: API):
    api = api_setup
    client = TestClient(api.app)
    
    # Change to worker (node_id != master_node_id)
    other_master = NodeId("node2")
    # Simulate session update
    api.session_id = SessionId(master_node_id=other_master, election_clock=1)
    
    resp = client.get("/node_id")
    assert resp.status_code == 403
    assert "Master Node" in resp.json()["detail"]

def test_dashboard_access_blocked(api_setup: API):
    api = api_setup
    client = TestClient(api.app)
    
    # Change to worker
    other_master = NodeId("node2")
    api.session_id = SessionId(master_node_id=other_master, election_clock=1)
    
    # Assuming / is dashboard
    resp = client.get("/")
    assert resp.status_code == 403
