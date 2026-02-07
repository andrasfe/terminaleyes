"""Tests for the HTTP endpoint server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from terminaleyes.endpoint.server import create_app


class TestEndpointServer:
    """Test the FastAPI endpoint server routes.

    TODO: Add tests for:
        - GET /health returns 200 with status info
        - POST /keystroke accepts valid key names
        - POST /keystroke rejects invalid requests
        - POST /key-combo accepts modifier+key combinations
        - POST /text accepts text input
        - GET /screen returns current terminal content
        - Server starts and stops cleanly
        - Shell integration works (keystroke -> shell -> display)
    """

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for the endpoint server.

        TODO: Set up with mock shell and display for isolated testing.
        """
        app = create_app()
        return TestClient(app)
