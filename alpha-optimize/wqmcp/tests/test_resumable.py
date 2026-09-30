"""A client whose stream drops mid-call gets the answer by reconnecting with
Last-Event-ID (2026-09-30c: answers sent into dead connections were lost and
clients waited ~300s)."""

import asyncio
import json
import os
import socket
import sys
import threading
import time

import pytest

pytest.importorskip("mcp")
httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import platform_functions as pf  # noqa: E402

PROTOCOL = "2025-11-25"


def _events(lines):
    """(id, data) of the SSE events in an iterator of lines."""
    event_id, data = None, []
    for line in lines:
        if line.startswith("id:"):
            event_id = line[3:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "":
            if event_id is not None or data:
                yield event_id, "\n".join(data)
            event_id, data = None, []


@pytest.fixture
def server(monkeypatch):
    if pf.EventMessage is None or "event_store" not in pf._resumable():
        pytest.skip("this mcp version has no resumable streams")

    async def slow_pnl(alpha_id, wait):
        await asyncio.sleep(1.5)
        return {"records": [[f"2024-01-{d:02d}", float(d * (1 if alpha_id == "A" else 2))] for d in range(1, 20)]}
    monkeypatch.setattr(pf.brain_client, "get_alpha_pnl", slow_pnl)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(pf.mcp.streamable_http_app(), host="127.0.0.1", port=port,
                                        log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    srv.should_exit = True
    thread.join(5)


def test_an_answer_sent_while_the_stream_was_gone_is_replayed(server):
    head = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
            "MCP-Protocol-Version": PROTOCOL}
    with httpx.Client(timeout=10) as http:
        init = http.post(server, headers=head, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": PROTOCOL, "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}}})
        session = init.headers["mcp-session-id"]
        head["mcp-session-id"] = session
        http.post(server, headers=head, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

        # the call starts; its stream is cut right after the first event (the priming one)
        with http.stream("POST", server, headers=head, json={
                "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                "params": {"name": "compare_alphas", "arguments": {"alpha_ids": ["A", "B"]}}}) as r:
            first_id, _ = next(_events(r.iter_lines()))
        assert first_id

        time.sleep(2.5)                       # the tool finishes while nobody listens
        with http.stream("GET", server, headers={**head, "Last-Event-ID": first_id}) as r:
            for event_id, data in _events(r.iter_lines()):
                if data:
                    message = json.loads(data)
                    break
    assert message["id"] == 7 and "result" in message
    assert len(json.loads(message["result"]["content"][0]["text"])["pairs"]) == 1
