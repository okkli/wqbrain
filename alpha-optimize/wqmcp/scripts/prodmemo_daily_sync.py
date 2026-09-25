#!/usr/bin/env python
"""Daily ProdMemo incremental sync (cron entry point).

Talks to the already-running MCP server over HTTP rather than importing the
service directly. That keeps a single BRAIN session (credd is designed as one
shared session — a second logging-in process risks the account lock), a single
DB connection pool, and avoids import-time side effects from
platform_functions/forum_functions.

Exit codes: 0 = sync completed, 1 = sync reported an error / timed out,
2 = could not reach or drive the MCP server. Cron mails non-zero exits.

Install:
    0 12 * * * /path/to/python \
        /path/to/wqmcp/scripts/prodmemo_daily_sync.py \
        >> /var/log/prodmemo-sync.log 2>&1
"""

import json
import os
import sys
import time
from datetime import datetime
import urllib.error
import urllib.request

MCP_URL = os.environ.get('PRODMEMO_MCP_URL', 'http://127.0.0.1:8761/mcp')
MODE = os.environ.get('PRODMEMO_SYNC_MODE', 'incremental')
# A full sync of several hundred alphas can take ~20 min; leave generous headroom before the
# next day's run. Poll slowly — the sync writes progress to Postgres, not here.
MAX_WAIT_SECONDS = int(os.environ.get('PRODMEMO_SYNC_MAX_WAIT', '5400'))
POLL_SECONDS = int(os.environ.get('PRODMEMO_SYNC_POLL', '60'))
TERMINAL = ('completed', 'error', 'stopped')


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class McpClient:
    """Minimal streamable-http MCP client: initialize, then tools/call."""

    def __init__(self, url):
        self.url = url
        self.session_id = None
        self._next_id = 0

    def _new_id(self):
        self._next_id += 1
        return self._next_id

    def _post(self, payload, expect_response=True):
        data = json.dumps(payload).encode()
        request = urllib.request.Request(self.url, data=data, method='POST')
        request.add_header('Content-Type', 'application/json')
        request.add_header('Accept', 'application/json, text/event-stream')
        if self.session_id:
            request.add_header('mcp-session-id', self.session_id)
        with urllib.request.urlopen(request, timeout=300) as response:
            if self.session_id is None:
                self.session_id = response.headers.get('mcp-session-id')
            body = response.read().decode()
        if not expect_response:
            return None
        # streamable-http frames JSON as SSE "data:" lines; skip server
        # notifications and pick the response to this request's id.
        for line in body.splitlines():
            line = line[5:].strip() if line.startswith('data:') else line.strip()
            if not line.startswith('{'):
                continue
            message = json.loads(line)
            if message.get('id') == payload.get('id'):
                return message
        return None

    def connect(self):
        self._post({'jsonrpc': '2.0', 'id': self._new_id(), 'method': 'initialize',
                    'params': {'protocolVersion': '2024-11-05', 'capabilities': {},
                               'clientInfo': {'name': 'prodmemo-cron', 'version': '1'}}})
        if not self.session_id:
            raise RuntimeError('MCP server did not return an mcp-session-id')
        self._post({'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                   expect_response=False)

    def call(self, name, arguments=None):
        response = self._post({'jsonrpc': '2.0', 'id': self._new_id(), 'method': 'tools/call',
                               'params': {'name': name, 'arguments': arguments or {}}})
        if not response:
            raise RuntimeError(f'empty response from tool {name}')
        if 'error' in response:
            raise RuntimeError(f'tool {name} failed: {response["error"]}')
        content = response.get('result', {}).get('content') or []
        if not content:
            return {}
        text = content[0].get('text', '')
        if response.get('result', {}).get('isError'):
            raise RuntimeError(f'tool {name} failed: {text}')
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {'raw': text}

    def tool_names(self):
        response = self._post({'jsonrpc': '2.0', 'id': self._new_id(), 'method': 'tools/list'})
        return {t.get('name') for t in ((response or {}).get('result') or {}).get('tools') or []}

    def close(self):
        """End the server-side MCP session (otherwise one leaks per run)."""
        if not self.session_id:
            return
        request = urllib.request.Request(self.url, method='DELETE')
        request.add_header('mcp-session-id', self.session_id)
        try:
            urllib.request.urlopen(request, timeout=30).close()
        except (urllib.error.URLError, OSError):
            pass


def main():
    client = McpClient(MCP_URL)
    try:
        client.connect()
    except (urllib.error.URLError, OSError, RuntimeError) as exc:
        log(f'FATAL: cannot reach MCP server at {MCP_URL}: {exc}')
        return 2

    try:
        # A server not yet restarted on the new code still has prodmemo_sync_status;
        # there prodmemo_sync(mode='status') would start another sync on every poll.
        status_call = (('prodmemo_sync_status',) if 'prodmemo_sync_status' in client.tool_names()
                       else ('prodmemo_sync', {'mode': 'status'}))
        before = client.call('prodmemo_stats')
        log(f"before: alphas={before.get('submitted_alpha_count')} "
            f"pnls={before.get('pnl_count')} "
            f"platform_corrs={before.get('platform_corr_count')} "
            f"refs={before.get('valid_reference_count')}")

        started = client.call('prodmemo_sync', {'mode': MODE})
        if not started.get('started'):
            # Yesterday's run still going, or another client kicked one off.
            log(f'sync not started: {started}')
            if started.get('reason') != 'already_running':
                return 1
            # A run older than MAX_WAIT is stuck, not busy: make cron say so.
            sync = started.get('sync') or {}
            started_at = sync.get('startedAt')
            try:
                age = time.time() - datetime.fromisoformat(
                    str(started_at).replace('Z', '+00:00')).timestamp()
            except (TypeError, ValueError):
                age = None
            if age is not None and age > MAX_WAIT_SECONDS:
                log(f'STUCK: a sync has been running since {started_at} ({int(age)}s); '
                    f'stop it with prodmemo_sync(mode="stop") and check the server log')
                return 1
            return 0

        deadline = time.time() + MAX_WAIT_SECONDS
        state = {}
        while time.time() < deadline:
            time.sleep(POLL_SECONDS)
            state = client.call(*status_call)
            if state.get('status') in TERMINAL and not state.get('running'):
                break
            log(f"  ...{state.get('phase')} {state.get('current') or ''}"
                f"/{state.get('total') or ''} {state.get('message') or ''}")
        else:
            log(f'TIMEOUT after {MAX_WAIT_SECONDS}s; last phase={state.get("phase")}')
            return 1

        after = client.call('prodmemo_stats')
        log(f"{state.get('status')}: {state.get('message')}")
        log(f"after:  alphas={after.get('submitted_alpha_count')} "
            f"pnls={after.get('pnl_count')} "
            f"platform_corrs={after.get('platform_corr_count')} "
            f"refs={after.get('valid_reference_count')}")
        for key in ('failedIds', 'backfillFailedIds'):
            if state.get(key):
                log(f'  {key}: {state[key]}')
        return 0 if state.get('status') == 'completed' else 1
    except Exception as exc:  # noqa: BLE001 - cron needs the reason in the log
        log(f'FATAL: {type(exc).__name__}: {exc}')
        return 2
    finally:
        client.close()


if __name__ == '__main__':
    sys.exit(main())
