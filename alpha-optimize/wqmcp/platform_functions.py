#!/usr/bin/env python3
"""Legacy entry point: ``python platform_functions.py`` starts the wqmcp server.

The implementation lives in server.py (MCP tools) and brain_client.py (BRAIN API
client); see README.md for the tool list and configuration.
"""

from server import brain, main, mcp  # noqa: F401  (re-exported for old imports)

brain_client = brain  # v1 name of the client instance

if __name__ == "__main__":
    main()
