"""Run the local server: .venv/bin/python -m companion.server [--port 8300]"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Companion server with the browser playground")
    parser.add_argument("--port", type=int, default=8300)
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="  · %(name)s: %(message)s")
    logging.getLogger("companion.db").setLevel(logging.WARNING)
    # 127.0.0.1 only: there is no login yet.
    uvicorn.run(create_app(), host="127.0.0.1", port=options.port, log_level="warning")


if __name__ == "__main__":
    main()
