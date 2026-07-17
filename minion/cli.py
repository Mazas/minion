"""
minion/cli.py

Entry point for the `minion` command.
Sets up the data directory and launches the Textual TUI.
"""

from __future__ import annotations

import sys

from minion.config import config
from minion.tui.app import MinionApp


def main() -> None:
    config.ensure_data_dir()
    app = MinionApp(config)
    app.run()


if __name__ == "__main__":
    main()
