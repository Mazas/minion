"""
minion/cli.py

Entry point for the `minion` command.
Sets up the data directory and launches the Textual TUI.
"""

from __future__ import annotations

from minion.config import config
from minion.memory.manager import MemoryManager
from minion.tools.search import get_search_provider
from minion.tui.app import MinionApp


def main() -> None:
    config.ensure_data_dir()
    memory = MemoryManager(config.db_path)
    search = get_search_provider(config)
    app = MinionApp(config, memory, search)
    try:
        app.run()
    finally:
        memory.close()


if __name__ == "__main__":
    main()
