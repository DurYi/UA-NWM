#!/usr/bin/env python3
"""Compatibility entry point; final-goal LPIPS now runs on the policy server."""

from __future__ import annotations

import logging

from run_navigation import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.warning(
        "run_navigation_with_lpips_stop.py is deprecated; LPIPS now runs on the server. "
        "Use scripts/run_navigation.py with the same hardware factories."
    )
    main()
