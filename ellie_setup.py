#!/usr/bin/env python3
# Copyright (c) 2026 Shreyash Padhi. All rights reserved.
# Licensed under the Ellie Source Available License v1.0 — see LICENSE file.
"""
ellie_setup.py — Entry point and first-time setup wizard for Ellie.

Opens a terminal, asks whether to register Ellie to run at Windows startup,
then hides the console and launches the main GUI.
"""

import os
import sys
import winreg

APP_NAME = "Ellie"
_STARTUP_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def _exe_path() -> str:
    """Return the path to this exe (or script when running unbuilt)."""
    if getattr(sys, "frozen", False):
        return sys.executable
    return os.path.abspath(__file__)


def _register_startup(exe_path: str) -> bool:
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _STARTUP_KEY,
            0,
            winreg.KEY_SET_VALUE,
        )
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, f'"{exe_path}"')
        winreg.CloseKey(key)
        return True
    except Exception as exc:
        print(f"  [!] Could not write to registry: {exc}")
        return False


def main() -> None:
    print("=" * 52)
    print("  Ellie — Setup")
    print("=" * 52)
    print()

    # ── Startup question ────────────────────────────────────────────────────
    print("Would you like Ellie to start automatically when Windows starts?")
    while True:
        answer = input("  [y/n]: ").strip().lower()
        if answer in ("y", "yes", "n", "no"):
            break
        print("  Please enter y or n.")

    print()

    if answer in ("y", "yes"):
        if _register_startup(_exe_path()):
            print("  ✓ Ellie will now start automatically with Windows.")
    else:
        print("  Ellie will not start automatically.")
        print("  You can run Ellie.exe directly at any time.")

    # ── Launch main app ─────────────────────────────────────────────────────
    print()
    print("  Starting Ellie...")

    # Hide the console window before the GUI takes over
    import ctypes
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE

    from ellie_claude import EllieApp
    EllieApp()


if __name__ == "__main__":
    main()
