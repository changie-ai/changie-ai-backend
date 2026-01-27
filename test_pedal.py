#!/usr/bin/env python3
"""
Quick test runner: runs main.py with optional prompt and prints exit.
"""
import subprocess
import sys

cmd = [sys.executable, "main.py"]
if len(sys.argv) > 1:
    cmd += ["--prompt", " ".join(sys.argv[1:])]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=False)
