from pathlib import Path
import sys

# Resolve project imports when launched as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import run_visione


if __name__ == "__main__":
    run_visione(use_realsense=True)
