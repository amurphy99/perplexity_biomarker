import numpy as np

RESET = "\033[0m"
BOLD  = "\033[1m"
def b(s): return f"{BOLD}{s}{RESET}"

def mean_std(neg: np.array, pos: np.array):
    print(f"Control:    {neg.mean():6.4f} (+/- {neg.std():6.4f})  | Count: {len(neg):4}")
    print(f"ProbableAD: {pos.mean():6.4f} (+/- {pos.std():6.4f})  | Count: {len(pos):4}")

def format_time(seconds: float | int | None) -> str:
    """MM:SS for a number of seconds. Returns "—" if None."""
    if seconds is None: return "—"
    mins = int(seconds) // 60
    secs = int(seconds)  % 60
    return f"{mins:02d}:{secs:02d}"
