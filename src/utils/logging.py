import numpy as np

RESET = "\033[0m"
BOLD  = "\033[1m"
def b(s): return f"{BOLD}{s}{RESET}"

def mean_std(neg: np.array, pos: np.array):
    print(f"Control:    {neg.mean():6.4f} (+/- {neg.std():6.4f})  | Count: {len(neg):4}")
    print(f"ProbableAD: {pos.mean():6.4f} (+/- {pos.std():6.4f})  | Count: {len(pos):4}")
