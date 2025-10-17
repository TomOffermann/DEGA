
import numpy as np
from pathlib import Path

def print_umda_evals(data_dir="data", benchmark="LO"):
    folder = Path(data_dir) / "UMDA" / benchmark
    for npz in sorted(folder.glob("*.npz")):
        arr = np.load(npz, allow_pickle=True)
        # arr["results"] is an object‐array of dicts
        raw = arr["results"]
        # unpack each element (some may be 0‐d arrays, some dicts)
        evals = []
        for r in raw:
            if isinstance(r, np.ndarray):
                d = r.item()
            else:
                d = r
            evals.append(d["evals"])
        print(f"\nFile: {npz.name}\nEvals per rep: {evals}")

# usage:
if __name__ == "__main__":
    print_umda_evals()
