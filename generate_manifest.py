import json
import os
import subprocess
import platform
import torch
import torch_geometric

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Unknown"

manifest = {
    "git_commit": get_git_commit(),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_geometric_version": torch_geometric.__version__,
    "seed": 42,
    "n_subjects": 100,
    "note": "Results generated from exact code state at the git commit above."
}

os.makedirs('results', exist_ok=True)
with open('results/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=4)
    
print("Saved results/manifest.json")
