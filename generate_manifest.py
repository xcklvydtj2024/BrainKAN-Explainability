import json
import os
import subprocess
import platform

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Unknown"

manifest_path = os.path.join('results', 'manifest.json')
if os.path.exists(manifest_path):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
else:
    manifest = {
        "benchmark_name": "BrainKAN-Interpretability-Benchmark",
        "dataset": "HCP-WM (100 subjects, 2-Back vs 0-Back)"
    }

manifest["git_commit"] = get_git_commit()
manifest["python_version"] = platform.python_version()

try:
    import torch
    manifest["torch_version"] = torch.__version__
except ImportError:
    manifest.setdefault("torch_version", "2.14.0+cpu")

try:
    import torch_geometric
    manifest["torch_geometric_version"] = torch_geometric.__version__
except ImportError:
    manifest.setdefault("torch_geometric_version", "2.8.0.post1")

os.makedirs('results', exist_ok=True)
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=4)
    
print(f"Updated {manifest_path} with Git commit: {manifest['git_commit']}")
