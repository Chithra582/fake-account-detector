"""
One-click setup: generates dataset, trains models, then starts the Flask server.
Run: python setup_and_run.py
"""
import subprocess, sys, os

def run(cmd, desc):
    print(f"\n{'='*55}\n▶  {desc}\n{'='*55}")
    result = subprocess.run([sys.executable] + cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌  Step failed: {desc}")
        sys.exit(1)
    print(f"✅  Done: {desc}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run(["generate_dataset.py"],   "Step 1 — Generate synthetic dataset")
    run(["train_model.py"],        "Step 2 — Train ML models")
    print("\n\n🚀  Starting Flask server at http://localhost:5000\n")
    os.execv(sys.executable, [sys.executable, "app.py"])
