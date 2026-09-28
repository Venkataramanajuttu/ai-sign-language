# How to run
1. Install Docker Desktop and open it (wait for "Engine running").
2. Unzip this folder.
3. Windows: right-click run.ps1 -> Run with PowerShell.  Mac/Linux: ./run.sh
4. Wait for the first build; the app opens at http://localhost:8501 automatically.

What to demo (no camera needed):
- "Demo Gallery" tab: pick any of the 14 signs -> skeleton + live prediction.
- "Upload Image" tab: upload a clear photo of a hand sign.
- "Camera" tab: uses the BROWSER camera, so it works even inside Docker.
- "Model & Training" tab: accuracy, confusion matrix, per-class F1.
- Sidebar: sentence builder — add recognized signs to build a translated sentence.

Stop later: docker rm -f signlanguage
Retrain (offline, ~30s): python train_model.py
