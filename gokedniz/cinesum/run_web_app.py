import os
import sys
from pathlib import Path


def _relaunch_in_project_venv() -> None:
    """Make the ordinary `python run_web_app.py` command use the CUDA environment."""
    base_dir = Path(__file__).resolve().parent
    venv_python = base_dir / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return
    try:
        current = Path(sys.executable).resolve()
        target = venv_python.resolve()
    except OSError:
        return
    if current == target or os.getenv("CINESUM_SKIP_VENV_REEXEC") == "1":
        return
    os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])


def launch_server():
    _relaunch_in_project_venv()
    import torch
    import uvicorn

    base_dir = Path(__file__).resolve().parent
    sys.path.append(str(base_dir))

    url = "http://localhost:8000"
    print("==================================================================")
    print(" CineSum AI -- Full-Stack Web Dashboard Sunucusu Baslatiliyor...")
    print(f" Canli Web Adresi: {url}")
    print(
        " GPU: "
        + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback")
    )
    print("==================================================================")

    # Launch uvicorn web server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    launch_server()
