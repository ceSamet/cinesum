import os
import sys
from pathlib import Path


def _relaunch_in_project_venv() -> None:
    """Use the shared Linux environment after the Samet/Gökdeniz merge."""
    base_dir = Path(__file__).resolve().parent
    candidates = (
        base_dir.parent / ".venv" / "Scripts" / "python.exe",
        base_dir.parent / "gokedniz" / "cinesum" / ".venv" / "Scripts" / "python.exe",
        base_dir.parent / ".venv" / "bin" / "python",
    ) if os.name == "nt" else (base_dir.parent / ".venv" / "bin" / "python",)
    venv_python = next((candidate for candidate in candidates if candidate.exists()), None)
    if venv_python is None:
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

    host = os.getenv("THANOS_HOST", "127.0.0.1")
    port = int(os.getenv("THANOS_PORT", "8001"))
    url = f"http://{host}:{port}"
    print("==================================================================")
    print(" CineSum AI -- Full-Stack Web Dashboard Sunucusu Baslatiliyor...")
    print(f" Canli Web Adresi: {url}")
    print(
        " GPU: "
        + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback")
    )
    print("==================================================================")

    # Launch uvicorn web server
    uvicorn.run("app:app", host=host, port=port, reload=False   )

if __name__ == "__main__":
    launch_server()
