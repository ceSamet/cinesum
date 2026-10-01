from __future__ import annotations

import argparse

from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import snapshot_download


DEFAULT_MODELS = {
    "blip": "Salesforce/blip-image-captioning-base",
    # 0.5B LLaVA is deliberately selected for a 6 GB laptop GPU. The old 7B
    # default is far too slow/heavy for near-real-time episode analysis.
    "llava": "llava-hf/llava-interleave-qwen-0.5b-hf",
    "whisper": "Systran/faster-whisper-small",
    "audio-events": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "embeddings": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

OPENCV_MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PoC vision-language models.")
    parser.add_argument(
        "models",
        nargs="*",
        choices=sorted([*DEFAULT_MODELS, "faces"]),
        default=[*DEFAULT_MODELS, "faces"],
        help="Models to download. Default: all",
    )
    args = parser.parse_args()

    for name in args.models:
        if name == "faces":
            target_dir = Path(__file__).resolve().parent.parent / "models" / "opencv"
            target_dir.mkdir(parents=True, exist_ok=True)
            for filename, url in OPENCV_MODELS.items():
                target = target_dir / filename
                if target.exists() and target.stat().st_size > 0:
                    print(f"Already downloaded faces: {target}", flush=True)
                    continue
                print(f"Downloading faces: {filename}", flush=True)
                urlretrieve(url, target)
                print(f"Downloaded faces to: {target}", flush=True)
            continue
        repo_id = DEFAULT_MODELS[name]
        target_dir = Path(__file__).resolve().parent.parent / "models" / name
        print(f"Downloading {name}: {repo_id}", flush=True)
        ignored = ["*.h5", "*.msgpack", "*.ot", "*.gguf", "onnx/*", "openvino/*"]
        if name in {"audio-events", "embeddings"}:
            ignored.append("pytorch_model.bin")  # safetensors is present
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            # Do not fetch duplicate TensorFlow/Flax/GGUF/ONNX exports when the
            # runtime uses PyTorch or CTranslate2 weights.
            ignore_patterns=ignored,
        )
        print(f"Downloaded {name} to: {path}", flush=True)


if __name__ == "__main__":
    main()
