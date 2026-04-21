from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def clear_proxy_env() -> None:
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"]:
        if key in os.environ:
            os.environ[key] = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 BERT 模型到本地目录（镜像优先）")
    parser.add_argument("--repo-id", default="bert-base-chinese", help="HuggingFace 模型仓库名")
    parser.add_argument("--output-dir", default="models/bert-base-chinese", help="本地输出目录")
    parser.add_argument(
        "--endpoints",
        nargs="*",
        default=[
            "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
            "https://hf-mirror.com",
        ],
        help="镜像端点，按顺序尝试",
    )
    args = parser.parse_args()

    clear_proxy_env()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for ep in args.endpoints:
        try:
            print(f"[try] endpoint={ep}")
            path = snapshot_download(
                repo_id=args.repo_id,
                local_dir=str(out_dir),
                endpoint=ep,
            )
            print(f"[ok] model saved to: {path}")
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[failed] endpoint={ep} err={type(exc).__name__}: {exc}")

    raise SystemExit(f"all endpoints failed: {last_err}")


if __name__ == "__main__":
    main()
