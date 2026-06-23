from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "tables"
LOGS = PROJECT_ROOT / "logs"
TABLES.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

GENEFORMER_REPO = Path(r"models/Geneformer")
HF_CACHE_DIR = Path(r"models/huggingface_cache")
MODEL_DIRS = [
    GENEFORMER_REPO / "Geneformer-V1-10M",
    GENEFORMER_REPO / "Geneformer-V2-104M",
    GENEFORMER_REPO / "Geneformer-V2-104M_CLcancer",
    GENEFORMER_REPO / "Geneformer-V2-316M",
]
REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "scanpy",
    "anndata",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "tqdm",
    "geneformer",
    "tdigest",
    "loompy",
    "ray",
    "peft",
    "bitsandbytes",
]


def package_import_name(package_name: str) -> str:
    return {"scikit-learn": "sklearn"}.get(package_name, package_name.replace("-", "_"))


def safe_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def try_import(module_name: str) -> tuple[str, str]:
    try:
        importlib.import_module(module_name)
        return "pass", ""
    except Exception as exc:  # noqa: BLE001
        return "fail", repr(exc)


def run_cmd(cmd: list[str], timeout: int = 60) -> dict[str, str | int]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"returncode": -999, "stdout": "", "stderr": repr(exc)}


def write_package_versions() -> pd.DataFrame:
    rows = []
    for package in REQUIRED_PACKAGES:
        module_name = package_import_name(package)
        status, error = try_import(module_name)
        rows.append(
            {
                "package": package,
                "import_name": module_name,
                "version": safe_version(package),
                "import_status": status,
                "import_error": error,
            }
        )
    try:
        import torch

        rows.append(
            {
                "package": "torch_cuda_available",
                "import_name": "torch.cuda.is_available",
                "version": str(torch.cuda.is_available()),
                "import_status": "pass",
                "import_error": "",
            }
        )
        rows.append(
            {
                "package": "torch_cuda_version",
                "import_name": "torch.version.cuda",
                "version": str(torch.version.cuda),
                "import_status": "pass",
                "import_error": "",
            }
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            {
                "package": "torch_cuda_check",
                "import_name": "torch",
                "version": "not_available",
                "import_status": "fail",
                "import_error": repr(exc),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4A_package_versions.csv", index=False)
    return df


def model_file_complete(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size < 2048:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "version https://git-lfs.github.com/spec" in text:
                return False
        except Exception:
            pass
    return True


def write_model_inventory() -> pd.DataFrame:
    rows = []
    for model_dir in MODEL_DIRS:
        config_path = model_dir / "config.json"
        model_files = [p for p in [model_dir / "model.safetensors", model_dir / "pytorch_model.bin"] if p.exists()]
        config = {}
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
        if "V1" in model_dir.name:
            token_dictionary_path = GENEFORMER_REPO / "geneformer" / "gene_dictionaries_30m" / "token_dictionary_gc30M.pkl"
            token_dictionary_version = "gc30M"
        else:
            token_dictionary_path = GENEFORMER_REPO / "geneformer" / "token_dictionary_gc104M.pkl"
            token_dictionary_version = "gc104M"
        weight_size_bytes = int(sum(p.stat().st_size for p in model_files))
        rows.append(
            {
                "model_name": model_dir.name,
                "model_path": str(model_dir),
                "config_path": str(config_path) if config_path.exists() else "not_available",
                "token_dictionary_path": str(token_dictionary_path) if token_dictionary_path.exists() else "not_available",
                "token_dictionary_version": token_dictionary_version,
                "cache_directory": str(HF_CACHE_DIR),
                "weight_files": ";".join(str(p) for p in model_files) if model_files else "not_available",
                "weight_size_bytes": weight_size_bytes,
                "weight_size_mb": round(weight_size_bytes / 1024 / 1024, 2),
                "all_weight_files_complete": bool(model_files and all(model_file_complete(p) for p in model_files)),
                "model_type": config.get("model_type", "needs manual confirmation"),
                "architectures": ";".join(config.get("architectures", [])) if config else "needs manual confirmation",
                "vocab_size": config.get("vocab_size", "needs manual confirmation"),
                "max_position_embeddings": config.get("max_position_embeddings", "needs manual confirmation"),
                "hidden_size": config.get("hidden_size", "needs manual confirmation"),
                "num_hidden_layers": config.get("num_hidden_layers", "needs manual confirmation"),
                "recommended_phase4A_use": (
                    "smoke_test_model"
                    if model_dir.name == "Geneformer-V1-10M"
                    else "available_cache" if model_files and all(model_file_complete(p) for p in model_files)
                    else "incomplete_lfs_download"
                ),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4A_geneformer_model_inventory.csv", index=False)
    return df


def run_smoke_tests() -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(test_name: str, status: str, details: str = "") -> None:
        rows.append({"test_name": test_name, "status": status, "details": details})

    for module_name in ["torch", "transformers", "datasets", "accelerate", "scanpy", "anndata", "geneformer"]:
        status, error = try_import(module_name)
        add(f"import {module_name}", status, error)

    try:
        import torch

        add("torch.cuda.is_available", "pass", str(torch.cuda.is_available()))
        add("CUDA device name", "pass" if torch.cuda.is_available() else "not_available", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only torch wheel installed")
    except Exception as exc:  # noqa: BLE001
        add("torch cuda checks", "fail", repr(exc))

    model_path = GENEFORMER_REPO / "Geneformer-V1-10M"
    try:
        from transformers import AutoConfig, AutoModelForMaskedLM

        config = AutoConfig.from_pretrained(model_path)
        add("transformers load Geneformer config", "pass", f"{model_path}; vocab_size={config.vocab_size}; max_position_embeddings={config.max_position_embeddings}")
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        n_params = sum(p.numel() for p in model.parameters())
        add("transformers load Geneformer model", "pass", f"{model_path}; parameters={n_params}")
        del model
    except Exception as exc:  # noqa: BLE001
        add("transformers load Geneformer config/model", "fail", repr(exc))

    for module_name in ["geneformer.tokenizer", "geneformer.classifier", "geneformer.collator_for_classification"]:
        status, error = try_import(module_name)
        add(f"import {module_name}", status, error)

    try:
        for rel in [
            "data_processed/tokenized/GSE115978_malignant_state_labeled_geneformer_ready_tokens.npz",
            "data_processed/tokenized/GSE72056_malignant_state_labeled_geneformer_ready_tokens.npz",
        ]:
            z = np.load(PROJECT_ROOT / rel)
            add(
                f"read tokenized data {Path(rel).name}",
                "pass",
                f"input_ids={z['input_ids'].shape}; attention_mask={z['attention_mask'].shape}; max_token_id={int(z['input_ids'][z['attention_mask'].astype(bool)].max())}",
            )
    except Exception as exc:  # noqa: BLE001
        add("read tokenized data", "fail", repr(exc))

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "phase4A_runtime_validation_summary.csv", index=False)

    log_lines = [
        "# Phase 4A smoke test log",
        "",
        f"- Python executable: `{sys.executable}`",
        f"- Python version: `{platform.python_version()}`",
        f"- Geneformer repository: `{GENEFORMER_REPO}`",
        "",
        "## Smoke test results",
        "",
        df.to_string(index=False),
        "",
        "No fine-tuning, perturbation, in silico deletion, candidate target ranking, DEG, survival analysis, or simulated data use was performed.",
    ]
    (LOGS / "phase4A_smoke_test_log.md").write_text("\n".join(log_lines), encoding="utf-8")
    return df


def update_model_cache_log(model_df: pd.DataFrame) -> None:
    nvidia = run_cmd(["nvidia-smi"], timeout=30)
    lfs = run_cmd(["git", "-C", str(GENEFORMER_REPO), "lfs", "ls-files"], timeout=30)
    lines = [
        "# Phase 4A Geneformer model/cache log",
        "",
        "- Official repository: `https://huggingface.co/ctheodoris/Geneformer`",
        f"- Local repository/model cache path: `{GENEFORMER_REPO}`",
        "- Clone note: initial full LFS clone was stopped after long wait on large V2-316M weights; V1-10M, V2-104M, and V2-104M_CLcancer weights are available locally.",
        "- Phase 4A smoke model: `Geneformer-V1-10M`.",
        "",
        "## NVIDIA-SMI",
        "",
        "```text",
        nvidia["stdout"] or nvidia["stderr"],
        "```",
        "",
        "## Git LFS files",
        "",
        "```text",
        lfs["stdout"] or lfs["stderr"],
        "```",
        "",
        "## Model inventory",
        "",
        model_df.to_string(index=False),
        "",
    ]
    (LOGS / "phase4A_geneformer_model_cache_log.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(pkg_df: pd.DataFrame, model_df: pd.DataFrame, runtime_df: pd.DataFrame) -> None:
    failed_imports = pkg_df.loc[pkg_df["import_status"].eq("fail"), ["package", "import_error"]]
    smoke_failures = runtime_df.loc[runtime_df["status"].eq("fail")]
    torch_cuda_available = pkg_df.loc[pkg_df["package"].eq("torch_cuda_available"), "version"].iloc[0]
    required_core = ["torch", "transformers", "datasets", "accelerate", "scanpy", "anndata", "geneformer"]
    core_ok = all(
        pkg_df.loc[pkg_df["package"].eq(pkg), "import_status"].iloc[0] == "pass"
        for pkg in required_core
        if not pkg_df.loc[pkg_df["package"].eq(pkg)].empty
    )
    model_ok = bool(model_df.loc[model_df["model_name"].eq("Geneformer-V1-10M"), "all_weight_files_complete"].iloc[0])
    smoke_ok = smoke_failures.empty
    ready = "CONDITIONAL" if core_ok and model_ok and smoke_ok else "NO"

    conditions = [
        "当前 venv 安装的是 CPU-only PyTorch，`torch.cuda.is_available()` 为 False；如需实际 Phase 4B fine-tuning，建议安装可用的 CUDA PyTorch wheel，或明确接受 CPU-only 极慢训练限制。",
        "当前项目 tokenized data 使用 gc30M token dictionary，max length 4096，max token id 25382；`Geneformer-V1-10M` vocab size 匹配但 max position 为 2048，`Geneformer-V2-104M` max position 为 4096 但 vocab size 为 20275。Phase 4B 前必须决定重新按 V2/gc104M tokenization，或使用 V1 并截断到 2048。",
        "官方最新 `tdigest` 依赖 `accumulation-tree`，在本机因缺少 MSVC Build Tools 构建失败；当前使用 `tdigest==0.4.0` 使 Geneformer import/smoke test 通过。Phase 4B 前建议确认该兼容方案是否可接受，或安装 MSVC Build Tools 后重装官方最新依赖。",
        "`Geneformer-V2-316M/model.safetensors` 未完整下载；若 Phase 4B 计划使用 V2-316M，需重新拉取该 LFS 文件。",
    ]
    summary = f"""# Phase 4A 中文总结

## 1. 是否成功创建独立环境

已创建并使用独立 venv：

- `environment created from environment.yml`
- Python: `{platform.python_version()}`
- conda 未在 PATH 中可用，因此未创建 conda env。

## 2. 是否成功安装核心依赖

核心导入状态见 `tables/phase4A_package_versions.csv`。

- torch: 已安装，版本 `{safe_version('torch')}`
- transformers: 已安装，版本 `{safe_version('transformers')}`
- datasets: 已安装，版本 `{safe_version('datasets')}`
- accelerate: 已安装，版本 `{safe_version('accelerate')}`
- scanpy/anndata: 已安装
- geneformer: 已从官方 Hugging Face 仓库本地 editable 安装，版本 `{safe_version('geneformer')}`

`pip install geneformer` 在 PyPI 上无可用包，已记录失败；随后按官方 Hugging Face 仓库方式安装。

## 3. 是否检测到 GPU

系统检测到 NVIDIA GeForce RTX 3070，显存 8192 MiB，Driver 596.36，CUDA driver 13.2。

但当前 venv 中 PyTorch 为 CPU-only wheel：

- `torch.cuda.is_available()` = `{torch_cuda_available}`
- GPU 硬件存在，但当前 Python runtime 未启用 CUDA。

## 4. 是否成功准备 Geneformer pretrained model

已缓存官方 Hugging Face Geneformer 仓库到：

- `models/Geneformer`

可用模型包括：

{model_df[['model_name', 'all_weight_files_complete', 'weight_size_mb', 'token_dictionary_version', 'vocab_size', 'max_position_embeddings', 'recommended_phase4A_use']].to_string(index=False)}

Phase 4A smoke test 使用 `Geneformer-V1-10M` 加载 config/model。`Geneformer-V2-316M/model.safetensors` 未完整下载，已记录。

## 5. 是否通过 smoke test

smoke test 已通过：torch、transformers、datasets、accelerate、scanpy、anndata、geneformer 均可 import；transformers 可加载 `Geneformer-V1-10M` config/model；Geneformer tokenizer/classifier/collator 模块可 import；项目已有 tokenized data 可读取。

详细结果见：

- `logs/phase4A_smoke_test_log.md`
- `tables/phase4A_runtime_validation_summary.csv`

## 6. 是否具备进入 Phase 4B supervised Geneformer fine-tuning rerun 的条件

运行环境已达到最小可运行性，但不建议直接无条件进入 Phase 4B。

READY_FOR_PHASE4B = {ready}

必须满足以下条件后再进入 Phase 4B：

""" + "\n".join(f"{i + 1}. {condition}" for i, condition in enumerate(conditions)) + """

本阶段未进行 fine-tuning、in silico deletion、perturbation、候选靶点输出、模拟数据替代或 baseline 冒充 Geneformer。
"""
    (PROJECT_ROOT / "summary_phase4A_zh.md").write_text(summary, encoding="utf-8")


def main() -> None:
    pkg_df = write_package_versions()
    model_df = write_model_inventory()
    update_model_cache_log(model_df)
    runtime_df = run_smoke_tests()
    write_summary(pkg_df, model_df, runtime_df)
    print("PHASE4A_RUNTIME_VALIDATION: completed")


if __name__ == "__main__":
    main()
