from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_ZS_MODELS = [
    "intfloat/multilingual-e5-base",
    "BAAI/bge-m3",
    "antoinelouis/biencoder-camembert-base-mmarcoFR",
]

DEFAULT_FINETUNE = [
    ("e5-base-bsard", "intfloat/multilingual-e5-base", 10, 32, False),
    ("bge-m3-bsard", "BAAI/bge-m3", 5, 8, True),
]


def log(msg: str, level: str = "") -> None:
    prefix = f"{level} " if level else ""
    print(f"[{time.strftime('%H:%M:%S')}] {prefix}{msg}", flush=True)


def rule(title: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 72, flush=True)


def run(cmd: list[str], dry_run: bool = False) -> int:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    log(printable)
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(ROOT))


def detect_gpu() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0)
    return False, "torch is installed, no CUDA GPU visible"


def preflight(args) -> dict:
    rule("0. Preflight checks")
    info: dict = {}

    log(f"python {sys.version.split()[0]}")
    missing = []
    for pkg in ("numpy", "scipy", "pandas", "sklearn"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(f"Missing packages: {missing}. Run: pip install -e .")

    has_torch = True
    try:
        import sentence_transformers
    except ImportError:
        has_torch = False
    info["has_torch"] = has_torch

    gpu, gpu_name = detect_gpu()
    info["gpu"] = gpu
    log(f"GPU: {gpu_name}")
    if not has_torch:
        log("sentence-transformers is missing: only BM25 and TF-IDF will be "
            "evaluated. Install torch and sentence-transformers for the rest.",
            "WARNING")

    data = Path(args.data_dir)
    needed = ["articles_fr.csv", "questions_fr_train.csv", "questions_fr_test.csv"]
    absent = [f for f in needed if not (data / f).exists()]
    if absent:
        sys.exit(f"Missing files in {data}: {absent}\n"
                 "Run: python scripts/download_bsard.py --out "
                 f"{args.data_dir}")
    synth = (data / "questions_fr_synthetic.csv").exists()
    info["synthetic"] = synth
    log(f"data: {data.resolve()}")
    if not synth and not args.skip_finetune:
        log("questions_fr_synthetic.csv is missing. Fine-tuning on the 886 "
            "annotated questions alone plateaus well below the published "
            "supervised baselines. Run download_bsard.py again.", "WARNING")

    if gpu:
        info["max_seq_length"] = args.max_seq_length or 512
        info["batch_size"] = args.batch_size or 32
    else:
        info["max_seq_length"] = args.max_seq_length or 256
        info["batch_size"] = args.batch_size or 16
        log(f"no GPU: sequence length reduced to "
            f"{info['max_seq_length']}, batches of {info['batch_size']}")
    return info


def stage_finetune(args, info) -> list[str]:
    rule("1-2. Fine-tuning")
    produced: list[str] = []

    if args.skip_finetune:
        log("stage skipped (--skip-finetune)")
        return produced
    if not info["has_torch"]:
        log("stage skipped: sentence-transformers is missing")
        return produced

    for name, base_model, epochs, batch, gpu_only in DEFAULT_FINETUNE:
        out_dir = Path(args.models_dir) / name
        if gpu_only and not info["gpu"]:
            log(f"{name} skipped: this model requires a GPU "
                "(CPU training is not reasonable)")
            continue
        if out_dir.exists() and not args.force:
            log(f"{name} already present, reused ({out_dir})")
            produced.append(str(out_dir))
            continue

        cmd = [sys.executable, "scripts/finetune_dense.py",
               "--data-dir", args.data_dir,
               "--model", base_model,
               "--out", str(out_dir),
               "--hard-negatives", "4",
               "--epochs", str(epochs),
               "--batch-size", str(batch),
               "--max-seq-length", str(info["max_seq_length"])]
        if info["synthetic"]:
            cmd += ["--use-synthetic", "--synthetic-max",
                    str(args.synthetic_max)]
        if not info["gpu"]:
            cmd += ["--max-positives", "2"]

        t0 = time.time()
        code = run(cmd, args.dry_run)
        if code != 0 and not args.dry_run:
            log(f"fine-tuning of {name} failed (code {code}). "
                "Continuing without this model.", "WARNING")
            shutil.rmtree(out_dir, ignore_errors=True)
            continue
        log(f"{name} trained in {(time.time() - t0) / 60:.1f} min")
        produced.append(str(out_dir))

    return produced


def stage_experiments(args, info, finetuned: list[str]) -> int:
    rule("3. Full experiments")

    cmd = [sys.executable, "scripts/run_paper.py",
           "--data-dir", args.data_dir,
           "--out-dir", args.out_dir,
           "--tfidf",
           "--max-seq-length", str(info["max_seq_length"]),
           "--batch-size", str(info["batch_size"])]

    if info["has_torch"]:
        cmd += ["--dense-models"] + (args.zs_models or DEFAULT_ZS_MODELS)
    if finetuned:
        cmd += ["--finetuned-models"] + finetuned
    if args.force:
        cmd += ["--force"]
    if not info["gpu"]:
        cmd += ["--threads", str(args.threads)]

    return run(cmd, args.dry_run)


def stage_summary(args) -> None:
    rule("4. Summary")
    out = Path(args.out_dir)
    paper = out / "paper.md"
    if not paper.exists():
        log("paper.md not found: stage 3 failed.", "WARNING")
        return

    text = paper.read_text(encoding="utf-8")
    if "## Completeness check" in text:
        block = text.split("## Completeness check", 1)[1]
        print("\nCompleteness check:")
        for line in block.strip().splitlines():
            if line.startswith("- "):
                print("  " + line[2:])
            elif not line.strip():
                break

    if "## Table 2b" in text:
        block = text.split("## Table 2b", 1)[1].split("\n## ", 1)[0]
        print("\nResults summary:")
        for line in block.splitlines():
            if line.startswith("|"):
                print("  " + line)

    print()
    for f in ("paper.md", "tables.tex", "figures.json", "results.json"):
        p = out / f
        if p.exists():
            log(f"{f:16} {p.stat().st_size / 1024:8.1f} kB  {p.resolve()}")

    try:
        raw = json.loads((out / "results.json").read_text(encoding="utf-8"))
        pareto = [(sysname, label)
                  for sysname, d in raw.get("systems", {}).items()
                  for label, v in d.get("variants", {}).items()
                  if v.get("pareto")]
        if pareto:
            print("\nConfigurations improving all seven metrics:")
            for sysname, label in pareto:
                print(f"  - {label}")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full pipeline: fine-tuning, then all experiments.")
    ap.add_argument("--data-dir", default="data/bsard")
    ap.add_argument("--out-dir", default="paper_final")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--zs-models", nargs="*", default=None,
                    help="replace the list of zero-shot encoders")
    ap.add_argument("--synthetic-max", type=int, default=30000)
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="512 with a GPU, 256 without, by default")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU threads when no GPU is available")
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="retrain and recompute even if the cache exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running anything")
    args = ap.parse_args()

    t_start = time.time()
    info = preflight(args)
    finetuned = stage_finetune(args, info)
    code = stage_experiments(args, info, finetuned)
    if code != 0 and not args.dry_run:
        sys.exit(f"The experiments stage failed (code {code}).")
    if not args.dry_run:
        stage_summary(args)

    rule(f"Finished in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
