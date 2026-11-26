"""
One-command runs: ``ponderosa run DATASET_DIR`` and ``ponderosa status DATASET_DIR``.

Put a dataset's files in one folder::

    MY_DATASET/
        cohort.ibd.gz            hap-IBD (or phasedIBD) output; per-chromosome files also work
        cohort.fam               PLINK FAM
        genetic.map              PLINK map (chr rsid cM bp), required for hap-IBD
        training.classif.pkl     optional: pretrained / simulation-trained classifier
        cohort.kin0              optional: KING kinship, for validation
        ages.txt                 optional: IID AGE
        ponderosa.yaml           optional: overrides (see below)

``ponderosa run MY_DATASET`` finds the inputs, checks them on the spot, writes a Slurm
job script (``MY_DATASET/ponderosa_job.sbatch``), submits it and records the job ID.
The job runs inference, duplicate QC, ranked family trees and (with a KING file) KING
validation; results go to ``MY_DATASET/outputs/``.

``ponderosa.yaml`` may set any of::

    name: SAC_MEGA                 # output prefix (default: folder name)
    ibd: cohort.ibd.gz             # or a list of files
    ibd_caller: hap-ibd            # hap-ibd | phasedibd
    fam: cohort.fam
    map: genetic.map               # or a list of files
    training: sim.classif.pkl
    king: cohort.kin0
    ages: ages.txt
    min_segment_length: 3
    min_total_ibd: 50
    genome_length: 3545
    close_only_king: true
    slurm: {account: bmhenngrp, partition: high, mem: 32G, cpus: 4, time: "04:00:00"}
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DEFAULT_SLURM = {"account": None, "partition": "high", "mem": "32G", "cpus": 4, "time": "04:00:00"}
STATE_FILE = ".ponderosa_run.json"


class RunSetupError(ValueError):
    pass


# --------------------------------------------------------------------------
# Input discovery
# --------------------------------------------------------------------------

def _files(folder: Path) -> List[Path]:
    skip = {"outputs", "logs", ".git", "venv"}
    out = []
    for p in sorted(folder.iterdir()):
        if p.name.startswith(".") or p.name in skip:
            continue
        if p.is_file() or (p.is_symlink() and p.exists()):
            out.append(p)
    return out


def _one(cands: List[Path], what: str, hint: str, required: bool = True) -> Optional[Path]:
    if len(cands) == 1:
        return cands[0]
    if not cands:
        if required:
            raise RunSetupError(f"No {what} found. {hint}")
        return None
    names = ", ".join(c.name for c in cands)
    raise RunSetupError(f"Several {what} candidates ({names}). Choose one in ponderosa.yaml, e.g. {hint}")


def detect_ibd_caller(path: Path) -> str:
    """hap-IBD: 8 tab-separated columns, no header, integer bp positions. Else phasedIBD."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                break
        else:
            raise RunSetupError(f"IBD file is empty: {path}")
    fields = line.rstrip("\n").split("\t")
    if len(fields) == 8 and fields[4].strip() and fields[5].isdigit() and fields[6].isdigit():
        return "hap-ibd"
    return "phasedibd"


def discover_inputs(folder: Path) -> Dict:
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise RunSetupError(f"Not a folder: {folder}")

    overrides = {}
    cfg = folder / "ponderosa.yaml"
    if cfg.exists():
        overrides = yaml.safe_load(cfg.read_text()) or {}

    files = _files(folder)
    names = lambda pat: [p for p in files if re.search(pat, p.name)]

    def resolve(v):
        if v is None:
            return None
        if isinstance(v, list):
            return [str((folder / x).resolve()) if not os.path.isabs(x) else x for x in v]
        return str((folder / v).resolve()) if not os.path.isabs(str(v)) else str(v)

    inp: Dict = {"folder": str(folder), "name": overrides.get("name", folder.name)}

    # IBD
    if "ibd" in overrides:
        inp["ibd"] = resolve(overrides["ibd"])
    else:
        ibd = [p for p in names(r"\.ibd(\.gz)?$") if not re.search(r"\.hbd(\.gz)?$", p.name)]
        if not ibd:
            ibd = names(r"(phased_?ibd|phasedibd).*\.(txt|tsv)(\.gz)?$")
        chrom = [p for p in ibd if re.search(r"chr\d+", p.name)]
        if len(chrom) >= 2 and len(chrom) == len(ibd):
            inp["ibd"] = [str(p) for p in sorted(chrom, key=lambda p: int(re.search(r"chr(\d+)", p.name).group(1)))]
        else:
            inp["ibd"] = str(_one(ibd, "IBD file", "ibd: cohort.ibd.gz"))
    first_ibd = Path(inp["ibd"][0] if isinstance(inp["ibd"], list) else inp["ibd"])
    inp["ibd_caller"] = overrides.get("ibd_caller") or detect_ibd_caller(first_ibd)

    inp["fam"] = resolve(overrides["fam"]) if "fam" in overrides else \
        str(_one(names(r"\.fam$"), "FAM file", "fam: cohort.fam"))

    if "map" in overrides:
        inp["map"] = resolve(overrides["map"])
    else:
        maps = [p for p in names(r"\.map$") if not p.name.endswith(".simmap")]
        chrom = [p for p in maps if re.search(r"chr\d+", p.name)]
        if len(chrom) >= 2 and len(chrom) == len(maps):
            inp["map"] = [str(p) for p in sorted(chrom, key=lambda p: int(re.search(r"chr(\d+)", p.name).group(1)))]
        else:
            m = _one(maps, "genetic map", "map: genetic.map", required=inp["ibd_caller"] == "hap-ibd")
            inp["map"] = str(m) if m else None

    for key, pat, hint in (("training", r"\.classif\.pkl$", "training: sim.classif.pkl"),
                           ("king", r"\.kin0$", "king: cohort.kin0"),
                           ("ages", r"(^|[_.])ages?\b.*\.(txt|tsv)$", "ages: ages.txt")):
        if key in overrides:
            inp[key] = resolve(overrides[key])
        else:
            hit = _one(names(pat), key, hint, required=False)
            inp[key] = str(hit) if hit else None

    inp["min_segment_length"] = float(overrides.get("min_segment_length", 3))
    inp["min_total_ibd"] = float(overrides.get("min_total_ibd", 50))
    inp["genome_length"] = float(overrides.get("genome_length", 3545))
    inp["close_only_king"] = bool(overrides.get("close_only_king", True))
    inp["iid_only_king"] = bool(overrides.get("iid_only_king", False))
    slurm = dict(DEFAULT_SLURM)
    slurm.update(overrides.get("slurm", {}) or {})
    inp["slurm"] = slurm
    return inp


# --------------------------------------------------------------------------
# Validation and job script
# --------------------------------------------------------------------------

def _file_args(inp: Dict) -> List[str]:
    args = []
    if isinstance(inp["ibd"], list):
        raise RunSetupError("internal: list IBD handled through a YAML config")
    args += ["--ibd", inp["ibd"], "--fam", inp["fam"], "--ibd-caller", inp["ibd_caller"]]
    if inp.get("map"):
        args += ["--map", inp["map"]]
    if inp.get("ages"):
        args += ["--ages", inp["ages"]]
    return args


def write_run_config(inp: Dict, out_prefix: Path) -> Path:
    """PONDEROSA YAML for the job (handles single or per-chromosome IBD/map files)."""
    files = {"fam": inp["fam"], "ibd_caller": inp["ibd_caller"]}
    files["ibd_files" if isinstance(inp["ibd"], list) else "ibd"] = inp["ibd"]
    if inp.get("map"):
        files["map_files" if isinstance(inp["map"], list) else "mapf"] = inp["map"]
    for k in ("ages", "training"):
        if inp.get(k):
            files[k] = inp[k]
    cfg = {
        "files": files,
        "algorithm": {"min_segment_length": inp["min_segment_length"],
                      "min_total_ibd": inp["min_total_ibd"],
                      "genome_length": inp["genome_length"]},
        "output": {"output": str(out_prefix), "verbose": 1},
    }
    path = Path(inp["folder"]) / "ponderosa_run_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def validate_inputs(inp: Dict, config_path: Path):
    from .config import PonderosaConfig
    from .input_validation import validate_run_inputs, format_validation_report
    config = PonderosaConfig.from_yaml(config_path)
    config.validate()
    report = validate_run_inputs(config, king_file=inp.get("king"))
    return report, format_validation_report(report)


def job_script(inp: Dict, config_path: Path, out_prefix: Path) -> str:
    s = inp["slurm"]
    folder = Path(inp["folder"])
    py = shlex.quote(sys.executable)
    q = lambda x: shlex.quote(str(x))
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=ponderosa_{inp['name']}",
        f"#SBATCH --time={s['time']}",
        f"#SBATCH --mem={s['mem']}",
        f"#SBATCH --cpus-per-task={s['cpus']}",
        f"#SBATCH --output={folder}/logs/ponderosa_%j.out",
        f"#SBATCH --error={folder}/logs/ponderosa_%j.err",
    ]
    if s.get("partition"):
        lines.append(f"#SBATCH --partition={s['partition']}")
    if s.get("account"):
        lines.append(f"#SBATCH --account={s['account']}")
    pairs = f"{out_prefix}_pairs.txt"
    lines += [
        "set -euo pipefail",
        f"cd {q(folder)}",
        "export PYTHONPYCACHEPREFIX=/tmp/${USER:-$(id -un)}_pycache",
        'echo "[$(date)] PONDEROSA run started on $(hostname)"',
        "",
        "echo '== 1/4 inference'",
        f"{py} -m ponderosa --config {q(config_path)} -v",
        "",
        "echo '== 2/4 duplicate QC'",
        f"{py} -m ponderosa qc-duplicates --pairs {q(pairs)} --output {q(out_prefix)} "
        f"--genome-length {inp['genome_length']}",
        "",
        "echo '== 3/4 ranked family trees'",
        f"{py} -m ponderosa pedigree --pairs {q(pairs)} --fam {q(inp['fam'])} --output {q(out_prefix)} "
        f"--genome-length {inp['genome_length']}" + (f" --ages {q(inp['ages'])}" if inp.get("ages") else ""),
    ]
    if inp.get("king"):
        lines += [
            "",
            "echo '== 4/4 KING validation'",
            f"{py} -m ponderosa king-validate --pairs {q(pairs)} --king {q(inp['king'])} --output {q(out_prefix)}"
            + (" --close-only" if inp["close_only_king"] else "") + (" --iid-only" if inp["iid_only_king"] else ""),
        ]
    else:
        lines += ["", "echo '== 4/4 KING validation skipped (no .kin0 in the folder)'"]
    lines += ["", f'echo "[$(date)] PONDEROSA run finished: {folder}/outputs/"', ""]
    return "\n".join(lines)


def _guess_account() -> Optional[str]:
    """First Slurm account for this user (HIVE prints these at login)."""
    if not shutil.which("sacctmgr"):
        return None
    try:
        out = subprocess.run(["sacctmgr", "-nP", "show", "associations", f"where user={os.environ.get('USER', '')}",
                              "format=account"], capture_output=True, text=True, timeout=20).stdout
        accts = [a.strip() for a in out.splitlines() if a.strip()]
        grp = [a for a in accts if a.endswith("grp")]
        return (grp or accts or [None])[0]
    except Exception:
        return None


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_run(argv: List[str]) -> int:
    p = argparse.ArgumentParser(prog="ponderosa run",
                                description="Find a dataset's inputs, check them and submit the whole pipeline to Slurm.")
    p.add_argument("folder", type=Path, help="Folder holding the dataset's input files")
    p.add_argument("--account", help="Slurm account (default: ponderosa.yaml, else your first *grp account)")
    p.add_argument("--partition", help="Slurm partition (default: high)")
    p.add_argument("--mem", help="Memory, e.g. 32G")
    p.add_argument("--cpus", type=int, help="CPUs per task")
    p.add_argument("--time", help="Time limit, e.g. 04:00:00")
    p.add_argument("--min-total-ibd", type=float, help="Override min_total_ibd (cM)")
    p.add_argument("--dry-run", action="store_true", help="Check inputs and write the job script, but do not submit")
    p.add_argument("--local", action="store_true", help="Run the job script here instead of submitting (small data only)")
    a = p.parse_args(argv)

    try:
        inp = discover_inputs(a.folder)
    except RunSetupError as e:
        print(f"ERROR: {e}")
        return 2
    for k in ("account", "partition", "mem", "cpus", "time"):
        if getattr(a, k):
            inp["slurm"][k] = getattr(a, k)
    if a.min_total_ibd is not None:
        inp["min_total_ibd"] = a.min_total_ibd
    if not inp["slurm"].get("account") and not a.local:
        inp["slurm"]["account"] = _guess_account()

    folder = Path(inp["folder"])
    (folder / "outputs").mkdir(exist_ok=True)
    (folder / "logs").mkdir(exist_ok=True)
    out_prefix = folder / "outputs" / inp["name"]

    show = lambda v: ", ".join(Path(x).name for x in v) if isinstance(v, list) else (Path(v).name if v else "-")
    print(f"Dataset: {inp['name']}  ({folder})")
    for k in ("ibd", "ibd_caller", "fam", "map", "training", "king", "ages"):
        val = inp.get(k)
        print(f"  {k:<11} {val if k == 'ibd_caller' else show(val)}")
    print(f"  {'filters':<11} min_segment_length={inp['min_segment_length']} cM, "
          f"min_total_ibd={inp['min_total_ibd']} cM")

    config_path = write_run_config(inp, out_prefix)
    print("\nChecking inputs...")
    try:
        report, text = validate_inputs(inp, config_path)
    except Exception as e:  # config-level problems
        print(f"ERROR: {e}")
        return 2
    print(text)
    if not report.ok:
        print("\nFix the errors above (or set the right files in ponderosa.yaml) and run again.")
        return 1

    script = folder / "ponderosa_job.sbatch"
    script.write_text(job_script(inp, config_path, out_prefix))
    script.chmod(0o755)
    print(f"\nJob script: {script}")

    if a.dry_run:
        print("Dry run: not submitted. Submit with: sbatch " + str(script))
        return 0
    if a.local:
        print("Running locally...")
        return subprocess.call(["bash", str(script)])
    if not shutil.which("sbatch"):
        print("ERROR: sbatch not found. Run this on the cluster login node, or use --local / --dry-run.")
        return 2
    res = subprocess.run(["sbatch", str(script)], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR: sbatch failed: {res.stderr.strip() or res.stdout.strip()}")
        return 1
    m = re.search(r"(\d+)", res.stdout)
    job_id = m.group(1) if m else None
    (folder / STATE_FILE).write_text(json.dumps(
        {"job_id": job_id, "submitted": datetime.now().isoformat(timespec="seconds"),
         "name": inp["name"], "out_prefix": str(out_prefix)}, indent=1))
    print(f"Submitted job {job_id}. Check on it with:  ponderosa status {a.folder}")
    return 0


def cmd_status(argv: List[str]) -> int:
    p = argparse.ArgumentParser(prog="ponderosa status", description="Show the state of a `ponderosa run` job.")
    p.add_argument("folder", type=Path)
    p.add_argument("--lines", type=int, default=15, help="Log lines to show")
    a = p.parse_args(argv)
    folder = a.folder.resolve()
    state_file = folder / STATE_FILE
    if not state_file.exists():
        print(f"No submitted run found in {folder}. Start one with: ponderosa run {a.folder}")
        return 1
    st = json.loads(state_file.read_text())
    jid = st.get("job_id")
    print(f"{st.get('name')}: job {jid}, submitted {st.get('submitted')}")
    state = None
    if shutil.which("squeue"):
        q = subprocess.run(["squeue", "-h", "-j", str(jid), "-o", "%T %M"], capture_output=True, text=True).stdout.strip()
        if q:
            state = q
    if state is None and shutil.which("sacct"):
        q = subprocess.run(["sacct", "-n", "-X", "-j", str(jid), "-o", "State%20,Elapsed"],
                           capture_output=True, text=True).stdout.strip()
        state = q or "unknown"
    print(f"State: {state or 'unknown (no Slurm commands here)'}")
    out = folder / "logs" / f"ponderosa_{jid}.out"
    err = folder / "logs" / f"ponderosa_{jid}.err"
    if out.exists():
        keep = [l for l in out.read_text().splitlines() if l.startswith(("==", "[", "Written", "Analysis", "Classifier"))
                or "families" in l or "ERROR" in l]
        print("\nProgress:")
        print("\n".join("  " + l for l in keep[-a.lines:]) or "  (nothing yet)")
    if err.exists():
        bad = [l for l in err.read_text().splitlines() if "Error" in l or "ERROR" in l or "Traceback" in l]
        if bad:
            print("\nProblems in the error log:")
            print("\n".join("  " + l for l in bad[-a.lines:]))
    done = sorted(Path(st["out_prefix"]).parent.glob(f"{st['name']}*"))
    if done:
        print(f"\nOutputs ({len(done)} files) in {Path(st['out_prefix']).parent}")
    return 0
