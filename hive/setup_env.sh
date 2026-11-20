#!/bin/bash
# One-time setup of PONDEROSA-IMPROVED on UC Davis HIVE (run on a login or interactive node).
#
#   bash hive/setup_env.sh /path/to/workdir
#
# Creates a private virtual environment (avoids shared-environment dependency drift),
# installs PONDEROSA-IMPROVED into it and, optionally, builds Ped-sim for
# simulation-based training. Module names are the ones used for the SacMEGA run;
# check `module avail` if they have changed.
set -euo pipefail

WORKDIR=${1:?usage: setup_env.sh WORKDIR}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$WORKDIR"
export PYTHONPYCACHEPREFIX=/tmp/${USER:-$(id -un)}_pycache   # writable bytecode cache

module load python 2>/dev/null || true
python3 -m venv "$WORKDIR/venv"
source "$WORKDIR/venv/bin/activate"
pip install --upgrade pip
pip install -e "$REPO"
python -c "import ponderosa; print('PONDEROSA', ponderosa.__version__, 'installed')"

# Ped-sim (only needed for `ponderosa simulate`): needs Boost headers.
if [[ "${BUILD_PEDSIM:-1}" == "1" ]]; then
    module load gcc/13.2.0 boost/1.84.0 2>/dev/null || echo "WARNING: could not load gcc/boost modules"
    if [[ ! -x "$WORKDIR/ped-sim/ped-sim" ]]; then
        git clone https://github.com/williamslab/ped-sim "$WORKDIR/ped-sim"
        make -C "$WORKDIR/ped-sim"
    fi
    # Sex-specific map expected by ponderosa simulate (pedsim_path/refined_mf.simmap)
    if [[ ! -f "$WORKDIR/ped-sim/refined_mf.simmap" ]]; then
        echo "NOTE: download the GRCh37 sex-specific map into $WORKDIR/ped-sim/refined_mf.simmap"
        echo "      (see https://github.com/williamslab/ped-sim#map-file); the shell cannot always reach it."
    fi
fi
echo "Environment ready: source $WORKDIR/venv/bin/activate"
