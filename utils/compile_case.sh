#!/bin/bash
## Clean and compile a channel or wing case using its own compile_script.
##
## Usage:
##   ./utils/compile_case.sh --path envs/cases/mini_channel [--solver drl|raw]
##
## --solver drl  (default) build against KTH_DRL_Framework -> ./nek5000
##               Coupled mode: rank 0 belongs to Python.
## --solver raw            build against KTH_Framework     -> ./nek5000_solo
##               Embedded mode: stock core, UPARAM(10)=1.
##
## Both binaries live in the same case folder and survive each other's
## builds. Switching solvers is a from-scratch rebuild, which costs
## nothing extra here because this script always cleans first anyway.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case_path="${SCRIPT_DIR}/../envs/cases/mini_channel"
solver=""

usage() {
    cat <<'EOF'
Usage: ./utils/compile_case.sh [--path CASE_DIR] [--solver drl|raw]

  --path CASE_DIR  case directory (default: envs/cases/mini_channel)
  --m CASE_DIR     legacy alias for --path; also takes a directory path
  --solver drl    coupled Python/Nek build -> nek5000 (default)
  --solver raw    Nek-only embedded-policy build -> nek5000_solo
  -h, --help      show this help

Run from the repository root with your MPI compiler environment loaded.
The case's compile_script selects the solver version and case name.
Each invocation cleans the selected build, preserving third-party libraries.
EOF
}

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --path|--m|--solver)
            [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || fail "Missing value for $1"
            if [[ "$1" == "--solver" ]]; then
                solver="$2"
            else
                case_path="$2"
            fi
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown parameter: $1 (see --help)" ;;
    esac
done

case "${solver:-drl}" in
    drl) binary="nek5000"; other_binary="nek5000_solo" ;;
    raw) binary="nek5000_solo"; other_binary="nek5000" ;;
    *) fail "Unknown solver '${solver}'; expected drl or raw" ;;
esac

target_path=$(realpath -e -- "${case_path}") || fail "Case directory not found: ${case_path}"
[[ -d "${target_path}" ]] || fail "Not a case directory: ${target_path}"
cd "${target_path}" || exit 1
[[ -f compile_script && -r compile_script ]] || fail "No readable compile_script in ${target_path}"
echo "[INFO] Compile ${target_path}"

# The v17 makenek.inc unconditionally removes 'nek5000' while configuring,
# even for a raw build. Preserve the other executable across success/failure
# without patching the extracted solver trees or their distribution archives.
backup_dir=""
restore_other_binary() {
    local status=$?
    if [[ -n "${backup_dir}" ]]; then
        mv -f -- "${backup_dir}/${other_binary}" "${other_binary}" || status=1
        rmdir -- "${backup_dir}" || status=1
    fi
    return "${status}"
}
if [[ -e "${other_binary}" || -L "${other_binary}" ]]; then
    backup_dir=$(mktemp -d .compile_case.XXXXXX) || fail "Cannot create binary backup"
    if ! cp -pP -- "${other_binary}" "${backup_dir}/${other_binary}"; then
        rmdir -- "${backup_dir}"
        fail "Cannot preserve ${other_binary}"
    fi
    trap 'restore_other_binary || exit $?' EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
fi

SOLVER_ARG=()
if [[ -n "${solver}" ]]; then
    SOLVER_ARG=(--solver "${solver}")
    echo "[INFO] Solver backend: ${solver}"
fi

#[MOD] makenek asks "cleanup all 3rd party dependencies too? [N]" on the
#[MOD] clean path, and asks again if it detects a configuration change.
#[MOD] Either would hang a batch job, so supply answers on standard input instead
#[MOD] of relying on somebody being at the keyboard. 'N' keeps the
#[MOD] prebuilt gslib/blasLapack, which live under each solver root and
#[MOD] are therefore never shared between the two builds.
# Invoke Bash explicitly: some case scripts are tracked without executable
# permission. A here-string avoids a pipe writer failing if no prompt is read.
bash ./compile_script "${SOLVER_ARG[@]}" --clean <<< $'N\nN' \
    || fail "Clean failed: ${target_path}"
# Some legacy makenek scripts mask make failures. Check that cleaning really
# removed the selected executable so a stale binary cannot pass verification.
[[ ! -e "${binary}" ]] || fail "Clean left ${binary} in place; inspect the compiler output"
bash ./compile_script "${SOLVER_ARG[@]}" --all <<< $'N\nN' \
    || fail "Build failed: ${target_path}"
[[ -x "${binary}" ]] || fail "Build did not produce ${binary}; inspect the compiler output"

echo "[INFO] Finished compile: ${target_path}/${binary}"
