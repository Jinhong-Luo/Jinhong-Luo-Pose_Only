#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 1

failed=0

ok() {
  printf '[OK] %s\n' "$1"
}

fail() {
  printf '[FAIL] %s\n' "$1"
  failed=1
}

info() {
  printf '[INFO] %s\n' "$1"
}

if ! command -v python >/dev/null 2>&1; then
  fail "python command is not available"
  exit 1
fi

py_version="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null)"
info "Python version: $py_version"
case "$py_version" in
  3.10.*) ok "Python 3.10 detected" ;;
  *) fail "Expected Python 3.10.x, got $py_version" ;;
esac

for mod in numpy scipy cv2 torch kornia yaml tqdm; do
  if python -c "import $mod" >/dev/null 2>&1; then
    ok "Import succeeded: $mod"
  else
    fail "Import failed: $mod"
  fi
done

if python -c "import pathlib, sys; root = pathlib.Path('.').resolve(); sys.path.insert(0, str(root / 'tools')); import _bootstrap; import lightglue" >/dev/null 2>&1; then
  ok "LightGlue import via _bootstrap succeeded"
else
  fail "LightGlue import via _bootstrap failed"
fi

scripts=(
  "tools/strecha_prepare.py"
  "tools/build_lightglue_tracks_npz.py"
  "tools/generate_rraa_input.py"
  "tools/RRAA_fast.py"
  "tools/Pose_Only_patched_v3_fixed.py"
  "tools/eval_rraa_rotation.py"
  "tools/eval_poseonly_strecha_mm.py"
)

for script in "${scripts[@]}"; do
  if python "$script" --help >/dev/null 2>&1; then
    ok "$script --help"
  else
    fail "$script --help failed"
  fi
done

if [ "$failed" -ne 0 ]; then
  fail "Local setup check finished with failures"
  exit 1
fi

ok "Local setup check passed"
