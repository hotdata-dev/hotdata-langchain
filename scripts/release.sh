#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die() { echo "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required"; }

# Reading pyproject.toml uses tomllib, which is Python 3.11+, while macOS still ships 3.9
# as python3. Prefer the system interpreter, else fall back to uv's. Most call sites are
# command substitutions, so RELEASE_PY only caches within one subshell; the probe is a
# process spawn, not a resolution the caller has to thread through.
#
# The fallback names a version floor. Left unconstrained, `uv run` accepts whatever
# interpreter it discovers first — measured on a machine with no managed Python and no
# .venv, that is the same /usr/bin/python3 3.9.6 the probe just rejected, so the fallback
# would reproduce the error it exists to avoid. A range rather than an exact version, so an
# existing 3.12 or 3.13 is used rather than downloading a 3.11 alongside it; uv fetches a
# managed interpreter only when nothing on the machine satisfies the floor.
#
# The uv path is probed by importing tomllib through it rather than by assuming it works,
# so a machine that cannot supply 3.11+ fails here with an actionable message instead of a
# traceback from the first call site.
PY_FLOOR='>=3.11'

# Reported in the failure message. An `a && b || c` chain would run c when either a or b
# failed, printing both the shell's own "command not found" and the fallback text.
found_python() {
  if command -v python3 >/dev/null 2>&1; then
    python3 --version 2>&1
  else
    echo 'no python3'
  fi
}

py() {
  if [[ -z "${RELEASE_PY:-}" ]]; then
    if python3 -c 'import tomllib' >/dev/null 2>&1; then
      RELEASE_PY=system
    elif command -v uv >/dev/null 2>&1 &&
      uv run --quiet --no-project --python "$PY_FLOOR" python -c 'import tomllib' >/dev/null 2>&1; then
      RELEASE_PY=uv
    else
      die "need a python with tomllib (3.11+), or uv to supply one; found $(found_python)"
    fi
  fi
  if [[ "$RELEASE_PY" == system ]]; then
    python3 "$@"
  else
    uv run --quiet --no-project --python "$PY_FLOOR" python "$@"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/release.sh prepare [patch|minor|major|X.Y.Z]
  ./scripts/release.sh publish

Workflow:
  1. Move notes from [Unreleased] in CHANGELOG.md (or add them there).
  2. ./scripts/release.sh prepare patch
  3. Merge the release PR.
  4. ./scripts/release.sh publish

Tag push triggers PyPI publish and GitHub Release creation in CI.
EOF
}

get_version() {
  py - <<'PY'
import tomllib
from pathlib import Path
print(tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"])
PY
}

get_pkg_name() {
  py - <<'PY'
import tomllib
from pathlib import Path
print(tomllib.loads(Path("pyproject.toml").read_text())["project"]["name"])
PY
}

set_version() {
  local ver="$1"
  py - "$ver" <<'PY'
import re, sys
from pathlib import Path
ver = sys.argv[1]
path = Path("pyproject.toml")
text = path.read_text()
new, n = re.subn(r'(?m)^version = "[^"]+"', f'version = "{ver}"', text, count=1)
if n != 1:
    raise SystemExit("could not update version in pyproject.toml")
path.write_text(new)
PY
}

bump_version() {
  local kind="$1" current="$2"
  py - "$kind" "$current" <<'PY'
import re, sys
kind, current = sys.argv[1], sys.argv[2]
match = re.match(r"^(\d+)\.(\d+)\.(\d+)(.*)$", current)
if not match:
    raise SystemExit(f"unsupported version: {current}")
major, minor, patch, suffix = int(match[1]), int(match[2]), int(match[3]), match[4]
if suffix:
    raise SystemExit("pre-release versions must be set explicitly as X.Y.Z")
if kind == "patch":
    patch += 1
elif kind == "minor":
    minor += 1
    patch = 0
elif kind == "major":
    major += 1
    minor = 0
    patch = 0
else:
    raise SystemExit(f"unknown bump kind: {kind}")
print(f"{major}.{minor}.{patch}")
PY
}

default_branch() {
  local remote="${1:-origin}"
  local branch
  branch="$(git symbolic-ref --quiet "refs/remotes/${remote}/HEAD" 2>/dev/null | sed "s|refs/remotes/${remote}/||")"
  if [[ -n "$branch" ]]; then
    echo "$branch"
    return
  fi
  branch="$(git branch -r --list "${remote}/main" "${remote}/master" | sed "s|^[[:space:]]*${remote}/||" | head -1)"
  if [[ -n "$branch" ]]; then
    echo "$branch"
    return
  fi
  echo main
}

ensure_clean() {
  [[ -z "$(git status --porcelain)" ]] || die "working tree is not clean"
}

update_changelog() {
  local ver="$1"
  local date
  date="$(date +%Y-%m-%d)"
  py scripts/update_changelog.py "$ver" "$date"
}

cmd_prepare() {
  local bump="${1:-}"
  [[ -n "$bump" ]] || { usage; die "missing bump kind or explicit version"; }
  need gh
  need uv
  ensure_clean

  local current new base branch pkg
  current="$(get_version)"
  if [[ "$bump" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    new="$bump"
  else
    new="$(bump_version "$bump" "$current")"
  fi
  [[ "$new" != "$current" ]] || die "new version ($new) equals current ($current)"

  base="$(default_branch)"
  git fetch origin "$base"
  git checkout "$base"
  git pull --ff-only origin "$base"
  ensure_clean

  set_version "$new"
  update_changelog "$new"
  # uv.lock records this project's own version, and CI installs with --locked.
  uv lock

  branch="release/v${new}"
  git checkout -b "$branch"
  git add pyproject.toml CHANGELOG.md uv.lock
  git commit -m "chore: release v${new}"

  pkg="$(get_pkg_name)"
  git push -u origin "$branch"
  gh pr create --base "$base" --head "$branch" \
    --title "chore: release ${pkg} v${new}" \
    --body "## Summary
Release **${pkg} v${new}**.

## Checklist
- [x] Version bumped in \`pyproject.toml\`
- [x] \`CHANGELOG.md\` updated
- [ ] CI green

After merge, run \`./scripts/release.sh publish\` from a clean \`${base}\` checkout."

  echo "Prepared ${pkg} v${new}. Merge the PR, then run: ./scripts/release.sh publish"
}

cmd_publish() {
  need gh
  ensure_clean

  local base ver tag
  base="$(default_branch)"
  git fetch origin "$base"
  git checkout "$base"
  git pull --ff-only origin "$base"
  ensure_clean

  ver="$(get_version)"
  tag="v${ver}"

  git rev-parse "$tag" >/dev/null 2>&1 && die "tag $tag already exists"
  [[ -f CHANGELOG.md ]] || die "CHANGELOG.md is required"
  py - "$ver" <<'PY'
import re, sys
from pathlib import Path
ver = sys.argv[1]
text = Path("CHANGELOG.md").read_text()
if not re.search(rf"^## \[{re.escape(ver)}\]", text, re.M):
    raise SystemExit(f"CHANGELOG.md missing section for {ver}")
PY

  git tag "$tag"
  git push origin "$tag"

  pkg="$(get_pkg_name)"
  echo "Pushed ${tag} for ${pkg}."
  echo "CI will publish to PyPI and create the GitHub Release."
}

case "${1:-}" in
  prepare) shift; cmd_prepare "${1:-}" ;;
  publish) cmd_publish ;;
  -h|--help|help|"") usage ;;
  *) usage; die "unknown command: $1" ;;
esac
