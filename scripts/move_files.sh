#!/usr/bin/env bash
# Move all entries from one directory into another.
#
# Usage:
#   ./scripts/move_files.sh SOURCE_DIR DEST_DIR
#   ./scripts/move_files.sh SOURCE_DIR DEST_DIR --files-only   # skip subdirectories
#
# Examples:
#   ./scripts/move_files.sh data/maestro-v3.0.0/2004 data/maestro-v3.0.0/train
#   ./scripts/move_files.sh ./inbox ./archive --files-only
set -euo pipefail

FILES_ONLY=false

usage() {
  cat <<'EOF'
Usage: move_files.sh SOURCE_DIR DEST_DIR [--files-only]

Moves each item directly inside SOURCE_DIR into DEST_DIR (creates DEST_DIR if needed).
Does not move SOURCE_DIR itself.

  --files-only   Move only regular files (not subdirectories)
  -h, --help     Show this help
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --files-only) FILES_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *)
      if [[ -z "${SRC:-}" ]]; then SRC=$1
      elif [[ -z "${DST:-}" ]]; then DST=$1
      else die "too many arguments"
      fi
      shift
      ;;
  esac
done

[[ -n "${SRC:-}" && -n "${DST:-}" ]] || { usage >&2; exit 1; }

SRC=$(cd "$SRC" 2>/dev/null && pwd) || die "source not found: $SRC"
mkdir -p "$DST"
DST=$(cd "$DST" && pwd)

[[ "$SRC" != "$DST" ]] || die "source and destination must differ"
[[ "$DST" != "$SRC/"* ]] || die "destination cannot be inside source"

shopt -s dotglob nullglob
entries=("$SRC"/*)
shopt -u dotglob nullglob

if [[ ${#entries[@]} -eq 0 || ! -e "${entries[0]}" ]]; then
  echo "Nothing to move in $SRC"
  exit 0
fi

count=0
for path in "${entries[@]}"; do
  name=$(basename "$path")
  if $FILES_ONLY && [[ ! -f "$path" ]]; then
    continue
  fi
  if [[ -e "$DST/$name" ]]; then
    die "already exists: $DST/$name (remove or rename first)"
  fi
  mv -v "$path" "$DST/"
  count=$((count + 1))
done

echo "Moved $count item(s) from $SRC -> $DST"
