#!/usr/bin/env bash
# docker-entrypoint.sh — fix cache directory ownership before starting the app.
#
# Named Docker volumes retain root ownership from the build stage. When the
# container runs as the non-root voicebox user, writes to those directories
# fail with PermissionError. This script chowns the cache dirs to the
# voicebox user (uid/gid are looked up so it works regardless of the exact IDs
# assigned by the base image).
set -euo pipefail

VOICEBOX_USER="voicebox"
CACHE_DIRS=(
    /home/voicebox/.cache/numba
    /home/voicebox/.cache/joblib
    /home/voicebox/.cache/huggingface
    /app/data
)

for d in "${CACHE_DIRS[@]}"; do
    if [ -d "$d" ]; then
        chown -R "$VOICEBOX_USER":"$VOICEBOX_USER" "$d" 2>/dev/null || true
    fi
done

exec "$@"