#!/usr/bin/env sh
# ==============================================================================
# Ultra-Fast Downloader for ML Lab Files
# ==============================================================================

set -e

# Target directory (current working directory by default)
TARGET_DIR="${TARGET_DIR:-.}"
mkdir -p "${TARGET_DIR}"

RESOLVED_DIR="$(cd "${TARGET_DIR}" 2>/dev/null && pwd || echo "${TARGET_DIR}")"

echo ""
echo "=========================================="
echo " [⚡] Starting Fast Parallel Download to: ${RESOLVED_DIR}"
echo "=========================================="

BASE_URL="https://raw.githubusercontent.com/Ashile99/ml-files/main"
FILES="${*:-hyperplane.ipynb kfold.ipynb knear.ipynb mlemap.ipynb multi.ipynb rbf.ipynb}"

# Check for curl or wget
if command -v curl >/dev/null 2>&1; then
    FETCH_CMD="curl"
elif command -v wget >/dev/null 2>&1; then
    FETCH_CMD="wget"
else
    echo "Error: Neither 'curl' nor 'wget' is installed." >&2
    exit 1
fi

# Download all files concurrently in parallel
PID_LIST=""
for file in ${FILES}; do
    if [ "${FETCH_CMD}" = "curl" ]; then
        curl -fsSL --connect-timeout 5 "${BASE_URL}/${file}" -o "${TARGET_DIR}/${file}" &
    else
        wget -q -O "${TARGET_DIR}/${file}" "${BASE_URL}/${file}" &
    fi
    PID_LIST="${PID_LIST} $!"
done

# Wait for all parallel jobs to complete
FAIL=0
for pid in ${PID_LIST}; do
    wait "${pid}" || FAIL=1
done

if [ ${FAIL} -ne 0 ]; then
    echo "Error: Failed to download one or more files." >&2
    exit 1
fi

echo ""
echo "=========================================="
echo " [✓] Download Completed Successfully!"
echo " Files saved to: ${RESOLVED_DIR}"
for file in ${FILES}; do
    SIZE="$(ls -lh "${TARGET_DIR}/${file}" 2>/dev/null | awk '{print $5}' || echo 'saved')"
    echo "   - ${file} (${SIZE})"
done
echo "=========================================="
echo ""
