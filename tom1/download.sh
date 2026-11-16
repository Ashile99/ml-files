#!/usr/bin/env sh
# ==============================================================================
# Ultra-Fast Silent Downloader for ML Lab Files
# ==============================================================================

set -e

# Target directory (current working directory by default)
TARGET_DIR="${TARGET_DIR:-.}"
mkdir -p "${TARGET_DIR}"

BASE_URL="${BASE_URL:-https://raw.githubusercontent.com/Ashile99/ml-files/main/tom1}"
FILES="${*:-Bais.py BSFOLD.py CANCER.py KNN.py RBF.py svm_iris.py}"

# Check for curl or wget silently
if command -v curl >/dev/null 2>&1; then
    FETCH_CMD="curl"
elif command -v wget >/dev/null 2>&1; then
    FETCH_CMD="wget"
else
    exit 1
fi

# Download all files concurrently in parallel silently
PID_LIST=""
for file in ${FILES}; do
    if [ "${FETCH_CMD}" = "curl" ]; then
        curl -fsSL --connect-timeout 5 "${BASE_URL}/${file}" -o "${TARGET_DIR}/${file}" 2>/dev/null &
    else
        wget -q -O "${TARGET_DIR}/${file}" "${BASE_URL}/${file}" 2>/dev/null &
    fi
    PID_LIST="${PID_LIST} $!"
done

# Wait for all parallel jobs to complete
FAIL=0
for pid in ${PID_LIST}; do
    wait "${pid}" 2>/dev/null || FAIL=1
done

if [ ${FAIL} -ne 0 ]; then
    exit 1
fi

echo "done"
