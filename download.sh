#!/usr/bin/env sh
# ==============================================================================
# Fast Downloader for ML Lab Files (exp9.ipynb, exp10.ipynb)
# Compatible with Ubuntu, Debian, Fedora, Arch, and POSIX shells (sh, bash, zsh)
# ==============================================================================

set -e

# Target directory
TARGET_DIR="${HOME}/Downloads"
mkdir -p "${TARGET_DIR}"

echo ""
echo "=========================================="
echo " Starting Download to: ${TARGET_DIR}"
echo "=========================================="

# ------------------------------------------------------------------------------
# Configuration: Replace USERNAME and REPO with your GitHub username & repo
# (Or replace with your direct file URLs)
# ------------------------------------------------------------------------------
BASE_URL="https://raw.githubusercontent.com/Ashile99/ml-files/main"

FILE1_NAME="exp9.ipynb"
FILE1_URL="${BASE_URL}/${FILE1_NAME}"

FILE2_NAME="exp10.ipynb"
FILE2_URL="${BASE_URL}/${FILE2_NAME}"

# Helper function to download files using curl or wget
fetch_file() {
    url="$1"
    destination="$2"
    filename="$3"

    echo ""
    echo "==> Downloading: ${filename}..."

    if command -v curl >/dev/null 2>&1; then
        if ! curl -fSL --progress-bar "${url}" -o "${destination}"; then
            echo "Error: Failed to download ${filename} via curl." >&2
            exit 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if ! wget -q --show-progress -O "${destination}" "${url}"; then
            echo "Error: Failed to download ${filename} via wget." >&2
            exit 1
        fi
    else
        echo "Error: Neither 'curl' nor 'wget' is installed on this system." >&2
        echo "Please install one (e.g. sudo apt install curl / sudo pacman -S curl / sudo dnf install curl) and try again." >&2
        exit 1
    fi
}

# Download each file
fetch_file "${FILE1_URL}" "${TARGET_DIR}/${FILE1_NAME}" "${FILE1_NAME}"
fetch_file "${FILE2_URL}" "${TARGET_DIR}/${FILE2_NAME}" "${FILE2_NAME}"

# Verify files exist and print final summary
echo ""
echo "=========================================="
echo " [✓] Download Completed Successfully!"
echo " Files saved to: ${TARGET_DIR}"
echo "   - ${FILE1_NAME} ($(ls -lh "${TARGET_DIR}/${FILE1_NAME}" 2>/dev/null | awk '{print $5}' || echo 'saved'))"
echo "   - ${FILE2_NAME} ($(ls -lh "${TARGET_DIR}/${FILE2_NAME}" 2>/dev/null | awk '{print $5}' || echo 'saved'))"
echo "=========================================="
echo ""
