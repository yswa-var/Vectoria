#!/bin/bash

set -e

echo " Uninstalling Vectoria CLI (vecto)..."

if ! command -v vecto &> /dev/null; then
    echo " vecto is not installed on your system."
    exit 1
fi

VECTO_PATH=$(which vecto)
echo " Found vecto at: $VECTO_PATH"

echo " Removing vecto binary..."
sudo rm -f "$VECTO_PATH"

if ! command -v vecto &> /dev/null; then
    echo " vecto uninstalled successfully!"
    echo ""
    echo " Note: Your data files (embeddings.db) are still in your project directories."
    echo "   If you want to remove them completely, delete the embeddings.db files manually."
else
    echo " Uninstallation failed. Please check the error messages above."
    exit 1
fi 