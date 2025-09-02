#!/bin/bash
set -e
echo " Installing Vectoria CLI (vecto)..."

if ! command -v cargo &> /dev/null; then
    echo " Rust is not installed. Please install Rust first:"
    echo "   Visit https://rustup.rs/ and follow the installation instructions"
    exit 1
fi

if [ ! -f "Cargo.toml" ]; then
    echo " Please run this script from the Vectoria project root directory"
    exit 1
fi

echo " Building vecto..."
cargo build --release

echo "🔧 Installing vecto globally..."
cargo install --path .

if command -v vecto &> /dev/null; then
    echo " vecto installed successfully!"
    echo ""
    echo " You can now use vecto from anywhere:"
    echo "   vecto --help"
    echo ""
    echo " Quick start:"
    echo "   vecto remember 'Your first note here'"
    echo "   vecto list-notes"
    echo "   vecto search 'your search query'"
    echo ""
    echo " For more information, visit: https://github.com/yswa-var/Vectoria"
else
    echo " Installation failed. Please check the error messages above."
    exit 1
fi 