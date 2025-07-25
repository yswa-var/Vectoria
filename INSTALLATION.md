# Vectoria CLI Installation Guide

This guide provides multiple ways to install the `vecto` CLI tool on your system.

## Prerequisites

- **Rust** (for building from source): Install from [rustup.rs](https://rustup.rs/)
- **BERT Model Files**: Download using `./download_model.sh`

## Installation Methods

### Method 1: Quick Installation Script (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/vecto.git
cd vecto

# Run the installation script
./install.sh
```

### Method 2: Using Cargo

```bash
# Install from source
git clone https://github.com/yourusername/vecto.git
cd vecto
cargo install --path .

# Or install from crates.io (when published)
cargo install vecto
```

### Method 3: Using Homebrew (macOS)

```bash
# Install from GitHub
brew install --HEAD https://raw.githubusercontent.com/yourusername/vecto/main/Formula/vecto.rb

# Or add the tap (when published)
brew tap yourusername/vecto
brew install vecto
```

### Method 4: Download Pre-built Binaries

1. Visit the [Releases page](https://github.com/yourusername/vecto/releases)
2. Download the appropriate binary for your system:
   - `vecto-linux-x86_64` for Linux
   - `vecto-macos-x86_64` for macOS
   - `vecto-windows-x86_64.exe` for Windows
3. Make it executable and move to PATH:

```bash
chmod +x vecto-linux-x86_64
sudo mv vecto-linux-x86_64 /usr/local/bin/vecto
```

### Method 5: Using Make

```bash
# Clone and install
git clone https://github.com/yourusername/vecto.git
cd vecto
make install
```

## Verification

After installation, verify that `vecto` is working:

```bash
# Check if installed
vecto --help

# Test with a simple command
vecto remember "Hello, Vectoria!"
vecto list-notes
```

## Uninstallation

### Using the uninstall script:

```bash
./uninstall.sh
```

### Using Cargo:

```bash
cargo uninstall vecto
```

### Using Make:

```bash
make uninstall
```

## Configuration

Copy the example configuration file:

```bash
cp vecto.toml.example ~/.config/vecto/config.toml
```

## Troubleshooting

### Common Issues

1. **"command not found: vecto"**

   - Ensure the binary is in your PATH
   - Try running `cargo install --path .` again

2. **"Could not load tokenizer"**

   - Run `./download_model.sh` to download BERT model files
   - Ensure the `models/` directory exists

3. **Permission denied**

   - Use `sudo` for system-wide installation
   - Or install to user directory with `cargo install --path .`

4. **Database errors**
   - Delete `embeddings.db` to reset the database
   - Ensure you have write permissions in the current directory

### Getting Help

- Run `vecto --help` for command help
- Check the [README.md](README.md) for detailed usage
- Report issues on [GitHub](https://github.com/yourusername/vecto/issues)

## Development Installation

For developers who want to work on the code:

```bash
# Clone the repository
git clone https://github.com/yourusername/vecto.git
cd vecto

# Build in development mode
cargo build

# Run tests
cargo test

# Install in development mode
cargo install --path .
```

## System Requirements

- **OS**: Linux, macOS, or Windows
- **Architecture**: x86_64
- **Memory**: 512MB RAM minimum
- **Storage**: 100MB free space
- **Network**: Internet connection for model download and RAG features

## Next Steps

After installation, check out the [README.md](README.md) for usage examples and advanced features.
