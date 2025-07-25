.PHONY: build install uninstall test clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  build     - Build the project in release mode"
	@echo "  install   - Install vecto globally"
	@echo "  uninstall - Remove vecto from system"
	@echo "  test      - Run tests"
	@echo "  clean     - Clean build artifacts"
	@echo "  release   - Build for release"
	@echo "  help      - Show this help message"

# Build the project
build:
	cargo build --release

# Install globally
install: build
	cargo install --path .

# Uninstall
uninstall:
	cargo uninstall vecto

# Run tests
test:
	cargo test

# Clean build artifacts
clean:
	cargo clean

# Build for release
release: clean build
	@echo "Release build complete!"
	@echo "Binary location: target/release/vecto"

# Quick development build
dev:
	cargo build

# Check code quality
check:
	cargo check
	cargo clippy

# Format code
fmt:
	cargo fmt

# Run with example
demo:
	@echo "Running demo..."
	vecto remember "This is a demo note for testing"
	vecto list-notes 