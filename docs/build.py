# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
#!/usr/bin/env python3
"""
Build script for Diffuse documentation using Sphinx Book Theme.

This script provides the same functionality as Flax's documentation build system.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def install_deps():
    """Install documentation dependencies."""
    print("Installing documentation dependencies...")
    return run_command([sys.executable, "-m", "pip", "install", "-r", "requirements-docs.txt"])


def clean():
    """Clean build directory."""
    print("Cleaning build directory...")
    return run_command(["make", "clean"])


def build_html():
    """Build HTML documentation."""
    print("Building HTML documentation...")
    return run_command(["make", "html"])


def serve():
    """Serve documentation locally."""
    print("Starting local server...")
    html_dir = Path("_build/html")
    if not html_dir.exists():
        print("Documentation not built. Building now...")
        if not build_html():
            return False

    import webbrowser
    import http.server
    import socketserver

    os.chdir(html_dir)
    port = 8000

    try:
        with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
            url = f"http://localhost:{port}"
            print(f"Serving at {url}")
            webbrowser.open(url)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        return True


def main():
    """Main build script."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Diffuse documentation")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--clean", action="store_true", help="Clean build directory")
    parser.add_argument("--serve", action="store_true", help="Serve documentation locally")

    args = parser.parse_args()

    # Change to docs directory
    docs_dir = Path(__file__).parent
    os.chdir(docs_dir)

    success = True

    if args.install_deps:
        success &= install_deps()

    if args.clean:
        success &= clean()

    # Always build HTML
    success &= build_html()

    if args.serve:
        serve()

    if not success:
        print("Build failed!")
        sys.exit(1)
    else:
        print("✅ Documentation built successfully!")
        print("📁 Location: _build/html/index.html")


if __name__ == "__main__":
    main()
