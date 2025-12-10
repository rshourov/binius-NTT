#!/usr/bin/env python3
"""
Binius-NTT GPU Trial Run on Kaggle
This script will build and test the project on Kaggle GPU
"""

import subprocess
import sys
import os
import configparser
import shlex
from urllib.parse import urlparse

def run_cmd(cmd, description):
    """Run command and print output"""
    print(f"\n{'='*60}")
    print(f"📍 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    print(f"✅ Success: {description}")
    return True

def main():
    print("🚀 Binius-NTT GPU Build and Test")
    
    # Check GPU
    if not run_cmd("nvidia-smi", "Checking GPU availability"):
        sys.exit(1)
    
    run_cmd("nvcc --version", "CUDA version")
    
    # Install CMake 3.27
    print("\n📦 Installing CMake 3.27...")
    cmds = [
        "cd /tmp",
        "wget -q https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh",
        "chmod +x cmake-3.27.0-linux-x86_64.sh",
        "sudo ./cmake-3.27.0-linux-x86_64.sh --skip-license --prefix=/usr/local"
    ]
    if not run_cmd(" && ".join(cmds), "Installing CMake"):
        sys.exit(1)
    
    run_cmd("cmake --version", "CMake version check")
    
    # Navigate to working directory
    work_dir = "/kaggle/working"
    os.chdir(work_dir)
    
    # Initialize git submodules (if git repository exists)
    print("\n📚 Setting up Git submodules...")
    if os.path.exists(f"{work_dir}/.git"):
        run_cmd(f"cd {work_dir} && git config --global --add safe.directory {work_dir}", "Git config")
        run_cmd(f"cd {work_dir} && git submodule update --init --recursive", "Initializing submodules")
    elif os.path.exists(f"{work_dir}/.gitmodules"):
        # When .git is not present but .gitmodules is, manually clone submodules
        print("⚠️  No .git directory found, manually cloning submodules...")
        config = configparser.ConfigParser()
        config.read(f"{work_dir}/.gitmodules")
        work_dir_abs = os.path.abspath(work_dir)
        
        for section in config.sections():
            if section.startswith('submodule'):
                path = config[section].get('path', '').strip()
                url = config[section].get('url', '').strip()
                
                # Validate path to prevent directory traversal attacks
                if not path:
                    continue
                # Normalize and validate path stays within work_dir
                normalized_path = os.path.normpath(os.path.join(work_dir_abs, path))
                if not normalized_path.startswith(work_dir_abs + os.sep) and normalized_path != work_dir_abs:
                    print(f"⚠️  Skipping {path}: Path escapes working directory (security check failed)")
                    continue
                
                # Validate URL structure using proper URL parsing
                if not url:
                    print(f"⚠️  Skipping {path}: Empty URL")
                    continue
                try:
                    parsed_url = urlparse(url)
                    if parsed_url.scheme != 'https' or parsed_url.netloc != 'github.com':
                        print(f"⚠️  Skipping {path}: URL must be https://github.com/ (got {url})")
                        continue
                    # Additional checks for suspicious patterns
                    if '..' in url or '@' in url.split('github.com')[1] if 'github.com' in url else True:
                        print(f"⚠️  Skipping {path}: Suspicious URL pattern detected in {url}")
                        continue
                except Exception as e:
                    print(f"⚠️  Skipping {path}: Invalid URL format ({url})")
                    continue
                
                target_dir = os.path.join(work_dir, path)
                if not os.path.exists(os.path.join(target_dir, '.git')):
                    # Shell-escape URL to prevent command injection
                    safe_url = shlex.quote(url)
                    safe_target = shlex.quote(target_dir)
                    if not run_cmd(f"git clone {safe_url} {safe_target}", f"Cloning {path} from {url}"):
                        print(f"❌ Failed to clone submodule {path} from {url}")
                        continue
                    # Verify the clone was successful
                    if not os.path.exists(os.path.join(target_dir, '.git')):
                        print(f"❌ Clone verification failed for {path} - .git directory not found after clone")
                        continue
    else:
        print("⚠️  No git metadata found, skipping submodule initialization")
        print("    Make sure submodules are already present in the uploaded files")
    
    # Build project
    print("\n🔨 Building project...")
    if not run_cmd(
        f'cd {work_dir} && cmake -B./build -DCMAKE_CUDA_HOST_COMPILER="g++" -DCMAKE_CXX_COMPILER="g++"',
        "CMake configuration"
    ):
        sys.exit(1)
    
    if not run_cmd(f"cd {work_dir} && cmake --build ./build -j$(nproc)", "Building with CMake"):
        sys.exit(1)
    
    # Run tests
    print("\n🧪 Running Tests...")
    
    tests = [
        ("./build/ntt_tests '[ntt][additive][0]' -d yes", "NTT Tests (quick)"),
        ("./build/finite_field_tests -d yes", "Finite Field Tests"),
        ("./build/sumcheck_test -d yes", "Sumcheck Tests"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_cmd(f"cd {work_dir} && {cmd}", desc)
        results.append((desc, success))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {desc}")
    
    # Optional: Run benchmarks if all tests passed
    if all(success for _, success in results):
        print("\n🏃 Running benchmarks...")
        run_cmd(f"cd {work_dir} && ./build/sumcheck_bench", "Sumcheck Benchmark")
    
    print("\n✨ Trial run complete!")

if __name__ == "__main__":
    main()
