#!/usr/bin/env python3
"""Setup script for context-aware recommendation system."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        command: Command to run.
        description: Description of what the command does.
        
    Returns:
        True if command succeeded, False otherwise.
    """
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main() -> None:
    """Main setup function."""
    print("Setting up Context-Aware Recommendation System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    commands = [
        ("pip install -r requirements.txt", "Installing dependencies"),
        ("pip install -e .", "Installing package in development mode"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
            break
    
    if not success:
        print("\nSetup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "results",
        "checkpoints",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run training: python scripts/train_evaluate.py")
    print("2. Launch demo: streamlit run scripts/demo.py")
    print("3. Run tests: pytest tests/")
    print("4. Explore notebook: jupyter notebook notebooks/example_usage.ipynb")


if __name__ == "__main__":
    main()
