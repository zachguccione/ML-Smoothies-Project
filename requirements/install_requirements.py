import subprocess
import sys
import os

def install_requirements():
    # Get the path to requirements.txt
    base_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(base_dir, "requirements.txt")

    # Check if requirements.txt exists
    if not os.path.exists(requirements_path):
        print(f"requirements.txt not found at {requirements_path}")
        sys.exit(1)

    # Install dependencies
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
