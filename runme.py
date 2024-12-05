import subprocess
import os
import sys

# Path to the requirements installation script
install_script_path = os.path.join(os.path.dirname(__file__), "requirements/install_requirements.py")

# Run the installation script
try:
    print("Installing requirements...")
    subprocess.check_call([sys.executable, install_script_path])
    print("Requirements installed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error installing requirements: {e}")
    exit(1)

# Main script logic
print("Running the main script...")

# Run the main script
try:
    result = subprocess.run(
        ['python', 'src/Application.py'],
        capture_output=True,
        text=True,
        check=True
    )
    # Display the script's output
    output = result.stdout.strip()
    print(output)
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the main script:\n{e.stderr}")
    exit(1)
