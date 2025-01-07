# File: src/downloader/cli_validator.py
import subprocess


def validate_crunch_cli() -> None:
    """
    Ensure that the Crunch CLI is installed and accessible.

    This function runs a test command to check if the Crunch CLI is properly installed
    and available in the system PATH. If the CLI is not installed or functioning
    correctly, an appropriate error is raised.

    Raises:
        RuntimeError: If the Crunch CLI is not installed or is not functioning correctly.
    """
    try:
        # Execute a test command to verify Crunch CLI availability
        subprocess.run(
            ["crunch", "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        print("[✔] Crunch CLI is installed and functioning.")
    except FileNotFoundError:
        # Raise an error if the Crunch CLI executable is not found
        raise RuntimeError("Crunch CLI is not installed or not in the system PATH.")
    except subprocess.CalledProcessError:
        # Raise an error if the Crunch CLI fails unexpectedly
        raise RuntimeError("Crunch CLI is installed but not functioning correctly.")
