import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import sys

# Add the project directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config.config_loader import ConfigLoader
from src.downloader.cli_validator import validate_crunch_cli

console = Console()


class CrunchDownloader:
    """
    A class to handle downloading data for the Autoimmune Disease Machine Learning Challenge.
    """

    def __init__(self, config_path: str):
        """
        Initialize the downloader with configuration and CLI validation.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_loader = ConfigLoader(config_path)
        self.token_file = self.config_loader.get_global_setting("token_file")
        self.micromamba_path = "/home/secondbook5/micromamba/bin/micromamba"
        validate_crunch_cli()  # Ensure the Crunch CLI is installed

    def write_tokens_to_file(self, tokens):
        """
        Write the tokens to the token file.

        Args:
            tokens (list): List of tokens for each Crunch.
        """
        if not self.token_file:
            raise ValueError("[ERROR] Token file path is missing in the configuration.")
        with open(self.token_file, "w") as file:
            for token in tokens:
                file.write(token + "\n")
        console.log(f"[INFO] Tokens written to {self.token_file}.")

    def run_command_in_tmux(self, command: str, pane_name: str):
        """
        Run a command in a new `tmux` pane using `micromamba run` for the `autoimmune_ml` environment.

        Args:
            command (str): The command to execute.
            pane_name (str): Name of the `tmux` pane.
        """
        try:
            # Create a new tmux session if it doesn't already exist
            subprocess.run(f"tmux new-session -d -s {pane_name}", shell=True, check=True)

            # Use micromamba run to execute the command
            tmux_command = f"tmux send-keys -t {pane_name} '{self.micromamba_path} run -n autoimmune_ml {command}' Enter"
            subprocess.run(tmux_command, shell=True, check=True)

            console.log(f"[INFO] Command '{command}' running in tmux session '{pane_name}'.")
        except subprocess.CalledProcessError as e:
            console.log(f"[ERROR] Failed to run command in tmux: {e}")
        except Exception as e:
            console.log(f"[ERROR] Unexpected error while running command in tmux: {e}")

    def process_crunches_parallel(self, commands, max_workers=3):
        """
        Process all Crunches in parallel using `tmux`.

        Args:
            commands (list): List of Crunch commands to execute.
            max_workers (int): Maximum number of workers for parallel execution.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            tasks = {cmd: progress.add_task(f"[cyan]{cmd.split()[2]}", total=100) for cmd in commands}

            def run_crunch(cmd):
                pane_name = cmd.split()[2]  # Use the competition type (e.g., broad-1) as the pane name
                self.run_command_in_tmux(cmd, pane_name)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_crunch, cmd): cmd for cmd in commands}
                for future in futures:
                    try:
                        future.result()  # Wait for all tmux commands to initialize
                        console.log(f"[INFO] Command submitted to tmux: {futures[future]}")
                    except Exception as e:
                        console.log(f"[ERROR] Failed to process command: {e}")

    def download_crunches(self):
        """
        Main method to process and download all Crunches.
        """
        crunches = self.config_loader.config.get("crunches", {})
        if not crunches:
            console.log("[ERROR] No Crunch configurations found.")
            return

        # Prompt for tokens
        tokens = [
            input(f"Enter token for {crunch_name} ({config['name']}): ").strip()
            for crunch_name, config in crunches.items()
        ]
        self.write_tokens_to_file(tokens)

        # Construct Crunch commands
        commands = []
        for idx, (crunch_name, config) in enumerate(crunches.items()):
            competition_name = config["crunch_type"]
            project_name = config["name"]
            dataset_size = config["dataset_size"]
            output_dir = config["paths"]["project_dir"]
            token = tokens[idx]
            commands.append(
                f"crunch setup {competition_name} {project_name} {output_dir} --token {token} --size {dataset_size} --force"
            )

        self.process_crunches_parallel(commands)

    def update_manifest(self):
        """
        Update the manifest with the current state of all Crunch directories.
        """
        console.log("[INFO] Updating manifest (not implemented in this version).")


def main():
    """
    Main entry point for the CrunchDownloader script.
    """
    config_path = "config.yaml"
    downloader = CrunchDownloader(config_path)

    action = input("Enter action: 'download' or 'update_manifest': ").strip()

    if action == "download":
        downloader.download_crunches()
    elif action == "update_manifest":
        downloader.update_manifest()
    else:
        console.log(f"[ERROR] Invalid action: {action}")


if __name__ == "__main__":
    main()
