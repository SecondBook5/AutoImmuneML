# File: src/downloader/crunch_data_downloader.py

import os
import subprocess
import time
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.config.config_loader import ConfigLoader
from src.downloader.token_handler import TokenHandler
from src.downloader.cli_validator import validate_crunch_cli
from src.downloader.manifest_manager import ManifestManager


class CrunchDownloader:
    """
    A class to handle downloading data for the Autoimmune Disease Machine Learning Challenge.
    """

    def __init__(self, config_path: str, manifest_path: str):
        """
        Initialize the downloader with configuration, manifest, and token handling.

        Args:
            config_path (str): Path to the YAML configuration file.
            manifest_path (str): Path to the manifest JSON file.
        """
        self.config_loader = ConfigLoader(config_path)
        self.manifest_manager = ManifestManager(manifest_path)

        # Retrieve the token file path from the global section of the configuration
        token_file = self.config_loader.get_global_setting("token_file")
        if not token_file:
            raise ValueError("[ERROR] Token file path is missing in the configuration.")
        self.token_handler = TokenHandler(token_file)

        # Validate CLI installation
        validate_crunch_cli()

    def download_data(self, crunch_name: str, competition_name: str, project_name: str, dataset_size: str, token: str, output_dir: str, dry_run: bool = False) -> bool:
        """
        Download data using the Crunch CLI.

        Args:
            crunch_name (str): Name of the Crunch.
            competition_name (str): Competition type (e.g., "broad-1").
            project_name (str): Project name in the Crunch system.
            dataset_size (str): Dataset size (e.g., "default" or "large").
            token (str): Authentication token.
            output_dir (str): Target output directory.
            dry_run (bool): If True, simulate the download.

        Returns:
            bool: True if the download succeeds, False otherwise.
        """
        command = [
            "crunch", "setup",
            competition_name, project_name,
            output_dir,
            "--token", token,
            "--size", dataset_size,
        ]

        if dry_run:
            print(f"[DRY-RUN] Would execute: {' '.join(command)}")
            return True

        try:
            with tqdm(total=100, desc=f"Downloading {crunch_name}", unit="%", dynamic_ncols=True) as pbar:
                for i in range(10):  # Simulated progress
                    time.sleep(0.5)  # Replace with actual progress tracking logic
                    pbar.update(10)
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Download failed for {crunch_name}: {e}")
            return False

    def process_crunch(self, crunch_name: str, dry_run: bool = False) -> float:
        """
        Process a single Crunch, downloading its data in parallel.

        Args:
            crunch_name (str): Name of the Crunch.
            dry_run (bool): If True, simulate the download.

        Returns:
            float: The runtime for processing this Crunch in seconds.
        """
        config = self.config_loader.get_crunch_setting(crunch_name, "paths")
        competition_name = self.config_loader.get_crunch_setting(crunch_name, "crunch_type")
        project_name = self.config_loader.get_crunch_setting(crunch_name, "name")
        dataset_size = self.config_loader.get_crunch_setting(crunch_name, "dataset_size", "default")
        output_dir = config["project_dir"]

        if self.manifest_manager.is_downloaded(crunch_name, output_dir):
            print(f"[✔] {crunch_name} already downloaded. Skipping.")
            return 0

        print(f"Processing {crunch_name}...")

        token = self.token_handler.get_token(1)  # Adjust for token retrieval logic.

        start_time = time.time()
        success = self.download_data(
            crunch_name=crunch_name,
            competition_name=competition_name,
            project_name=project_name,
            dataset_size=dataset_size,
            token=token,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        runtime = time.time() - start_time

        if success:
            size = self.manifest_manager.calculate_directory_size(output_dir) if os.path.exists(output_dir) else 0
            self.manifest_manager.update_manifest(crunch_name, output_dir, "downloaded", size)
            print(f"[✔] Successfully processed {crunch_name} in {runtime:.2f} seconds.")
        else:
            print(f"[✘] Failed to process {crunch_name}.")

        return runtime

    def process_all(self, dry_run: bool = False):
        """
        Process all Crunches using inter-crunch parallelism.

        Args:
            dry_run (bool): If True, simulate the download.
        """
        crunches = self.config_loader.config.get("crunches", {})
        if not crunches:
            print("[ERROR] No Crunch configurations found.")
            return

        total_runtime = 0
        success_count = 0
        failure_count = 0
        skipped_count = 0

        def process_crunch_wrapper(crunch_name: str):
            try:
                return self.process_crunch(crunch_name, dry_run=dry_run)
            except Exception as e:
                print(f"[ERROR] Error processing {crunch_name}: {e}")
                return None

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_crunch_wrapper, crunch_name): crunch_name for crunch_name in crunches}

            with tqdm(total=len(futures), desc="Processing All Crunches", unit="Crunch", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    crunch_name = futures[future]
                    try:
                        runtime = future.result()
                        if runtime is not None:
                            if runtime > 0:
                                total_runtime += runtime
                                success_count += 1
                                print(f"[INFO] {crunch_name} completed in {runtime:.2f} seconds.")
                            else:
                                skipped_count += 1
                        else:
                            failure_count += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"[ERROR] {crunch_name} failed: {e}")
                        failure_count += 1
                        pbar.update(1)

        # Summary
        print("\n--- Summary ---")
        print(f"Total runtime: {total_runtime:.2f} seconds.")
        print(f"Crunches successfully processed: {success_count}")
        print(f"Crunches failed: {failure_count}")
        print(f"Crunches skipped (already downloaded): {skipped_count}")

    def update_manifest(self):
        """
        Update the manifest with the current state of all Crunch directories.
        """
        config = self.config_loader.config
        self.manifest_manager.update_from_config(config)


def main():
    """
    Main entry point for the CrunchDownloader script.
    """
    config_path = "../../config.yaml"
    manifest_path = "manifest.json"

    downloader = CrunchDownloader(config_path, manifest_path)

    dry_run = input("Enable dry-run mode? (y/n): ").strip().lower() == "y"

    action = input("Enter action: 'all', 'update_manifest', or Crunch name (e.g., 'crunch1'): ").strip()
    if action == "all":
        downloader.process_all(dry_run=dry_run)
    elif action == "update_manifest":
        downloader.update_manifest()
    elif action in downloader.config_loader.config.get("crunches", {}):
        downloader.process_crunch(action, dry_run=dry_run)
    else:
        print(f"[ERROR] Invalid action: {action}")


if __name__ == "__main__":
    main()
