import os
import glob
from pathlib import Path
import shutil
import sys

def confirm_deletion():
    """Prompt user to confirm deletion."""
    response = input("Are you sure you want to delete generated files and folders? (y/n): ")
    return response.lower() == 'y'

def clean_project(project_dir="."):
    """
    Clean generated files and folders in the project directory, protecting all files in the 'data' folder.
    Retains 'figures' and 'output' folders but deletes their contents.

    Parameters:
        project_dir (str): Root directory of the project (default: current directory).
    """
    project_dir = Path(project_dir)
    print(f"Cleaning project directory: {project_dir.resolve()}")

    # List of files and folders to delete
    patterns = [
        # Data files in root or other directories
        "*.csv",
        "*.parquet",
        # Specific generated files
        "nuts2_passenger_demand.csv",
        "region_level_enriched.csv",
        "enriched_route_level.parquet",
        "route_consistency.parquet",
        "route_consistency_summary.parquet",
        "moran_results.csv",
        # Contents of specific directories (excluding 'figures' and 'output' folders themselves)
        "decade_exports/*",
        "shap_outputs/*",
        "figures/*",  # Delete files inside 'figures' but keep the folder
        "output/*",   # Delete files inside 'output' but keep the folder
        "figs/*",
    ]

    # List of directories to remove entirely if empty (excluding 'figures' and 'output')
    directories = [
        "decade_exports",
        "shap_outputs",
        "figs",
    ]

    # Protect all files in the 'data' folder (including subdirectories) and Python scripts
    protected_patterns = [
        "data/**/*",  # Protect everything in the 'data' folder
        "*.py",       # Protect Python scripts in the root directory
    ]

    # Collect all protected files
    protected_files = set()
    for pattern in protected_patterns:
        protected_files.update(str(p) for p in project_dir.rglob(pattern))

    # Collect files to delete
    files_to_delete = set()
    for pattern in patterns:
        files_to_delete.update(str(p) for p in project_dir.rglob(pattern))

    # Exclude protected files
    files_to_delete = files_to_delete - protected_files

    if not files_to_delete:
        print("No files to delete.")
        return

    print("Files and folders to be deleted:")
    for f in sorted(files_to_delete):
        print(f"  - {f}")

    # Confirm deletion
    if not confirm_deletion():
        print("Cleanup aborted.")
        return

    # Delete files
    for file_path in files_to_delete:
        file_path = Path(file_path)
        try:
            if file_path.is_file():
                file_path.unlink()
                print(f"Deleted file: {file_path}")
            elif file_path.is_dir():
                shutil.rmtree(file_path, ignore_errors=True)
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    # Remove empty directories (excluding 'figures' and 'output')
    for dir_name in directories:
        dir_path = project_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                dir_path.rmdir()
                print(f"Removed empty directory: {dir_path}")
            except OSError:
                pass  # Directory not empty or already removed

    print("Cleanup completed.")

if __name__ == "__main__":
    clean_project()