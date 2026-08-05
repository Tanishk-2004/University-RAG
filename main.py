import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command):
    """Run a command and exit if it fails."""
    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nError: Command failed:\n{' '.join(command)}")
        sys.exit(result.returncode)

# python main.py --db_name chroma_db1

def main():
    parser = argparse.ArgumentParser(
        description="Create database if needed and launch the RAG application."
    )
    parser.add_argument(
        "--db_name",
        required=True,
        help="Name of the vector database."
    )

    args = parser.parse_args()

    # Project root (directory containing main.py)
    project_root = Path(__file__).resolve().parent

    # Database directory
    db_path = project_root c

    # Create DB only if it doesn't exist
    if db_path.exists():
        print(f"Database already exists: {db_path}")
    else:
        print(f"Database not found. Creating: {db_path}")

        run_command([
            sys.executable,
            str(project_root / "create_db.py"),
            "--db_name",
            args.db_name,
        ])

    # Launch the application
    print("\nLaunching application...\n")

    run_command([
        sys.executable,
        str(project_root / "app.py"),
        "--path_to_db",
        str(db_path),
    ])


if __name__ == "__main__":
    main()