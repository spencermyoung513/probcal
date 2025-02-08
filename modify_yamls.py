import sys
from pathlib import Path

from ruamel.yaml import YAML


def update_yaml_value(file_path, key_name, new_value):
    """
    Updates a specified key's value in a YAML file while preserving formatting and comments.

    Args:
        file_path: Path to the YAML file
        key_name: Name of the key to update
        new_value: New value to set for the specified key

    Returns:
        bool: True if changes were made, False otherwise
    """
    # Initialize YAML parser with round-trip mode to preserve formatting
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Read the existing YAML file
    with open(file_path, "r") as file:
        data = yaml.load(file)

    # Track whether we made any changes
    changes_made = False

    def update_value_recursive(d):
        """
        Recursively search through nested structures to find and update the specified key.

        Args:
            d: Current dictionary or list being processed

        Returns:
            bool: True if any changes were made in this branch of the structure
        """
        nonlocal changes_made

        if isinstance(d, dict):
            # Check if the current dictionary contains our target key
            if key_name in d:
                if d[key_name] != new_value:
                    d[key_name] = new_value
                    changes_made = True
            # Continue searching through nested dictionaries
            for v in d.values():
                update_value_recursive(v)
        elif isinstance(d, list):
            # Search through list items
            for item in d:
                update_value_recursive(item)
        return changes_made

    # Apply the update and check if changes were made
    changes_made = update_value_recursive(data)

    # Only write to file if we actually made changes
    if changes_made:
        with open(file_path, "w") as file:
            yaml.dump(data, file)
        return True
    return False


def main():
    # Check if we have the correct number of arguments
    if len(sys.argv) < 4:
        print("Usage: python update_yaml.py <directory_path> <key_name> <new_value>")
        print("Example: python update_yaml.py ./config_files lambda 0.5")
        sys.exit(1)

    # Parse command line arguments
    directory = Path(sys.argv[1])
    key_name = sys.argv[2]
    new_value = sys.argv[3]

    # Validate that the directory exists
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)

    # Try to convert the new value to appropriate type
    # First attempt float conversion
    try:
        new_value = float(new_value)
        # If it's a whole number, convert to int
        if new_value.is_integer():
            new_value = int(new_value)
    except ValueError:
        # If conversion fails, try boolean conversion
        if new_value.lower() == "true":
            new_value = True
        elif new_value.lower() == "false":
            new_value = False
        # Otherwise, keep it as a string

    if new_value.lower() == "none":
        new_value = None

    # Find all YAML files recursively
    yaml_files = list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))

    if not yaml_files:
        print(f"No YAML files found in {directory} or its subdirectories")
        return

    # Initialize counters for reporting
    total_files = len(yaml_files)
    files_updated = 0
    files_skipped = 0
    files_errored = 0

    print(f"Found {total_files} YAML files to process")
    print(f"Looking for key '{key_name}' to update to value: {new_value}")

    for yaml_file in yaml_files:
        try:
            # Get relative path for cleaner output
            relative_path = yaml_file.relative_to(directory)
            print(f"Processing {relative_path}...", end=" ")

            if update_yaml_value(yaml_file, key_name, new_value):
                print("Updated!")
                files_updated += 1
            else:
                print(f"No '{key_name}' values found or values already up to date")
                files_skipped += 1

        except Exception as e:
            print(f"Error: {str(e)}")
            files_errored += 1

    # Print summary report
    print("\nSummary:")
    print(f"Total files processed: {total_files}")
    print(f"Files updated: {files_updated}")
    print(f"Files skipped (no changes needed): {files_skipped}")
    print(f"Files with errors: {files_errored}")


if __name__ == "__main__":
    main()
