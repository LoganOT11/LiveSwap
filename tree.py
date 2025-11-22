import os

def print_project_tree(root, prefix="", exclude_dirs=None, include_files=None):
    if exclude_dirs is None:
        exclude_dirs = [".venv", "__pycache__", ".git", "dataset"]
    if include_files is None:
        include_files = [".py", ".txt", ".wav"]

    entries = sorted(os.listdir(root))
    # Filter out unwanted dirs and files
    entries = [e for e in entries if e not in exclude_dirs and 
               (os.path.isdir(os.path.join(root, e)) or any(e.endswith(ext) for ext in include_files))]

    for i, entry in enumerate(entries):
        path = os.path.join(root, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "│   "
            print_project_tree(path, prefix + extension, exclude_dirs, include_files)

# Get the absolute path of the script you are running right now
current_script_path = os.path.abspath(__file__)

# Get the folder containing this script
current_folder = os.path.dirname(current_script_path)

# Print the project tree starting from the current folder
project_root = current_folder
print_project_tree(project_root)
