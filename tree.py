import os

def print_project_tree(root, prefix="", exclude_dirs=None, include_files=None):
    if exclude_dirs is None:
        exclude_dirs = ["venv", "__pycache__", ".git"]
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

# Example usage
project_root = "."  # Replace with your path
print_project_tree(project_root)
