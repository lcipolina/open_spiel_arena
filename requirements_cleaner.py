'''Same functionality as 'build_cleaner'. It crawls the directory and adds dependencies to the requirements.txt file'''


import os
import ast
import subprocess

def get_imported_packages(project_dir):
    """Crawl the project to find all imported packages."""
    imported_packages = set()
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    try:
                        tree = ast.parse(f.read(), filename=file_path)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imported_packages.add(alias.name.split(".")[0])
                            elif isinstance(node, ast.ImportFrom):
                                imported_packages.add(node.module.split(".")[0])
                    except SyntaxError:
                        print(f"Skipping {file_path}: Syntax Error")
    return imported_packages

def get_installed_packages():
    """Get a list of installed packages using pip freeze."""
    result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE, text=True)
    installed_packages = set(
        line.split("==")[0] for line in result.stdout.splitlines()
    )
    return installed_packages

def update_requirements(project_dir, requirements_file="requirements.txt"):
    """Update the requirements.txt file."""
    imported_packages = get_imported_packages(project_dir)
    installed_packages = get_installed_packages()

    # Determine missing and unused packages
    missing_packages = imported_packages - installed_packages
    unused_packages = installed_packages - imported_packages

    # Update requirements.txt
    with open(requirements_file, "r") as f:
        current_requirements = set(line.strip().split("==")[0] for line in f.readlines())

    updated_requirements = current_requirements | missing_packages
    with open(requirements_file, "w") as f:
        for package in sorted(updated_requirements):
            f.write(f"{package}\n")

    print("Updated requirements.txt.")
    print(f"Missing packages added: {missing_packages}")
    print(f"Unused packages: {unused_packages}")
    print("Note: Unused packages are not automatically removed. Please review manually.")

if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    update_requirements(PROJECT_DIR)
