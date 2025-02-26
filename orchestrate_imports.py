import os
import ast
import importlib.util
import logging
import subprocess
from typing import List, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ImportOrchestrator")

def find_python_files(root: str) -> List[str]:
    """Recursively find all Python files in the given root directory."""
    python_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".py"):
                python_files.append(os.path.join(dirpath, fname))
    return python_files

def get_imported_modules(file_path: str) -> Set[str]:
    """Parse the Python file and return a set of imported module names."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            node = ast.parse(f.read(), filename=file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return set()
    modules = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                modules.add(alias.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            # Skip relative imports (level > 0)
            if n.level == 0 and n.module:
                modules.add(n.module.split('.')[0])
    return modules

def module_exists(module_name: str) -> bool:
    """Return True if module can be imported; otherwise, False."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def check_imports(root: str) -> None:
    """Scan all Python files in the given root and log any unresolved imports."""
    python_files = find_python_files(root)
    logger.info(f"Scanning {len(python_files)} Python files for import issues...")
    for file_path in python_files:
        modules = get_imported_modules(file_path)
        for mod in modules:
            if not module_exists(mod):
                logger.warning(f"Unresolved import '{mod}' in file: {file_path}")
                # Suggestion: If the module is part of your project, consider using relative imports.
                # Here you could try to search for a candidate file in your project.
                # For now, we only log the issue.

def generate_requirements(root: str, output_file: str = "requirements_full.txt") -> None:
    """
    Generate a requirements file by running pipreqs on the project directory.
    pipreqs scans your project for import statements.
    """
    try:
        # pipreqs should be installed (pip install pipreqs)
        subprocess.run(["pipreqs", root, "--force", "--savepath", output_file], check=True)
        logger.info(f"Requirements generated at {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate requirements: {e}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(".")  # Adjust as needed (e.g., "./kaleidoscope_ai")
    logger.info(f"Starting import orchestration in project root: {PROJECT_ROOT}")
    check_imports(PROJECT_ROOT)
    generate_requirements(PROJECT_ROOT)
