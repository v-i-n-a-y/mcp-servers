import difflib
import fnmatch
import os
import argparse
import logging
import base64
import mimetypes
import glob
import json
from datetime import datetime
from typing import Dict, Any, List
from functools import wraps

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("filesystem-tool")

# Global Exclusion List (files/folders not to be processed)
EXCLUDED_ITEMS = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".venv",
    "venv",
    ".env",
    ".idea",
    ".vscode",
    "*.egg-info",
    "dist",
    "build",
    ".pytest_cache",
    ".coverage",
    "htmlcov",
    ".DS_Store",  # macOS
    "Thumbs.db",  # Windows
]

ALLOWED_DIRECTORIES: List[str] = []


def configure_allowed_directories(dirs: List[str]) -> None:
    """Configure allowed directories for the server."""
    global ALLOWED_DIRECTORIES
    ALLOWED_DIRECTORIES = [os.path.abspath(d) for d in dirs]


def is_path_allowed(path: str) -> bool:
    """Check if a path is within the allowed directories."""
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(allowed) for allowed in ALLOWED_DIRECTORIES)


def is_excluded(path: str) -> bool:
    """Check if the given file/directory path matches any exclusion pattern."""
    for pattern in EXCLUDED_ITEMS:
        if fnmatch.fnmatch(path, pattern):
            return True
    return False


def exclude_files(func):
    """Decorator to exclude files that match exclusion patterns."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        path = args[0]  # Assume the first argument is the path
        if is_excluded(path):
            return {
                "success": False,
                "error": f"{path} is excluded from this operation.",
            }
        return func(*args, **kwargs)

    return wrapper


@mcp.tool()
def get_allowed_directories():
    """Returns the list of allowed directories."""
    return ALLOWED_DIRECTORIES


@mcp.tool()
async def list_directory(path: str) -> Dict[str, Any]:
    """List the files and folders in the given directory."""
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"Access to {path} is not allowed."}

        entries = []
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            entry_type = "dir" if os.path.isdir(full_path) else "file"
            entries.append({"name": name, "type": entry_type})

        return {"success": True, "data": entries}
    except Exception as e:
        logger.error(f"Failed to list directory: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_file_line_count(filepath: str) -> Dict[str, Any]:
    """Get the number of lines in a file."""
    try:
        if not is_path_allowed(filepath):
            return {"success": False, "error": f"Access to {filepath} is not allowed."}

        with open(filepath, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)

        return {"success": True, "data": {"line_count": line_count}}

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return {"success": False, "error": f"File not found: {filepath}"}
    except Exception as e:
        logger.error(f"Failed to get line count for {filepath}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def read_file(
    filepath: str, head: int = None, tail: int = None
) -> Dict[str, Any]:
    """Read a file content with head and tail options."""
    try:
        if not is_path_allowed(filepath):
            return {"success": False, "error": f"Path {filepath} not allowed"}

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # If both head and tail are provided, return a combination of both
        if head is not None and tail is not None:
            head_lines = lines[:head]
            tail_lines = lines[-tail:]
            combined_lines = head_lines + tail_lines
            return {"success": True, "data": {"content": "".join(combined_lines)}}

        # If only head is specified, return the first 'head' lines
        elif head is not None:
            return {"success": True, "data": {"content": "".join(lines[:head])}}

        # If only tail is specified, return the last 'tail' lines
        elif tail is not None:
            return {"success": True, "data": {"content": "".join(lines[-tail:])}}

        # If neither head nor tail is provided, return the entire file content
        return {"success": True, "data": {"content": "".join(lines)}}

    except Exception as e:
        logger.error(f"Failed to read file {filepath}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def replace_in_file(
    filepath: str, old_text: str, new_text: str
) -> Dict[str, Any]:
    """Replace occurrences of old_text with new_text in a file."""
    try:
        if not is_path_allowed(filepath):
            return {"success": False, "error": f"Access to {filepath} is not allowed."}

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        logger.debug(f"Original content in file {filepath}:\n{content}")

        new_content = content.replace(old_text, new_text)

        if content == new_content:
            return {
                "success": True,
                "data": {"message": "No replacements made. Text already present."},
            }

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        return {
            "success": True,
            "data": {
                "message": f"Replaced '{old_text}' with '{new_text}' in {filepath}"
            },
        }

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return {"success": False, "error": f"File not found: {filepath}"}
    except Exception as e:
        logger.error(f"Failed to replace text in file {filepath}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def create_new_file(filepath: str, content: str) -> Dict[str, Any]:
    """Create a new file with the specified content."""
    try:
        if not is_path_allowed(filepath):
            return {"success": False, "error": f"Access to {filepath} is not allowed."}

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return {"success": True, "data": {"message": f"New file created at {filepath}"}}

    except Exception as e:
        logger.error(f"Failed to create file {filepath}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def delete_file(filepath: str) -> Dict[str, Any]:
    """Delete a file or directory."""
    try:
        if not is_path_allowed(filepath):
            return {"success": False, "error": f"Access to {filepath} is not allowed."}

        if os.path.isfile(filepath):
            os.remove(filepath)
            return {"success": True, "data": {"message": f"File deleted: {filepath}"}}
        elif os.path.isdir(filepath):
            import shutil

            shutil.rmtree(filepath)
            return {
                "success": True,
                "data": {"message": f"Directory deleted: {filepath}"},
            }
        else:
            return {
                "success": False,
                "error": f"File or directory not found: {filepath}",
            }

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return {"success": False, "error": f"File not found: {filepath}"}
    except Exception as e:
        logger.error(f"Failed to delete file/directory {filepath}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def create_directory(path: str) -> Dict[str, Any]:
    """Create a directory at the specified path."""
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}

        os.makedirs(path, exist_ok=True)
        return {"success": True, "data": {"message": f"Directory created at {path}"}}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def rename_move_file(source: str, destination: str) -> Dict[str, Any]:
    """Rename or move a file from source to destination."""
    try:
        if not (is_path_allowed(source) and is_path_allowed(destination)):
            return {"success": False, "error": "Not allowed"}

        if os.path.exists(destination):
            return {"success": False, "error": "Destination already exists"}

        os.rename(source, destination)
        return {
            "success": True,
            "data": {"message": f"File renamed/moved from {source} to {destination}"},
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _tree(path: str, exclude: List[str] = []) -> Dict[str, Any]:
    """Generate a directory tree."""
    node = {"name": os.path.basename(path), "type": "directory", "children": []}
    exclude_patterns = (
        EXCLUDED_ITEMS + exclude
    )  # Merge global exclusions with local ones

    try:
        for name in os.listdir(path):
            full_path = os.path.join(path, name)

            if is_excluded(full_path):
                continue

            if os.path.isdir(full_path):
                node["children"].append(_tree(full_path, exclude_patterns))
            else:
                node["children"].append({"name": name, "type": "file"})

    except Exception as e:
        logger.error(f"Error generating tree for {path}: {e}")

    return node


@mcp.tool()
async def directory_tree(path: str, excludePatterns: List[str] = []) -> Dict[str, Any]:
    """Generates a directory tree for the specified path, considering exclusions."""
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}

        # Merge the global exclusions with the ones provided in the function call
        combined_exclusions = EXCLUDED_ITEMS + excludePatterns

        return {"success": True, "data": _tree(path, combined_exclusions)}

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def search_files_by_metadata(
    path: str,
    file_type: str = None,
    size_min: int = None,
    size_max: int = None,
    modified_since: str = None,
) -> Dict[str, Any]:
    """Search for files by metadata like type, size, or modification date."""
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}

        # Search logic based on file metadata
        matches = []
        for root, dirs, files in os.walk(path):
            for name in files:
                full_path = os.path.join(root, name)
                file_type_ok = file_type is None or name.endswith(file_type)
                file_size_ok = (
                    size_min is None or os.path.getsize(full_path) >= size_min
                ) and (size_max is None or os.path.getsize(full_path) <= size_max)
                if file_type_ok and file_size_ok:
                    matches.append(full_path)

        return {"success": True, "data": {"matches": matches}}

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def compare_directories(dir1: str, dir2: str) -> Dict[str, Any]:
    """Compare two directories and list differences."""
    try:
        if not (is_path_allowed(dir1) and is_path_allowed(dir2)):
            return {
                "success": False,
                "error": "Access to one or both directories is not allowed.",
            }

        diff = []
        # List files in both directories
        dir1_files = set(os.listdir(dir1))
        dir2_files = set(os.listdir(dir2))

        # Find files that are in dir1 but not dir2, and vice versa
        only_in_dir1 = dir1_files - dir2_files
        only_in_dir2 = dir2_files - dir1_files
        common_files = dir1_files & dir2_files

        for file in only_in_dir1:
            diff.append(f"Only in {dir1}: {file}")
        for file in only_in_dir2:
            diff.append(f"Only in {dir2}: {file}")
        for file in common_files:
            # Compare file metadata (size, last modified date)
            file1 = os.path.join(dir1, file)
            file2 = os.path.join(dir2, file)
            if os.path.getsize(file1) != os.path.getsize(file2):
                diff.append(f"Size mismatch for {file}")
            elif os.path.getmtime(file1) != os.path.getmtime(file2):
                diff.append(f"Last modified date mismatch for {file}")

        return {"success": True, "data": {"differences": diff}}

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def change_file_permissions(filepath: str, permissions: str) -> Dict[str, Any]:
    """Change the permissions of a file or directory."""
    try:
        if not is_path_allowed(filepath):
            return {"success": False, "error": f"Access to {filepath} is not allowed."}

        os.chmod(filepath, int(permissions, 8))  # Convert octal string to int
        return {
            "success": True,
            "data": {"message": f"Permissions changed for {filepath}"},
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


import shutil


@mcp.tool()
async def get_disk_usage(path: str) -> Dict[str, Any]:
    """Get disk usage for a given path or mount point."""
    try:
        total, used, free = shutil.disk_usage(path)
        return {"success": True, "data": {"total": total, "used": used, "free": free}}

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allowed", nargs="+", required=True, help="Allowed directories"
    )
    args = parser.parse_args()
    configure_allowed_directories(args.allowed)

    logger.info(f"Allowed directories: {args.allowed}")
    mcp.run()


if __name__ == "__main__":
    main()
