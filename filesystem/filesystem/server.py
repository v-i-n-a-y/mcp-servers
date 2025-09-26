import difflib
import os
import argparse
import logging
import base64
import mimetypes
import glob
import json
from datetime import datetime
from typing import Dict, Any, List

from mcp.server.fastmcp import FastMCP

def get_logger(name: str):
    logger = logging.getLogger(name)
    return logger

logger = get_logger(__name__)

mcp = FastMCP("filesystem-tool")

ALLOWED_DIRECTORIES: List[str] = []


def configure_allowed_directories(dirs: List[str]) -> None:
    """Configure allowed directories for the server."""
    global ALLOWED_DIRECTORIES
    ALLOWED_DIRECTORIES = [os.path.abspath(d) for d in dirs]

@mcp.tool()
def get_allowed_directories():
    """Returns the list of allowed directories."""
    return ALLOWED_DIRECTORIES

def is_path_allowed(path: str) -> bool:
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(d) for d in ALLOWED_DIRECTORIES)

@mcp.tool()
async def list_directory(path: str) -> Dict[str, Any]:
    """List files and directories within a given directory."""
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"Path {path} is outside allowed directories."}
        entries = os.listdir(path)
        return {
            "success": True,
            "data": {
                "entries": entries
            }
        }
    except FileNotFoundError:
        logger.error(f"Directory not found: {path}")
        return {
            "success": False,
            "error": f"Directory not found: {path}"
        }
    except Exception as e:
        logger.error(str(e))
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def read_file(filepath: str) -> Dict[str, Any]:
    """Read the content of a file."""
    try:
        if not is_path_allowed(filepath):
            return {"success": False, "error": f"Path {filepath} is outside allowed directories."}
        with open(filepath, 'r') as f:
            content = f.read()
            return {
                "success": True,
                "data": {
                    "content": content
                }
            }
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return {
            "success": False,
            "error": f"File not found: {filepath}"
        }
    except Exception as e:
        logger.error(str(e))
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def search_files(path: str, pattern: str) -> Dict[str, Any]:
    """Search for files matching a given pattern within a directory."""
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"Path {path} is outside allowed directories."}
        matches = glob.glob(os.path.join(path, pattern))
        return {
            "success": True,
            "data": {
                "matches": matches
            }
        }
    except Exception as e:
        logger.error(str(e))
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def create_file(filepath: str) -> Dict[str, Any]:
    """Create an empty file at the specified path."""
    try:
        if not is_path_allowed(filepath):
            return {"success": False, "error": f"Path {filepath} is outside allowed directories."}
        with open(filepath, 'w') as f:
            pass  # Create an empty file
        return {"success": True, "data": {"message": f"File created at {filepath}"}}
    except Exception as e:
        logger.error(str(e))
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def delete_file(filepath: str) -> Dict[str, Any]:
    """Delete a file at the specified path."""
    try:
        if not is_path_allowed(filepath):
            return {"success": False, "error": f"Path {filepath} is outside allowed directories."}
        os.remove(filepath)
        return {"success": True, "data": {"message": f"File deleted at {filepath}"}}
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return {
            "success": False,
            "error": f"File not found: {filepath}"
        }
    except Exception as e:
        logger.error(str(e))
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def search_file_content(path: str, query: str) -> Dict[str, Any]:
    """Search for files containing a specific string within their content."""
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"Path {path} is outside allowed directories."}
        matches = []
        for root, _, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if query in content:
                            matches.append(filepath)
                except Exception:
                    pass  # Ignore files that can't be read
        return {
            "success": True,
            "data": {
                "matches": matches
            }
        }
    except Exception as e:
        logger.error(str(e))
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def read_text_file(path: str, head: int = None, tail: int = None) -> Dict[str, Any]:
    """Read a UTF-8 text file (with head/tail options)."""
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if head and tail:
            return {"success": False, "error": "Cannot specify both head and tail"}
        if head:
            lines = lines[:head]
        elif tail:
            lines = lines[-tail:]
        return {"success": True, "data": {"content": "".join(lines)}}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def read_media_file(path: str) -> Dict[str, Any]:
    """Read an image/audio file as base64."""
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}
        mime_type, _ = mimetypes.guess_type(path)
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return {"success": True, "data": {"mime": mime_type, "base64": data}}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def read_multiple_files(paths: List[str]) -> Dict[str, Any]:
    results = {}
    for p in paths:
        if not is_path_allowed(p):
            results[p] = {"success": False, "error": "Not allowed"}
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                results[p] = {"success": True, "content": f.read()}
        except Exception as e:
            results[p] = {"success": False, "error": str(e)}
    return {"success": True, "data": results}

@mcp.tool()
async def write_file(path: str, content: str) -> Dict[str, Any]:
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "data": {"message": f"Wrote {path}"}}
    except Exception as e:
        return {"success": False, "error": str(e)}

import difflib

@mcp.tool()
async def edit_file(path: str, edits: List[Dict[str, str]], dryRun: bool = False) -> Dict[str, Any]:
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = content
        for e in edits:
            old, new = e["oldText"], e["newText"]
            new_content = new_content.replace(old, new)

        if dryRun:
            diff = "\n".join(difflib.unified_diff(
                content.splitlines(), new_content.splitlines(),
                fromfile=path, tofile=path, lineterm=""
            ))
            return {"success": True, "data": {"diff": diff}}

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return {"success": True, "data": {"message": f"Edited {path}"}}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def create_directory(path: str) -> Dict[str, Any]:
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}
        os.makedirs(path, exist_ok=True)
        return {"success": True, "data": {"message": f"Directory created at {path}"}}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def list_directory_with_sizes(path: str, sortBy: str = "name") -> Dict[str, Any]:
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}
        entries = []
        total_size = total_files = total_dirs = 0
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                size = os.path.getsize(full)
                entries.append({"name": name, "type": "file", "size": size})
                total_files += 1
                total_size += size
            else:
                entries.append({"name": name, "type": "dir", "size": 0})
                total_dirs += 1
        if sortBy == "size":
            entries.sort(key=lambda e: e["size"], reverse=True)
        else:
            entries.sort(key=lambda e: e["name"])
        return {"success": True, "data": {
            "entries": entries,
            "summary": {"files": total_files, "dirs": total_dirs, "total_size": total_size}
        }}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def move_file(source: str, destination: str) -> Dict[str, Any]:
    try:
        if not (is_path_allowed(source) and is_path_allowed(destination)):
            return {"success": False, "error": "Not allowed"}
        if os.path.exists(destination):
            return {"success": False, "error": "Destination exists"}
        os.rename(source, destination)
        return {"success": True, "data": {"message": f"Moved {source} -> {destination}"}}
    except Exception as e:
        return {"success": False, "error": str(e)}

def _tree(path: str, exclude: List[str] = []) -> Dict[str, Any]:
    node = {"name": os.path.basename(path), "type": "directory", "children": []}
    try:
        for entry in os.listdir(path):
            if any(glob.fnmatch.fnmatch(entry, pat) for pat in exclude):
                continue
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                node["children"].append(_tree(full, exclude))
            else:
                node["children"].append({"name": entry, "type": "file"})
    except Exception:
        pass
    return node

@mcp.tool()
async def directory_tree(path: str, excludePatterns: List[str] = []) -> Dict[str, Any]:
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}
        return {"success": True, "data": _tree(path, excludePatterns)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_file_info(path: str) -> Dict[str, Any]:
    try:
        if not is_path_allowed(path):
            return {"success": False, "error": f"{path} not allowed"}
        stat = os.stat(path)
        return {"success": True, "data": {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "type": "directory" if os.path.isdir(path) else "file",
            "permissions": oct(stat.st_mode)[-3:]
        }}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed", nargs="+", required=True, help="Allowed directories")
    args = parser.parse_args()
    configure_allowed_directories("/Users/vinay/Desktop")

    logger.info(f"Allowed directories: {ALLOWED_DIRECTORIES}")
    mcp.run()

if __name__ == "__main__":
    mcp.run()
