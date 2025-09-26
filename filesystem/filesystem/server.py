import os
import logging
from typing import Dict, Any
import glob
import hashlib  # For checksum calculation
import asyncio
from mcp.server.fastmcp import FastMCP

def get_logger(name: str):
    logger = logging.getLogger(name)
    return logger

logger = get_logger(__name__)

# Create server instance using FastMCP
mcp = FastMCP("filesystem-tool")

ALLOWED_DIRECTORIES = ["/home/vinay/Desktop/perseus"]  # Default allowed directory - configurable!

@mcp.tool()
def get_allowed_directories() -> list[str]:
    """Returns the list of allowed directories."""
    return ALLOWED_DIRECTORIES

def is_path_allowed(path: str) -> bool:
    """Check if a path is within the allowed directories."""
    for allowed_dir in ALLOWED_DIRECTORIES:
        if path.startswith(allowed_dir):
            return True
    return False

@mcp.tool()
async def filesystem_list_directory(path: str) -> Dict[str, Any]:
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
async def filesystem_read_file(filepath: str) -> Dict[str, Any]:
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
async def filesystem_search_files(path: str, pattern: str) -> Dict[str, Any]:
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
async def filesystem_create_file(filepath: str) -> Dict[str, Any]:
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
async def filesystem_delete_file(filepath: str) -> Dict[str, Any]:
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
async def filesystem_search_file_content(path: str, query: str) -> Dict[str, Any]:
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

def main() -> None:
    """Run the MCP server."""
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    mcp.run()
