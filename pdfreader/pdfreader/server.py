import os
import io
import logging
from typing import Dict, Any
import asyncio
import PyPDF2
import requests
from mcp.server.fastmcp import FastMCP
import re  # For regular expressions (e.g., file type validation)

def get_logger(name: str):
    logger = logging.getLogger(name)
    return logger

logger = get_logger(__name__)

# Create server instance using FastMCP
mcp = FastMCP("pdf-reader")

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from a PDF file."""
    try:
        with PyPDF2.PdfReader(pdf_file) as file:
            text = ""
            for page in file.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def extract_metadata_from_pdf(pdf_file) -> Dict[str, Any]:
    """Extract metadata from a PDF file."""
    try:
        with PyPDF2.PdfReader(pdf_file) as file:
            info = file.metadata
            metadata = {}
            if info:  # Check if metadata exists
                for key in info:
                    metadata[key] = str(info[key]) # Convert to string for consistency
            return metadata
    except Exception as e:
        logger.error(f"Failed to extract metadata from PDF: {str(e)}")
        return {}  # Return empty dictionary on error

def is_valid_pdf(file_path: str) -> bool:
    """Check if a file is likely a PDF based on its magic number."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)  # Read the first 4 bytes
            return header == b'%PDF'
    except Exception:
        return False

async def process_pdf(path_or_url: str, task_type: str, page_numbers: list[int] = None) -> Dict[str, Any]:
    """Asynchronously processes a PDF file or URL."""
    try:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            response = requests.get(path_or_url)
            response.raise_for_status()
            pdf_file = io.BytesIO(response.content)
        else:
            if not is_valid_pdf(path_or_url):
                return {"success": False, "error": "Invalid PDF file"}
            with open(path_or_url, 'rb') as file:
                pdf_file = file

        reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(reader.pages)

        if task_type == "read":
            text = ""
            pages_to_extract = page_numbers if page_numbers else range(num_pages)
            for i in pages_to_extract:
                if 0 <= i < num_pages:  # Validate page number
                    text += reader.pages[i].extract_text() + "\n"
            return {"success": True, "data": {"text": text}}
        elif task_type == "metadata":
            metadata = extract_metadata_from_pdf(reader)
            return {"success": True, "data": {"metadata": metadata}}
        else:
            logger.error(f"Invalid task type: {task_type}")
            return {"success": False, "error": f"Invalid task type: {task_type}"}

    except requests.RequestException as e:
        logger.error(f"Failed to fetch PDF from URL: {str(e)}")
        return {"success": False, "error": f"Failed to fetch PDF from URL: {str(e)}"}
    except FileNotFoundError:
        logger.error(f"PDF file not found: {path_or_url}")
        return {"success": False, "error": f"PDF file not found: {path_or_url}"}
    except Exception as e:
        logger.error(str(e))
        return {"success": False, "error": str(e)}

@mcp.tool()
async def read_local_pdf(path: str, page_numbers: list[int] = None) -> Dict[str, Any]:
    """Read text content from a local PDF file."""
    return await process_pdf(path, "read", page_numbers)

@mcp.tool()
async def read_pdf_url(url: str, page_numbers: list[int] = None) -> Dict[str, Any]:
    """Read text content from a PDF URL."""
    return await process_pdf(url, "read", page_numbers)

@mcp.tool()
async def extract_local_pdf_metadata(path: str) -> Dict[str, Any]:
    """Extract metadata from a local PDF file."""
    return await process_pdf(path, "metadata")

@mcp.tool()
async def extract_pdf_url_metadata(url: str) -> Dict[str, Any]:
    """Extract metadata from a PDF URL."""
    return await process_pdf(url, "metadata")


def main() -> None:
    """Run the MCP server."""
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) # Use asyncio.run to start the async event loop
