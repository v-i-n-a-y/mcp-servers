#!/usr/bin/env python3

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
from mcp.server.fastmcp import FastMCP
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-reader-server")

# Initialize FastMCP server
mcp = FastMCP("PDF Reader Server")

def validate_file_path(file_path: str) -> Path:
    """Validate that the file path exists and is a PDF"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {file_path}")
    return path

def get_page_range(doc: fitz.Document, page_range: Optional[Dict] = None) -> Tuple[int, int]:
    """Get validated page range for the document"""
    total_pages = len(doc)
    
    if page_range is None:
        return 0, total_pages - 1
    
    start = page_range.get('start', 1) - 1  # Convert to 0-based indexing
    end = page_range.get('end', total_pages) - 1
    
    start = max(0, min(start, total_pages - 1))
    end = max(start, min(end, total_pages - 1))
    
    return start, end

@mcp.tool()
def read_pdf_text(file_path: str, page_range: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Extract text content from a PDF file
    
    Args:
        file_path: Path to the PDF file to read
        page_range: Optional dict with 'start' and 'end' page numbers (1-indexed)
    
    Returns:
        Dictionary containing extracted text and metadata
    """
    try:
        path = validate_file_path(file_path)
        
        with fitz.open(str(path)) as doc:
            start_page, end_page = get_page_range(doc, page_range)
            
            pages_text = []
            total_text = ""
            
            for page_num in range(start_page, end_page + 1):
                page = doc[page_num]
                page_text = page.get_text()
                pages_text.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "word_count": len(page_text.split())
                })
                total_text += page_text + "\n"
            
            return {
                "success": True,
                "file_path": str(path),
                "pages_processed": f"{start_page + 1}-{end_page + 1}",
                "total_pages": len(doc),
                "pages_text": pages_text,
                "combined_text": total_text.strip(),
                "total_word_count": len(total_text.split()),
                "total_character_count": len(total_text)
            }
            
    except Exception as e:
        logger.error(f"Error reading PDF text: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

@mcp.tool()
def extract_pdf_images(file_path: str, output_dir: Optional[str] = None, page_range: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Extract all images from a PDF file
    
    Args:
        file_path: Path to the PDF file
        output_dir: Directory to save extracted images (optional, defaults to temp dir)
        page_range: Optional dict with 'start' and 'end' page numbers (1-indexed)
    
    Returns:
        Dictionary containing information about extracted images
    """
    try:
        path = validate_file_path(file_path)
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="pdf_images_")
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        extracted_images = []
        
        with fitz.open(str(path)) as doc:
            start_page, end_page = get_page_range(doc, page_range)
            
            for page_num in range(start_page, end_page + 1):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Skip if image is too small or has alpha channel issues
                        if pix.width < 10 or pix.height < 10:
                            pix = None
                            continue
                        
                        # Convert to PNG if needed
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        # Save image
                        img_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                        img_path = Path(output_dir) / img_filename
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        extracted_images.append({
                            "page_number": page_num + 1,
                            "image_index": img_index + 1,
                            "filename": img_filename,
                            "path": str(img_path),
                            "width": pix.width,
                            "height": pix.height,
                            "size_bytes": len(img_data)
                        })
                        
                        pix = None
                        
                    except Exception as img_error:
                        logger.warning(f"Failed to extract image {img_index + 1} from page {page_num + 1}: {img_error}")
                        continue
        
        return {
            "success": True,
            "file_path": str(path),
            "output_directory": output_dir,
            "pages_processed": f"{start_page + 1}-{end_page + 1}",
            "images_extracted": len(extracted_images),
            "images": extracted_images
        }
        
    except Exception as e:
        logger.error(f"Error extracting PDF images: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

@mcp.tool()
def read_pdf_with_ocr(file_path: str, page_range: Optional[Dict] = None, ocr_language: str = "eng") -> Dict[str, Any]:
    """
    Extract text from PDF including OCR text from images
    
    Args:
        file_path: Path to the PDF file
        page_range: Optional dict with 'start' and 'end' page numbers (1-indexed)
        ocr_language: OCR language code (default: 'eng')
    
    Returns:
        Dictionary containing extracted text from both text and images
    """
    try:
        path = validate_file_path(file_path)
        
        with fitz.open(str(path)) as doc:
            start_page, end_page = get_page_range(doc, page_range)
            
            pages_data = []
            total_text = ""
            total_ocr_text = ""
            
            for page_num in range(start_page, end_page + 1):
                page = doc[page_num]
                
                # Extract regular text
                page_text = page.get_text()
                
                # Extract and OCR images
                image_texts = []
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Skip very small images
                        if pix.width < 50 or pix.height < 50:
                            pix = None
                            continue
                        
                        # Convert to PIL Image for OCR
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        # Perform OCR
                        with Image.open(BytesIO(img_data)) as pil_image:
                            ocr_text = pytesseract.image_to_string(
                                pil_image, 
                                lang=ocr_language,
                                config='--psm 6'  # Uniform block of text
                            ).strip()
                            
                            if ocr_text:
                                image_texts.append({
                                    "image_index": img_index + 1,
                                    "ocr_text": ocr_text,
                                    "confidence": "high" if len(ocr_text) > 10 else "low"
                                })
                        
                        pix = None
                        
                    except Exception as ocr_error:
                        logger.warning(f"OCR failed for image {img_index + 1} on page {page_num + 1}: {ocr_error}")
                        continue
                
                # Combine all OCR text from this page
                page_ocr_text = "\n".join([img["ocr_text"] for img in image_texts])
                
                page_data = {
                    "page_number": page_num + 1,
                    "text": page_text,
                    "ocr_text": page_ocr_text,
                    "images_with_text": image_texts,
                    "combined_text": f"{page_text}\n{page_ocr_text}".strip(),
                    "text_word_count": len(page_text.split()),
                    "ocr_word_count": len(page_ocr_text.split())
                }
                
                pages_data.append(page_data)
                total_text += page_text + "\n"
                total_ocr_text += page_ocr_text + "\n"
            
            combined_all_text = f"{total_text}\n{total_ocr_text}".strip()
            
            return {
                "success": True,
                "file_path": str(path),
                "pages_processed": f"{start_page + 1}-{end_page + 1}",
                "total_pages": len(doc),
                "ocr_language": ocr_language,
                "pages_data": pages_data,
                "summary": {
                    "total_text_word_count": len(total_text.split()),
                    "total_ocr_word_count": len(total_ocr_text.split()),
                    "combined_word_count": len(combined_all_text.split()),
                    "combined_character_count": len(combined_all_text),
                    "images_processed": sum(len(p["images_with_text"]) for p in pages_data)
                },
                "combined_text": total_text.strip(),
                "combined_ocr_text": total_ocr_text.strip(),
                "all_text_combined": combined_all_text
            }
            
    except Exception as e:
        logger.error(f"Error reading PDF with OCR: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

@mcp.tool()
def get_pdf_info(file_path: str) -> Dict[str, Any]:
    """
    Get metadata and information about a PDF file
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        Dictionary containing PDF metadata and statistics
    """
    try:
        path = validate_file_path(file_path)
        file_stats = path.stat()
        
        with fitz.open(str(path)) as doc:
            # Get basic document info
            metadata = doc.metadata
            
            # Count images across all pages
            total_images = 0
            page_info = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                images_on_page = len(page.get_images())
                total_images += images_on_page
                
                page_info.append({
                    "page_number": page_num + 1,
                    "images_count": images_on_page,
                    "text_length": len(page.get_text()),
                    "has_text": bool(page.get_text().strip()),
                    "page_width": page.rect.width,
                    "page_height": page.rect.height
                })
            
            return {
                "success": True,
                "file_path": str(path),
                "file_info": {
                    "size_bytes": file_stats.st_size,
                    "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                    "created": file_stats.st_ctime,
                    "modified": file_stats.st_mtime
                },
                "pdf_metadata": {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", "")
                },
                "document_stats": {
                    "total_pages": len(doc),
                    "total_images": total_images,
                    "pages_with_text": sum(1 for p in page_info if p["has_text"]),
                    "pages_with_images": sum(1 for p in page_info if p["images_count"] > 0),
                    "is_encrypted": doc.needs_pass,
                    "can_extract_text": not doc.is_closed
                },
                "page_details": page_info
            }
            
    except Exception as e:
        logger.error(f"Error getting PDF info: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

@mcp.tool()
def analyze_pdf_structure(file_path: str) -> Dict[str, Any]:
    """
    Analyze PDF structure including pages, images, and text blocks
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        Dictionary containing detailed structural analysis
    """
    try:
        path = validate_file_path(file_path)
        
        with fitz.open(str(path)) as doc:
            structure_analysis = {
                "document_structure": {
                    "total_pages": len(doc),
                    "is_encrypted": doc.needs_pass,
                    "pdf_version": doc.pdf_version() if hasattr(doc, 'pdf_version') else "unknown"
                },
                "content_analysis": {
                    "pages_with_text": 0,
                    "pages_with_images": 0,
                    "pages_text_only": 0,
                    "pages_images_only": 0,
                    "pages_mixed_content": 0,
                    "total_text_blocks": 0,
                    "total_images": 0
                },
                "page_details": []
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text blocks
                text_blocks = page.get_text("dict")["blocks"]
                text_block_count = len([block for block in text_blocks if "lines" in block])
                
                # Get images
                images = page.get_images()
                image_count = len(images)
                
                # Get text
                page_text = page.get_text().strip()
                has_text = bool(page_text)
                has_images = image_count > 0
                
                # Categorize page content
                if has_text and has_images:
                    content_type = "mixed"
                    structure_analysis["content_analysis"]["pages_mixed_content"] += 1
                elif has_text:
                    content_type = "text_only"
                    structure_analysis["content_analysis"]["pages_text_only"] += 1
                elif has_images:
                    content_type = "images_only"
                    structure_analysis["content_analysis"]["pages_images_only"] += 1
                else:
                    content_type = "empty"
                
                if has_text:
                    structure_analysis["content_analysis"]["pages_with_text"] += 1
                if has_images:
                    structure_analysis["content_analysis"]["pages_with_images"] += 1
                
                structure_analysis["content_analysis"]["total_text_blocks"] += text_block_count
                structure_analysis["content_analysis"]["total_images"] += image_count
                
                page_detail = {
                    "page_number": page_num + 1,
                    "content_type": content_type,
                    "text_blocks": text_block_count,
                    "image_count": image_count,
                    "text_length": len(page_text),
                    "dimensions": {
                        "width": page.rect.width,
                        "height": page.rect.height
                    },
                    "rotation": page.rotation
                }
                
                structure_analysis["page_details"].append(page_detail)
            
            # Add summary statistics
            structure_analysis["summary"] = {
                "content_distribution": {
                    "text_only_pages": structure_analysis["content_analysis"]["pages_text_only"],
                    "images_only_pages": structure_analysis["content_analysis"]["pages_images_only"],
                    "mixed_content_pages": structure_analysis["content_analysis"]["pages_mixed_content"],
                    "empty_pages": len(doc) - sum([
                        structure_analysis["content_analysis"]["pages_text_only"],
                        structure_analysis["content_analysis"]["pages_images_only"],
                        structure_analysis["content_analysis"]["pages_mixed_content"]
                    ])
                },
                "avg_images_per_page": round(structure_analysis["content_analysis"]["total_images"] / len(doc), 2),
                "avg_text_blocks_per_page": round(structure_analysis["content_analysis"]["total_text_blocks"] / len(doc), 2)
            }
            
            return {
                "success": True,
                "file_path": str(path),
                **structure_analysis
            }
            
    except Exception as e:
        logger.error(f"Error analyzing PDF structure: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

if __name__ == "__main__":
    mcp.run()
