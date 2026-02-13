import pdfplumber
from typing import Dict, Optional


class PDFParser:
    """Extract text and tables from financial report pdfs"""

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Extract all text from pdf"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error parsing PDF {pdf_path}: {str(e)}")
    
    @staticmethod
    def extract_tables(pdf_path: str) -> list:
        """Extract tables from pdf"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                tables = []
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
                return tables
        except Exception as e:
            raise Exception(f"Error extracting tables: {str(e)}")