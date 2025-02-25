import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import camelot
import pdfplumber
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from PyPDF2 import PdfReader
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction"""
    pl_keywords: List[str] = None
    revenue_markers: List[str] = None
    expense_markers: List[str] = None
    profit_markers: List[str] = None
    
    def __post_init__(self):
        # Default configurations if none provided
        self.pl_keywords = self.pl_keywords or [
            "Profit and Loss", "Statement of Profit", "Income Statement",
            "Revenue", "Expenses", "consolidated statement of profit"
        ]
        self.revenue_markers = self.revenue_markers or [
            "Revenue from operations", "Total revenue", "Net revenue",
            "Other income"
        ]
        self.expense_markers = self.expense_markers or [
            "Employee benefit", "Cost of", "Operating expense",
            "Depreciation", "Finance cost"
        ]
        self.profit_markers = self.profit_markers or [
            "Profit before tax", "Net profit", "Profit for the period",
            "Earnings per share"
        ]

class PDFValidator:
    """Validates and analyzes PDF structure"""
    
    @staticmethod
    def validate_pdf(pdf_path: str) -> bool:
        """Validate if PDF is readable and contains relevant content"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = " ".join([page.extract_text() for page in pdf.pages]).lower()
                return any(keyword.lower() in text for keyword in ExtractionConfig().pl_keywords)
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False
    
    @staticmethod
    def identify_pl_pages(pdf_path: str, config: ExtractionConfig) -> List[int]:
        """Identify pages containing P&L information"""
        pl_pages = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text().lower()
                    if any(keyword.lower() in text for keyword in config.pl_keywords):
                        pl_pages.append(i + 1)
            return pl_pages
        except Exception as e:
            logger.error(f"Page identification failed: {str(e)}")
            return []

class RobustPLExtractor:
    """Robust P&L data extraction with multiple fallback methods"""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Updated ChromaDB client initialization
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        collection_name = "pl_data"

        # Check if the collection already exists
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' already exists. Using the existing collection.")
        except ValueError:
            # Create the collection if it doesn't exist
            collection = self.chroma_client.create_collection(name=collection_name)
            print(f"Collection '{collection_name}' created successfully.")


    def extract_tables(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables with multiple methods and fallbacks."""
        if not PDFValidator.validate_pdf(pdf_path):
            raise ValueError("Invalid or unsupported PDF format")

        pl_pages = PDFValidator.identify_pl_pages(pdf_path, self.config)
        if not pl_pages:
            raise ValueError("No P&L pages found in PDF")

        tables = []
        for page in pl_pages:
            try:
                # Try Camelot first
                camelot_tables = camelot.read_pdf(
                    pdf_path, 
                    pages=str(page),
                    flavor='lattice'  # Try lattice first
                )
                if not camelot_tables:
                    # Fallback to stream flavor
                    camelot_tables = camelot.read_pdf(
                        pdf_path,
                        pages=str(page),
                        flavor='stream'
                    )
                
                for table in camelot_tables:
                    if self._is_valid_pl_table(table.df):
                        tables.append(table.df)
            
            except Exception as e:
                logger.warning(f"Camelot extraction failed for page {page}, trying pdfplumber: {str(e)}")

                # Fallback to pdfplumber
                try:

                    with pdfplumber.open(pdf_path) as pdf:
                        pdf_reader = PdfReader(pdf_path)
                        page_obj = pdf.pages[page - 1]  # Note: pdf.pages is zero-indexed
                        plumber_tables = page_obj.extract_tables()

                        for table in plumber_tables:
                            # Convert table to DataFrame
                            if len(table) > 1:  # Ensure the table has rows and headers
                                df = pd.DataFrame(table[1:], columns=table[0])
                                if self._is_valid_pl_table(df):
                                    tables.append(df)
                except Exception as e2:
                    logger.error(f"All extraction methods failed for page {page}: {str(e2)}")

        return tables

    def _is_valid_pl_table(self, df: pd.DataFrame) -> bool:
        """Validate if extracted table is a P&L table"""
        if df.empty:
            return False
        
        # Convert all content to string and lowercase for checking
        text_content = df.astype(str).values.ravel()
        text_content = [str(x).lower() for x in text_content]
        
        # Check for minimum required P&L elements
        has_revenue = any(any(marker.lower() in text for text in text_content) 
                         for marker in self.config.revenue_markers)
        has_expenses = any(any(marker.lower() in text for text in text_content) 
                          for marker in self.config.expense_markers)
        has_profits = any(any(marker.lower() in text for text in text_content) 
                         for marker in self.config.profit_markers)
        
        return has_revenue and has_expenses and has_profits

    def preprocess_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize extracted table"""
        try:
            # Remove empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Convert numeric columns
            for col in df.columns:
                try:
                    # Remove common financial formatting
                    df[col] = df[col].str.replace('₹', '').str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    continue
                    
            return df
        except Exception as e:
            logger.error(f"Table preprocessing failed: {str(e)}")
            return df

    def extract_pl_components(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Extract P&L components with flexible matching"""
        components = {}
        try:
            # Function to find section boundaries
            def find_section_rows(markers: List[str]) -> List[int]:
                return df.index[df.iloc[:, 0].str.contains('|'.join(markers), 
                                                         case=False, 
                                                         na=False)].tolist()

            # Extract each section
            revenue_rows = find_section_rows(self.config.revenue_markers)
            expense_rows = find_section_rows(self.config.expense_markers)
            profit_rows = find_section_rows(self.config.profit_markers)

            if revenue_rows:
                components['revenue'] = df.loc[revenue_rows]
            if expense_rows:
                components['expenses'] = df.loc[expense_rows]
            if profit_rows:
                components['profits'] = df.loc[profit_rows]

        except Exception as e:
            logger.error(f"Component extraction failed: {str(e)}")

        return components

    def create_searchable_chunks(self, components: Dict[str, pd.DataFrame]) -> None:
        """Create searchable chunks with error handling"""
        try:
            for section, data in components.items():
                if data.empty:
                    continue
                
                # Create meaningful text chunks
                chunks = []
                metadata = []
                
                for idx, row in data.iterrows():
                    # Create a detailed text representation
                    chunk = f"{section.title()} - {row.iloc[0]}: "
                    values = [f"{col}: {val}" for col, val in 
                             zip(data.columns[1:], row.iloc[1:])]
                    chunk += ", ".join(values)
                    
                    chunks.append(chunk)
                    metadata.append({
                        "section": section,
                        "line_item": row.iloc[0],
                        "row_index": idx
                    })

                # Create embeddings
                embeddings = self.model.encode(chunks)
                
                # Store in ChromaDB
                self.collection.add(
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=metadata,
                    ids=[f"{section}_{i}" for i in range(len(chunks))]
                )
                
        except Exception as e:
            logger.error(f"Chunk creation failed: {str(e)}")

    def query_data(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query P&L data with error handling"""
        try:
            query_embedding = self.model.encode(query)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []
        
    def save_extracted_tables(self, pdf_path: str, output_dir: str = 'extracted_tables'):
        """
        Extract tables from PDF and save each as a separate CSV
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to save CSV files
        """
        import os

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract tables
        tables = self.extract_tables(pdf_path)

        # Save each table to a CSV
        for i, table in enumerate(tables):
            # Preprocess the table
            processed_table = self.preprocess_table(table)
            
            # Save to CSV
            csv_path = os.path.join(output_dir, f'table_{i+1}.csv')
            processed_table.to_csv(csv_path, index=False)
            logger.info(f"Saved table {i+1} to {csv_path}")

        return len(tables)