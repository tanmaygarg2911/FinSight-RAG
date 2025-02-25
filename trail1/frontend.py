import streamlit as st
import pandas as pd
from backend import RobustPLExtractor, ExtractionConfig, PDFValidator
import tempfile
import os

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and return path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def display_pl_components(components):
    """Display P&L components in an organized manner"""
    for section, data in components.items():
        st.subheader(f"{section.title()} Section")
        st.dataframe(data)
        st.markdown("---")

def main():
    st.title("PDF P&L Statement Analyzer")
    st.markdown("""
    Upload a PDF containing Profit & Loss statements to analyze and query the financial data.
    """)

    # Initialize session state
    if 'extractor' not in st.session_state:
        st.session_state.extractor = None
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False

    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

    if uploaded_file is not None:
        with st.spinner('Processing PDF...'):
            try:
                # Save uploaded file
                pdf_path = save_uploaded_file(uploaded_file)
                
                if pdf_path:
                    # Validate PDF
                    if not PDFValidator.validate_pdf(pdf_path):
                        st.error("Invalid PDF or no P&L information found.")
                        return

                    # Initialize extractor
                    st.session_state.extractor = RobustPLExtractor()
                    
                    # Extract tables
                    tables = st.session_state.extractor.extract_tables(pdf_path)
                    
                    if not tables:
                        st.error("No valid P&L tables found in the PDF.")
                        return
                    
                    # Process each table
                    for i, table in enumerate(tables):
                        st.subheader(f"Table {i+1}")
                        
                        # Preprocess table
                        processed_table = st.session_state.extractor.preprocess_table(table)
                        
                        # Extract components
                        components = st.session_state.extractor.extract_pl_components(processed_table)
                        
                        # Create searchable chunks
                        st.session_state.extractor.create_searchable_chunks(components)
                        
                        # Display components
                        display_pl_components(components)
                    
                    st.session_state.file_processed = True
                    
                    # Clean up temp file
                    os.unlink(pdf_path)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state.file_processed = False

    # Query interface
    if st.session_state.file_processed:
        st.markdown("---")
        st.subheader("Query P&L Data")
        query = st.text_input("Enter your query (e.g., 'What is the total revenue?')")
        n_results = st.slider("Number of results to display", min_value=1, max_value=10, value=3)
        
        if query and st.button("Search"):
            with st.spinner('Searching...'):
                results = st.session_state.extractor.query_data(query, n_results)
                
                if results:
                    st.subheader("Search Results")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} - {result['metadata']['section'].title()}"):
                            st.write(f"**Line Item:** {result['metadata']['line_item']}")
                            st.write(f"**Details:** {result['content']}")
                else:
                    st.info("No results found for your query.")

if __name__ == "__main__":
    main()