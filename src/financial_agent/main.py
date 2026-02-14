from pathlib import Path
from typing import List
from .agent.graph import create_financial_agent
from .agent.state import AgentState


def analyse_reports(pdf_paths: List[str]) -> str:
    """
    Analyze financial reports and generate comparison CSV
    
    Args:
        pdf_paths: List of paths to earnings report PDFs
    
    Returns:
        Path to generated comparison CSV
    """
    
    # Validate PDF paths
    for pdf_path in pdf_paths:
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    print(f"\n{'='*60}")
    print(f"Starting analysis of {len(pdf_paths)} reports")
    print(f"{'='*60}\n")
    
    # Create agent
    agent = create_financial_agent()
    
    # Initialize state
    initial_state: AgentState = {
        "messages": [],
        "pdf_paths": pdf_paths,
        "current_pdf_index": 0,  # Start at first PDF
        "current_pdf": None,
        "extracted_text": None,
        "current_company": None,
        "raw_metrics": None,
        "company_metrics": [],
        "comparison_csv_path": None,
        "error": None,
    }
    
    # Run agent
    result = agent.invoke(initial_state)
    
    if result.get("error"):
        raise Exception(f"Agent error: {result['error']}")
    
    csv_path = result.get("comparison_csv_path")
    if not csv_path:
        raise Exception("No CSV generated")
    
    return csv_path


if __name__ == "__main__":
    # Example usage
    pdf_files = [
        "data/inputs/apple_q4_2025.pdf",
        "data/inputs/google_q4_2025.pdf",
    ]
    
    try:
        output_csv = analyse_reports(pdf_files)
        print(f"\n✅ Success! Comparison generated: {output_csv}")
    except Exception as e:
        print(f"\n❌ Error: {e}")