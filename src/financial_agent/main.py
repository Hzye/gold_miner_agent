from pathlib import Path
from .agent.graph import create_financial_agent
from .agent.graph import AgentState
from typing import List


def analyse_reports(pdf_paths: List[str]) -> str:
    """
    Analyse financial reports and generate comparison csv
    
    Args:
        pdf_paths: List of paths to earnings report pdfs
        
    Returns:
        path to generated comparison csv
    """

    # create agent
    agent = create_financial_agent()

    # initialise state
    initial_state: AgentState = {
        "messages": [],
        "pdf_paths": pdf_paths,
        "current_pdf": None,
        "extracted_text": None,
        "current_company": None,
        "company_metrics": [],
        "comparison_csv_path": None,
        "error": None,
    }

    # run agent
    result = agent.invoke(initial_state)

    if result.get("error"):
        raise Exception(f"Agent error: {result['error']}")
    
    return result.get("comparison_csv_path")

if __name__ == "__main__":
    pdf_files = [
        "data/inputs/apple_q4_2025.pdf"
    ]

    output_csv = analyse_reports(pdf_files)
    print(f"Comparison generated: {output_csv}")