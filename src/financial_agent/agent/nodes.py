from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from ..tools.pdf_parser import PDFParser
from ..tools.calculators import CALCULATION_TOOLS
from ..utils.llm import get_llm_with_tools
from .state import AgentState, CompanyMetrics


def parse_pdf_node(state: AgentState) -> Dict[str, Any]:
    """Node to parse pdf and extract text"""
    current_pdf = state["pdf_paths"][0] if state["pdf_paths"] else None
    
    if not current_pdf:
        return {"error": "No PDF provided"}
    
    try:
        parser = PDFParser()
        extracted_text = parser.extract_text(current_pdf)
        
        return {
            "current_pdf": current_pdf,
            "extracted_text": extracted_text,
            "messages": [HumanMessage(content=f"Extracted text from {current_pdf}")]
        }
    except Exception as e:
        return {"error": str(e)}


def extract_metrics_node(state: AgentState) -> Dict[str, Any]:
    """Node to extract financial metrics using llm"""

    llm = get_llm_with_tools(CALCULATION_TOOLS)

    SYSTEM_PROMPT = f"""
    You are a financial analyst. Analyse the following earnings reports.

    Extract the following info:
    - Company name
    - Revenue (total revenue or sales)
    - Net income/net profit
    - Operating income
    - Shares outstanding
    - Previous period revenue

    Text from earnings report:
    {state["extracted_text"][:8000]}

    After extracting values, use the available tools to calculate:
    - Profit margin

    Provide results in structured format
    """

    response = llm.invoke([HumanMessage(content=SYSTEM_PROMPT)])

    return {
        "messages": [response],
        "current_company": "ExtractedCompany"
    }


def calculate_metrics_node(state: AgentState) -> Dict[str, Any]:
    """Node to calculate all financial metrics"""

    metrics: CompanyMetrics = {
        "company_name": state.get("current_company", "Unknown"),
        "revenue": None,
        "net_income": None,
        "profit_margin": None,
    }

    company_metrics = state.get("company_metrics", [])
    company_metrics.append(metrics)

    return {"company_metrics": company_metrics}


def should_process_more_pdfs(state: AgentState) -> str:
    """Conditional edge: check if more PDFs to process"""
    processed = len(state.get("company_metrics", []))
    total = len(state.get("pdf_paths", []))
    
    if processed < total:
        return "parse_pdf"
    else:
        return "generate_comparison"
    
def generate_comparison_node(state: AgentState) -> Dict[str, Any]:
    """Node to generate CSV comparison"""
    import pandas as pd
    from datetime import datetime
    
    metrics_list = state.get("company_metrics", [])
    
    if not metrics_list:
        return {"error": "No metrics to compare"}
    
    df = pd.DataFrame(metrics_list)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/outputs/comparison_{timestamp}.csv"
    
    df.to_csv(output_path, index=False)
    
    return {
        "comparison_csv_path": output_path,
        "messages": [AIMessage(content=f"Generated comparison at {output_path}")]
    }