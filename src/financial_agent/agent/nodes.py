from typing import Dict, Any
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from ..tools.pdf_parser import PDFParser
from ..tools.calculators import CALCULATION_TOOLS
from ..utils.llm import get_ollama_llm
from .state import AgentState, CompanyMetrics
import json
import re


def parse_pdf_node(state: AgentState) -> Dict[str, Any]:
    """Node to parse PDF and extract text"""
    pdf_index = state.get("current_pdf_index", 0)
    pdf_paths = state.get("pdf_paths", [])
    
    if pdf_index >= len(pdf_paths):
        return {"error": "No more PDFs to process"}
    
    current_pdf = pdf_paths[pdf_index]
    
    try:
        parser = PDFParser()
        extracted_text = parser.extract_text(current_pdf)
        
        print(f"\n{'='*60}")
        print(f"Processing PDF {pdf_index + 1}/{len(pdf_paths)}: {current_pdf}")
        print(f"Extracted {len(extracted_text)} characters")
        print(f"{'='*60}\n")
        
        return {
            "current_pdf": current_pdf,
            "extracted_text": extracted_text,
            "messages": [HumanMessage(content=f"Extracted text from {current_pdf}")]
        }
    except Exception as e:
        return {"error": f"Failed to parse {current_pdf}: {str(e)}"}


def extract_metrics_node(state: AgentState) -> Dict[str, Any]:
    """Node to extract financial metrics using LLM"""
    
    llm = get_ollama_llm()
    
    # Truncate text to avoid token limits
    text_sample = state["extracted_text"][:10000]
    
    prompt = f"""Extract financial data from the earnings report below and return ONLY a JSON object.

IMPORTANT RULES:
- Output ONLY the JSON object, no explanations
- No markdown formatting, no ```json```, no extra text
- Use exact field names shown in the example
- All values in millions

Example output format:
{{"company_name": "Apple", "revenue": 102466, "net_income": 14736, "operating_income": 21496, "shares_outstanding": 15204, "previous_revenue": 94930}}

Earnings report:
{text_sample}

JSON output:"""
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        print(f"\nLLM Response:\n{response_text[:500]}...\n")
        
        # Try multiple extraction methods
        raw_metrics = None
        
        # Method 1: Look for JSON in code blocks
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_block_match:
            raw_metrics = json.loads(json_block_match.group(1))
        
        # Method 2: Look for standalone JSON object
        if not raw_metrics:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    raw_metrics = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        # Method 3: Fallback - extract numbers from text
        if not raw_metrics:
            print("JSON parsing failed, extracting values from text...")
            raw_metrics = extract_values_from_text(response_text, text_sample)
        
        # Ensure company_name exists
        if "company_name" not in raw_metrics or not raw_metrics["company_name"]:
            pdf_name = Path(state["current_pdf"]).stem
            raw_metrics["company_name"] = pdf_name.split('_')[0].title()
        
        company_name = raw_metrics["company_name"]
        print(f"✓ Successfully extracted metrics for: {company_name}")
        print(f"  Revenue: {raw_metrics.get('revenue')}")
        print(f"  Net Income: {raw_metrics.get('net_income')}")
        print(f"  Operating Income: {raw_metrics.get('operating_income')}")
        
        return {
            "raw_metrics": raw_metrics,
            "current_company": company_name,
            "messages": [AIMessage(content=f"Extracted metrics for {company_name}")]
        }
        
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        company_name = Path(state["current_pdf"]).stem.split('_')[0].title()
        return {
            "raw_metrics": {"company_name": company_name},
            "current_company": company_name,
            "messages": [AIMessage(content=f"Failed extraction for {company_name}")]
        }


def extract_values_from_text(response_text: str, original_text: str) -> Dict:
    """Fallback: Extract financial values from LLM's text response"""
    metrics = {}
    
    # Extract company name from original text (look for common patterns)
    company_patterns = [
        r"^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+Inc",
        r"^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+Corp",
        r"FORM 10-Q.*?([A-Z][a-zA-Z]+\s+Inc\.)",
    ]
    for pattern in company_patterns:
        match = re.search(pattern, original_text, re.MULTILINE)
        if match:
            metrics["company_name"] = match.group(1)
            break
    
    # Extract revenue (look for various patterns)
    revenue_patterns = [
        r"[Tt]otal [Nn]et [Ss]ales[:\s]+\$?([\d,]+)",
        r"[Rr]evenue[:\s]+\$?([\d,]+)",
        r"[Nn]et [Ss]ales[:\s]+\$?([\d,]+)",
    ]
    for pattern in revenue_patterns:
        match = re.search(pattern, response_text)
        if match:
            metrics["revenue"] = float(match.group(1).replace(',', ''))
            break
    
    # Extract net income
    net_income_patterns = [
        r"[Nn]et [Ii]ncome[:\s]+\$?([\d,]+)",
        r"[Nn]et [Ee]arnings[:\s]+\$?([\d,]+)",
    ]
    for pattern in net_income_patterns:
        match = re.search(pattern, response_text)
        if match:
            metrics["net_income"] = float(match.group(1).replace(',', ''))
            break
    
    # Extract operating income
    operating_patterns = [
        r"[Oo]perating [Ii]ncome[:\s]+\$?([\d,]+)",
        r"\\boxed\{([\d,]+)\}",  # From your LLM's math notation
    ]
    for pattern in operating_patterns:
        match = re.search(pattern, response_text)
        if match:
            metrics["operating_income"] = float(match.group(1).replace(',', ''))
            break
    
    # Extract previous revenue
    prev_revenue_patterns = [
        r"[Pp]revious.*?[Rr]evenue[:\s]+\$?([\d,]+)",
        r"prior year.*?\$?([\d,]+)",
    ]
    for pattern in prev_revenue_patterns:
        match = re.search(pattern, response_text)
        if match:
            metrics["previous_revenue"] = float(match.group(1).replace(',', ''))
            break
    
    # Extract shares outstanding
    shares_patterns = [
        r"[Ss]hares [Oo]utstanding[:\s]+([\d,]+)",
        r"[Ww]eighted.*?[Ss]hares[:\s]+([\d,]+)",
    ]
    for pattern in shares_patterns:
        match = re.search(pattern, response_text)
        if match:
            metrics["shares_outstanding"] = float(match.group(1).replace(',', ''))
            break
    
    return metrics


def calculate_metrics_node(state: AgentState) -> Dict[str, Any]:
    """Node to calculate all financial metrics"""
    
    raw = state.get("raw_metrics", {})
    
    if not raw:
        print("Warning: No raw metrics found, skipping calculations")
        return {}
    
    # Extract raw values
    revenue = raw.get("revenue")
    net_income = raw.get("net_income")
    operating_income = raw.get("operating_income")
    shares_outstanding = raw.get("shares_outstanding")
    previous_revenue = raw.get("previous_revenue")
    
    # Calculate metrics
    profit_margin = None
    if revenue and net_income and revenue != 0:
        profit_margin = (net_income / revenue) * 100
    
    operating_margin = None
    if revenue and operating_income and revenue != 0:
        operating_margin = (operating_income / revenue) * 100
    
    eps = None
    if net_income and shares_outstanding and shares_outstanding != 0:
        eps = net_income / shares_outstanding
    
    revenue_growth = None
    if revenue and previous_revenue and previous_revenue != 0:
        revenue_growth = ((revenue - previous_revenue) / previous_revenue) * 100
    
    # Create structured metrics
    metrics: CompanyMetrics = {
        "company_name": state.get("current_company", "Unknown"),
        "revenue": revenue,
        "net_income": net_income,
        "eps": round(eps, 2) if eps else None,
        "operating_margin": round(operating_margin, 2) if operating_margin else None,
        "profit_margin": round(profit_margin, 2) if profit_margin else None,
        "revenue_growth": round(revenue_growth, 2) if revenue_growth else None,
    }
    
    print(f"\nCalculated metrics for {metrics['company_name']}:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()
    
    # Append to company_metrics list
    company_metrics = state.get("company_metrics", []).copy()
    company_metrics.append(metrics)
    
    # Increment PDF index for next iteration
    current_index = state.get("current_pdf_index", 0)
    
    return {
        "company_metrics": company_metrics,
        "current_pdf_index": current_index + 1,
        "messages": [AIMessage(content=f"Calculated metrics for {metrics['company_name']}")]
    }


def should_process_more_pdfs(state: AgentState) -> str:
    """Conditional edge: check if more PDFs to process"""
    current_index = state.get("current_pdf_index", 0)
    total_pdfs = len(state.get("pdf_paths", []))
    
    print(f"\nProgress: Processed {current_index}/{total_pdfs} PDFs")
    
    if current_index < total_pdfs:
        return "parse_pdf"
    else:
        return "generate_comparison"


def generate_comparison_node(state: AgentState) -> Dict[str, Any]:
    """Node to generate CSV comparison"""
    import pandas as pd
    from datetime import datetime
    from pathlib import Path
    
    metrics_list = state.get("company_metrics", [])
    
    if not metrics_list:
        return {"error": "No metrics to compare"}
    
    print(f"\nGenerating comparison for {len(metrics_list)} companies...")
    
    df = pd.DataFrame(metrics_list)
    
    # Ensure output directory exists
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"comparison_{timestamp}.csv"
    
    df.to_csv(output_path, index=False)
    
    print(f"\nComparison CSV saved to: {output_path}")
    print(f"\nPreview:\n{df.to_string()}\n")
    
    return {
        "comparison_csv_path": str(output_path),
        "messages": [AIMessage(content=f"Generated comparison at {output_path}")]
    }