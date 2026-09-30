from langchain_core.tools import tool
from data_loader import DissifyDataManager

data_mgr = DissifyDataManager.get_instance()


@tool
def resolve_historical_terms(term: str) -> str:
    """
    Search for historical scientific terms and their modern equivalents.
    Useful when a modern query needs to be mapped to historical German terminology
    (e.g., 'lactation' -> 'Laktation / Lactogenesis', or disease names).
    
    Args:
        term: The concept or keyword to look up (English or German).
    """
    matches = data_mgr.resolve_terminology(term)
    if not matches:
        return f"No direct terminology mappings found for term: '{term}'. Proceed with standard keyword search."
    
    output = [f"Found {len(matches)} terminology notes for '{term}':"]
    for m in matches:
        output.append(
            f"- Historical: '{m['historical_term']}' <-> Modern: '{m['modern_equivalent']}'\n"
            f"  Explanation: {m['explanation']} (from Doc ID {m['doc_id']})"
        )
    return "\n".join(output)


@tool
def search_dissertation_catalog(query: str, domain: str = "") -> str:
    """
    Search the historical dissertations catalog by topic, keyword, or academic field.
    Returns candidate dissertations with document IDs, authors, titles, publication years,
    academic domains, and relevance scores.
    
    Args:
        query: Keywords, topic, or subject to search for.
        domain: Optional academic domain filter (e.g. 'Biochemistry', 'Physiology', 'Agriculture').
    """
    docs = data_mgr.search_catalog(query=query, domain=domain, limit=10)
    if not docs:
        return f"No dissertations matched the query '{query}' (domain: '{domain}')."
    
    output = [f"Found {len(docs)} matching dissertation(s):"]
    for d in docs:
        domains_str = ", ".join(f"{dom['name']} ({dom['translation']})" for dom in d["domains"][:3])
        output.append(
            f"\n• [Doc ID: {d['id']}] '{d['title']}'\n"
            f"  Author: {d['author']} | Year: {d['year']} | Relevance Score: {d['relevance_score']}/10\n"
            f"  Domains: {domains_str}\n"
            f"  Summary: {d['summary_en'][:220]}..."
        )
    return "\n".join(output)


@tool
def inspect_table_of_contents(doc_id: str) -> str:
    """
    Retrieve the structured Table of Contents for a specific dissertation ID.
    Shows all section titles (Inhalt, Einleitung, Versuchsergebnisse, etc.)
    and their start and end page numbers.
    Use this tool to find the exact pages to read for a specific question!
    
    Args:
        doc_id: The document ID (e.g., '10013639').
    """
    clean_id = doc_id.strip()
    sections = data_mgr.get_table_of_contents(clean_id)
    if not sections:
        return f"No Table of Contents found for Document ID '{clean_id}'."
    
    output = [f"Table of Contents for Document {clean_id}:"]
    for s in sections:
        output.append(
            f"- Section: '{s['title']}' | Category: {s['category']} | Pages: {s['pages']} (Start: {s['start_page']}, End: {s['end_page']})"
        )
    return "\n".join(output)


@tool
def read_page_ocr(doc_id: str, page_number: int) -> str:
    """
    Read the actual primary OCR text of a specific page from a dissertation.
    Use this tool AFTER inspecting the Table of Contents to read targeted pages.
    
    Args:
        doc_id: The dissertation ID (e.g., '10013639').
        page_number: The specific page number to read (integer, e.g., 81).
    """
    clean_id = doc_id.strip()
    return data_mgr.read_page_text(clean_id, int(page_number))


@tool
def inspect_author_timeline(doc_id: str) -> str:
    """
    Retrieve the author's biographical details, and timeline extracted from their dissertation CV.
    Provides birth date, places of education, degrees, military service, and career dates.
    Use when the query asks about the author's life, background, or vital dates.
    
    Args:
        doc_id: The dissertation ID (e.g., '10013639').
    """
    clean_id = doc_id.strip()
    timeline_data = data_mgr.get_author_timeline(clean_id)
    if not timeline_data:
        return f"No biographical timeline extracted for author of Document ID '{clean_id}'."
    
    name = timeline_data.get("primary_author_name", "Unknown")
    title = timeline_data.get("author_title", "")
    birth = timeline_data.get("birth", {})
    religion = timeline_data.get("religion", "")
    military = timeline_data.get("military_service_indicated", False)
    
    events = timeline_data.get("timeline", [])
    output = [
        f"Biographical Profile for {name} ({title}):",
        f"Birth: {birth.get('date', 'Unknown')} at {birth.get('place', 'Unknown')}",
        f"Religion: {religion} | Military Service: {'Yes' if military else 'No'}",
        "\nTimeline Milestones:"
    ]
    for ev in events[:6]:
        output.append(f"- [{ev.get('date', '')}] {ev.get('event_type', '').title()}: {ev.get('description', '')} ({ev.get('place', '')})")
    
    return "\n".join(output)
