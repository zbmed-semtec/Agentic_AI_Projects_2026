"""
Data Loader & In-Memory Indexer for Dissify Exploration Agent.
Parses merged_dissertations.xml, cv_extracted_timeline.json, and page-level OCR files.
Provides ultra-fast in-memory search for agent tools without re-parsing XML on each turn.
"""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import XML_PATH, CV_TIMELINE_PATH, TEXT_INDIVIDUAL_DIR, MAX_PAGE_CHARS

logger = logging.getLogger("dissify_agent.data_loader")


class DissifyDataManager:
    _instance = None

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.cv_timelines: Dict[str, Dict[str, Any]] = {}
        self.terminology_index: List[Dict[str, Any]] = []
        self._is_loaded = False

    @classmethod
    def get_instance(cls) -> "DissifyDataManager":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_all()
        return cls._instance

    def load_all(self):
        if self._is_loaded:
            return
        self._load_xml()
        self._load_cv_timelines()
        self._is_loaded = True
        logger.info(f"Loaded {len(self.documents)} dissertations and {len(self.terminology_index)} terminology pairs.")

    def _load_xml(self):
        if not XML_PATH.exists():
            logger.warning(f"XML file not found at {XML_PATH}")
            return

        logger.info(f"Parsing XML master file: {XML_PATH}")
        try:
            tree = ET.parse(str(XML_PATH))
            root = tree.getroot()

            for doc_elem in root.findall("document"):
                doc_id = doc_elem.get("id", "").strip()
                if not doc_id:
                    continue

                lang = doc_elem.get("lang", "de")
                meta_elem = doc_elem.find("metadata")
                title = ""
                author = ""
                year = ""
                city = ""

                if meta_elem is not None:
                    t_elem = meta_elem.find("title")
                    title = t_elem.text.strip() if t_elem is not None and t_elem.text else ""
                    a_elem = meta_elem.find("author/name")
                    author = a_elem.text.strip() if a_elem is not None and a_elem.text else ""
                    y_elem = meta_elem.find("year")
                    year = y_elem.text.strip() if y_elem is not None and y_elem.text else ""
                    c_elem = meta_elem.find("city")
                    city = c_elem.text.strip() if c_elem is not None and c_elem.text else ""

                # Subject Analysis
                domains = []
                keywords = []
                relevance_score = 0
                subj_elem = doc_elem.find("subject_analysis")
                if subj_elem is not None:
                    for d in subj_elem.findall("domains/domain"):
                        domains.append({
                            "id": d.get("id", ""),
                            "vocabulary": d.get("vocabulary", ""),
                            "translation": d.get("translation", ""),
                            "name": d.text.strip() if d.text else ""
                        })
                    for k in subj_elem.findall("subject_keywords/keyword"):
                        keywords.append({
                            "keyword": k.text.strip() if k.text else "",
                            "translation": k.get("translation", "")
                        })
                    rel_elem = subj_elem.find("relevance_assessment/relevance_score")
                    if rel_elem is not None and rel_elem.text:
                        try:
                            relevance_score = int(rel_elem.text.strip())
                        except ValueError:
                            relevance_score = 0

                # Synthesis & Terminology Notes
                summary_en = ""
                summary_de = ""
                broad_specialty = ""
                historical_context = ""
                doc_terms = []
                synth_elem = doc_elem.find("synthesis")
                if synth_elem is not None:
                    se_elem = synth_elem.find("summary_english")
                    summary_en = se_elem.text.strip() if se_elem is not None and se_elem.text else ""
                    sd_elem = synth_elem.find("summary_german")
                    summary_de = sd_elem.text.strip() if sd_elem is not None and sd_elem.text else ""
                    sp_elem = synth_elem.find("broad_specialty")
                    broad_specialty = sp_elem.text.strip() if sp_elem is not None and sp_elem.text else ""
                    hc_elem = synth_elem.find("historical_context")
                    historical_context = hc_elem.text.strip() if hc_elem is not None and hc_elem.text else ""

                    for t_elem in synth_elem.findall("terminology_notes/term_note"):
                        ht = t_elem.find("historical_term")
                        me = t_elem.find("modern_equivalent")
                        ex = t_elem.find("explanation")
                        h_term = ht.text.strip() if ht is not None and ht.text else ""
                        m_term = me.text.strip() if me is not None and me.text else ""
                        explanation = ex.text.strip() if ex is not None and ex.text else ""
                        if h_term or m_term:
                            note = {
                                "doc_id": doc_id,
                                "historical_term": h_term,
                                "modern_equivalent": m_term,
                                "explanation": explanation
                            }
                            doc_terms.append(note)
                            self.terminology_index.append(note)

                # Table of Contents
                sections = []
                toc_elem = doc_elem.find("table_of_contents/sections")
                if toc_elem is not None:
                    for s_elem in toc_elem.findall("section"):
                        s_title = s_elem.find("title")
                        title_text = s_title.text.strip() if s_title is not None and s_title.text else ""
                        p_elem = s_elem.find("pages")
                        pages_text = p_elem.text.strip() if p_elem is not None and p_elem.text else ""
                        sections.append({
                            "category": s_elem.get("category", "body"),
                            "title": title_text,
                            "start_page": s_elem.get("start_page", ""),
                            "end_page": s_elem.get("end_page", ""),
                            "pages": pages_text
                        })

                self.documents[doc_id] = {
                    "id": doc_id,
                    "lang": lang,
                    "title": title,
                    "author": author,
                    "year": year,
                    "city": city,
                    "domains": domains,
                    "keywords": keywords,
                    "relevance_score": relevance_score,
                    "summary_en": summary_en,
                    "summary_de": summary_de,
                    "broad_specialty": broad_specialty,
                    "historical_context": historical_context,
                    "terminology_notes": doc_terms,
                    "sections": sections
                }
        except Exception as e:
            logger.error(f"Error parsing {XML_PATH}: {e}")

    def _load_cv_timelines(self):
        if not CV_TIMELINE_PATH.exists():
            logger.warning(f"CV timeline not found at {CV_TIMELINE_PATH}")
            return
        try:
            with open(CV_TIMELINE_PATH, "r", encoding="utf-8") as f:
                self.cv_timelines = json.load(f)
            logger.info(f"Loaded {len(self.cv_timelines)} author biographical records.")
        except Exception as e:
            logger.error(f"Error reading {CV_TIMELINE_PATH}: {e}")

    # Tool helper methods
    def resolve_terminology(self, query_term: str) -> List[Dict[str, Any]]:
        """Finds historical or modern term matches."""
        q = query_term.lower().strip()
        matches = []
        seen = set()
        for item in self.terminology_index:
            h = item["historical_term"].lower()
            m = item["modern_equivalent"].lower()
            if q in h or q in m or h in q or any(word in h or word in m for word in q.split() if len(word) > 3):
                key = (item["historical_term"], item["modern_equivalent"])
                if key not in seen:
                    seen.add(key)
                    matches.append(item)
        return matches[:8]

    def search_catalog(self, query: str, domain: str = "", min_relevance: int = 0, limit: int = 5) -> List[Dict[str, Any]]:
        """Multi-field keyword search across titles, summaries, keywords, and domains."""
        tokens = [w.lower() for w in query.split() if len(w) > 2]
        domain_filter = domain.lower().strip()
        results = []

        for doc_id, doc in self.documents.items():
            if min_relevance and doc["relevance_score"] < min_relevance:
                continue

            # Domain check if specified
            if domain_filter:
                doc_domains = [d["name"].lower() for d in doc["domains"]] + [d["translation"].lower() for d in doc["domains"]]
                if not any(domain_filter in dom for dom in doc_domains):
                    continue

            # Calculate match score based on author, title, keywords, summaries
            author_lower = doc["author"].lower()
            title_lower = doc["title"].lower()
            searchable_text = (
                f"{doc['title']} {doc['author']} {doc['broad_specialty']} {doc['summary_en']} {doc['summary_de']} "
                f"{' '.join(k['keyword'] + ' ' + k['translation'] for k in doc['keywords'])} "
                f"{' '.join(d['name'] + ' ' + d['translation'] for d in doc['domains'])}"
            ).lower()

            score = 0
            for token in tokens:
                if token in author_lower:
                    score += 6
                if token in title_lower:
                    score += 5
                if token in searchable_text:
                    score += 2

            if score > 0 or not tokens:
                results.append((score, doc))

        results.sort(key=lambda x: (x[0], x[1]["relevance_score"]), reverse=True)
        return [r[1] for r in results[:limit]]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get(doc_id)

    def get_table_of_contents(self, doc_id: str) -> List[Dict[str, Any]]:
        doc = self.documents.get(doc_id)
        if not doc:
            return []
        return doc.get("sections", [])

    def read_page_text(self, doc_id: str, page_number: int) -> str:
        """Reads specific page file: {TEXT_INDIVIDUAL_DIR}/{doc_id}/{doc_id}_{page_number}.txt"""
        page_file = TEXT_INDIVIDUAL_DIR / doc_id / f"{doc_id}_{page_number}.txt"
        if not page_file.exists():
            return f"Page {page_number} for Document {doc_id} was not found on disk."

        try:
            with open(page_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > MAX_PAGE_CHARS:
                content = content[:MAX_PAGE_CHARS] + f"\n... [Truncated: showing first {MAX_PAGE_CHARS} characters of page {page_number}]"
            return content
        except Exception as e:
            return f"Error reading page file: {e}"

    def get_author_timeline(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.cv_timelines.get(doc_id)
