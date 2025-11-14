#!/usr/bin/env python
# coding: utf-8

# """
# AI-based Parser Agent
# This file converts your existing Text_Parsing notebook into a AI agent that:
# - Reads an Excel file and extracts only the columns: LogoURL, BannerImage, Mod2, Mod3
# - Parses a .eml file (email) and extracts HTML attributes (src, href, background), image links and anchor links
# - Normalizes URLs deterministically
# - Compares Excel URLs with EML content deterministically to create a DataFrame of matches
# - Optionally uses an OpenAI LLM (via AI's LLM wrapper) for normalization edge-cases and explanations
# 
# 
# Usage (high level):
# - Populate .env with OPENAI_API_KEY if you want LLM capabilities
# - Run this script or import as a module. The bottom section demonstrates running the agent.
# 
# 
# Note: Comments in the cells explain the code in detail.
# """

# In[65]:


import os
import re
import time
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse, urlunparse, urljoin, parse_qs


import pandas as pd
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup


# AI imports (agent orchestration)
from ai import Agent, Task, , Process, LLM
from ai.tools import tool


# dotenv for secrets
from dotenv import load_dotenv


# In[66]:


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# In[118]:


# -----------------------
# Utility functions (deterministic)
# -----------------------

def deterministic_normalize_url(raw: str, base: str = None) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.startswith("data:"):
        return s
    if base and not bool(urlparse(s).netloc):
        s = urljoin(base, s)
    try:
        p = urlparse(s)
    except Exception:
        return s
    query = parse_qs(p.query, keep_blank_values=True)
    filtered_q = {k: v for k, v in query.items() if not re.match(r"^(utm_|gclid|fbclid)", k, flags=re.I)}
    qs_parts = []
    for k in sorted(filtered_q.keys()):
        vals = filtered_q[k]
        if not vals:
            qs_parts.append(k)
        else:
            qs_parts.append("{}={}".format(k, ",".join(vals)))
    new_query = "&".join(qs_parts)
    normalized = urlunparse((p.scheme, p.netloc.lower(), p.path or '/', p.params, new_query, ""))
    return normalized


# In[120]:


# Tools: wrap the core functions so AI Tasks can call them.
# Each tool is deterministic, except where we explicitly call the LLM via a separate tool.

# from typing import List
# import pandas as pd
# from ai.tools import tool

@tool
def read_excel_tool(xlsx_path: str, main_entity_col: str = 'MainEntityName') -> List[Dict[str, Any]]:
    """
    Read Excel and return list of dicts for rows where MainEntityName == 'All Other' (case-insensitive).
    Ensures keys: LogoURL, BannerImage, Mod2, Mod3, MainEntityName are present in each dict.
    """
    expected_cols = ["LogoURL", "BannerImage", "Mod2", "Mod3", main_entity_col]
    df = pd.read_excel(xlsx_path, dtype=str)

    # Map expected canonical columns to actual headers (case-insensitive, contains fallback)
    found_map = {}
    for expected in expected_cols:
        exact = [c for c in df.columns if c.strip().lower() == expected.strip().lower()]
        if exact:
            found_map[expected] = exact[0]
        else:
            contains = [c for c in df.columns if expected.strip().lower() in c.strip().lower()]
            found_map[expected] = contains[0] if contains else None

    # Build canonical DataFrame with all expected keys
    canon_df = pd.DataFrame()
    for k in expected_cols:
        if found_map.get(k):
            canon_df[k] = df[found_map[k]].astype(str).where(~df[found_map[k]].isna(), None)
        else:
            canon_df[k] = [None] * len(df)

    # Filter rows where MainEntityName == 'All Other' (case-insensitive trim)
    def is_all_other(x):
        if x is None:
            return False
        return str(x).strip().lower() == 'all other'

    mask = canon_df[main_entity_col].apply(is_all_other)
    filtered = canon_df.loc[mask].reset_index(drop=True)

    # Return list of dicts; each dict contains LogoURL,BannerImage,Mod2,Mod3,MainEntityName
    return filtered.to_dict(orient='records')


# In[124]:


@tool
def parse_eml_tool(eml_path: str) -> Dict[str, Any]:
    """
    Parse a .eml file and return structured artifacts useful for matching.
    Returns a dict with keys:
      - html: (str) the HTML body or empty string
      - images: (List[str]) src attributes from <img>
      - anchors: (List[str]) href values from <a>
      - raw_attrs: (List[Tuple[tag, attr_name, attr_value]]) for background/style/src/href
      - text_urls: (List[str]) absolute URLs found in plain text nodes

    The function reads the .eml file from `eml_path` and parses the HTML part using BeautifulSoup.
    """
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    html = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/html':
                html = part.get_content()
                break
    else:
        if msg.get_content_type() == 'text/html':
            html = msg.get_content()

    soup = BeautifulSoup(html or "", 'html.parser')
    images = []
    anchors = []
    raw_attrs = []

    # Extract <img src>
    for img in soup.find_all('img'):
        src = img.get('src')
        if src:
            images.append(src)
            raw_attrs.append(('img', 'src', src))

    # Extract <a href>
    for a in soup.find_all('a'):
        href = a.get('href')
        if href:
            anchors.append(href)
            raw_attrs.append(('a', 'href', href))

    # background attributes and url(...) in style attributes
    for tag in soup.find_all(True):
        bg = tag.get('background')
        if bg:
            raw_attrs.append((tag.name, 'background', bg))
        style = tag.get('style')
        if style and 'url(' in style:
            urls = re.findall(r'url\\(([^)]+)\\)', style)
            for u in urls:
                raw_attrs.append((tag.name, 'style:url', u.strip("\"'")))

    text = soup.get_text(separator=' ')
    text_urls = [m.group(0) for m in re.finditer(r'https?://[^\\s\"<>\\)]+', text)]

    return {
        'html': html or "",
        'images': images,
        'anchors': anchors,
        'raw_attrs': raw_attrs,
        'text_urls': text_urls
    }


# In[125]:


@tool
def compare_tool(excel_rows: List[Dict[str, Any]], eml_struct: Dict[str, Any], base_url: str = None) -> List[Dict[str, Any]]:
    """
    For each filtered excel row, evaluate each of the four columns (LogoURL,BannerImage,Mod2,Mod3).
    Emit one output record per non-empty column for the selected rows. Each record contains row_index
    (index within the filtered list), source_column, source_value, normalized_value, matched (bool),
    match_symbol, match_locations, comparison_method.
    """
    priority_cols = ["LogoURL","BannerImage","Mod2","Mod3"]

    # Build eml candidate normalization map
    eml_candidates = []
    for tag, attr, val in eml_struct.get('raw_attrs', []):
        eml_candidates.append((val,f"eml:{tag}[{attr}]") )
    for v in eml_struct.get('images',[]):
        eml_candidates.append((v,'eml:img[src]'))
    for v in eml_struct.get('anchors',[]):
        eml_candidates.append((v,'eml:a[href]'))
    for v in eml_struct.get('text_urls',[]):
        eml_candidates.append((v,'eml:text'))

    eml_norm_map = {}
    for raw_val, loc in eml_candidates:
        norm = deterministic_normalize_url(raw_val, base=base_url)
        eml_norm_map.setdefault(norm,set()).add(loc)
        eml_norm_map.setdefault(raw_val,set()).add(loc)

    records = []
    for row_idx, row in enumerate(excel_rows):
        for col in priority_cols:
            raw_val = row.get(col)
            if raw_val is None or str(raw_val).strip() == '':
                continue
            val_str = str(raw_val).strip()
            norm_val = deterministic_normalize_url(val_str, base=base_url)

            matched = False
            match_locations = []
            comparison_method = None

            if val_str in eml_norm_map:
                matched = True
                match_locations = list(eml_norm_map[val_str])
                comparison_method = 'exact'
            elif norm_val in eml_norm_map:
                matched = True
                match_locations = list(eml_norm_map[norm_val])
                comparison_method = 'normalized'
            else:
                for candidate, locs in eml_norm_map.items():
                    if candidate and val_str in candidate:
                        matched = True
                        match_locations = list(locs)
                        comparison_method = 'substring'
                        break

            records.append({
                'row_index': row_idx,
                'source_column': col,
                'source_value': val_str,
                'normalized_value': norm_val,
                'matched': bool(matched),
                'match_symbol': '✔️' if matched else '❌',
                'match_locations': match_locations,
                'comparison_method': comparison_method or 'none'
            })
    return records



# In[127]:


@tool
def llm_review_tool(ambiguous_items: List[Dict[str, Any]], llm_model: str = 'gpt-4o', max_items_per_call: int = 10) -> List[Dict[str, Any]]:
    """
    Inspect ambiguous url-like items with an LLM and return suggestions/explanations.

    Input:
      - ambiguous_items: list of dicts with keys: source_column, source_value, normalized_value (optional)
      - llm_model: model identifier used by AI LLM wrapper
      - max_items_per_call: batch size for LLM prompting

    Output:
      - returns a list of dicts (one per input item) augmented with:
          - llm_suggestion: (str|None) cleaned/normalized URL suggestion or None
          - llm_explanation: (str|None) short explanation or raw LLM text if parsing failed

    IMPORTANT: This tool DOES NOT change deterministic matched flags. It only provides suggestions for human review.
    """
    import json
    results: List[Dict[str, Any]] = []

    # instantiate client using AI wrapper (reads OPENAI_API_KEY from env)
    llm_client = LLM(model=llm_model, api_key=OPENAI_API_KEY)

    def _safe_call(prompt_text: str):
        """
        Try a couple of common invocation styles for the LLM wrapper.
        Return the raw response object or raise RuntimeError if none work.
        """
        # try common call styles defensively
        try:
            # common: call(prompt=...)
            return llm_client.call(prompt=prompt_text)
        except TypeError:
            pass
        except Exception as e:
            # network/auth errors should surface
            raise

        try:
            # some wrappers expect input as dict
            return llm_client.call({"input": prompt_text})
        except Exception:
            pass

        try:
            # some wrappers accept a single positional string
            return llm_client.call(prompt_text)
        except Exception:
            pass

        # if none worked, surface a helpful error so you can inspect dir(llm_client)
        raise RuntimeError(f"LLM client doesn't accept known call signatures. Inspect dir(llm_client).")

    def _extract_text(resp: Any) -> str:
        """Best-effort extraction of meaningful text from common response shapes."""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            # common keys
            for k in ("text", "output_text", "content"):
                if k in resp and isinstance(resp[k], str):
                    return resp[k]
            # openai-like choices
            if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                first = resp["choices"][0]
                if isinstance(first, dict):
                    if "message" in first and isinstance(first["message"], dict):
                        return first["message"].get("content") or first["message"].get("text") or str(first)
                    if "text" in first:
                        return first["text"]
                return str(first)
            if "generations" in resp and isinstance(resp["generations"], list) and resp["generations"]:
                gen = resp["generations"][0]
                if isinstance(gen, dict):
                    return gen.get("text") or gen.get("output_text") or str(gen)
                return str(gen)
            return str(resp)
        # fallback to attribute access
        if hasattr(resp, "text"):
            try:
                return resp.text
            except Exception:
                pass
        if hasattr(resp, "content"):
            try:
                return resp.content
            except Exception:
                pass
        return str(resp)

    # Batch items to reduce token usage
    for i in range(0, len(ambiguous_items), max_items_per_call):
        batch = ambiguous_items[i:i + max_items_per_call]

        prompt_lines = [
            "You are a careful assistant. For each item below (a possibly-messy URL-like string),",
            "return a JSON array where each element is an object with keys:",
            "  - llm_suggestion: cleaned URL or empty string if none",
            "  - explanation: 1-2 short sentences explaining why this may or may not match.",
            "Respond ONLY with valid JSON (an array).",
            "Items:"
        ]
        for it in batch:
            prompt_lines.append(f"- column: {it.get('source_column','')} | value: {it.get('source_value','')}")
        prompt = "\n".join(prompt_lines)

        # call the LLM defensively
        try:
            raw = _safe_call(prompt)
        except Exception as err:
            # If the LLM can't be called, attach the error message as explanation and continue
            for it in batch:
                out = it.copy()
                out["llm_suggestion"] = None
                out["llm_explanation"] = f"LLM call failed: {err}"
                results.append(out)
            continue

        text = _extract_text(raw)

        # parse JSON output
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                parsed = [parsed]
            for item, suggestion in zip(batch, parsed):
                out = item.copy()
                if isinstance(suggestion, dict):
                    out["llm_suggestion"] = suggestion.get("llm_suggestion")
                    out["llm_explanation"] = suggestion.get("explanation")
                else:
                    out["llm_suggestion"] = None
                    out["llm_explanation"] = str(suggestion)
                results.append(out)
        except Exception:
            # fallback: store raw text as explanation
            for it in batch:
                out = it.copy()
                out["llm_suggestion"] = None
                out["llm_explanation"] = text[:1000]
                results.append(out)

    return results


# In[129]:


@tool("llm_review_tool")
def llm_review_tool(ambiguous_items: List[Dict[str, Any]], llm_model: str = 'gpt-4o', max_items_per_call: int = 10) -> List[Dict[str, Any]]:
    """
    Use an LLM to produce conservative suggestions for ambiguous URL-like items.

    Inputs:
      - ambiguous_items: list of dicts with keys: source_column, source_value, normalized_value (optional)
      - llm_model: model id used by AI's LLM wrapper
      - max_items_per_call: number of items to batch per LLM call

    Output:
      - List[Dict]: each input dict augmented with:
          - llm_suggestion: (str|None) suggested normalized URL
          - llm_explanation: (str|None) brief explanation or raw LLM text
    Note: This tool does NOT change deterministic matched flags; it only returns suggestions/explanations.
    """
    import json
    results: List[Dict[str, Any]] = []

    # instantiate LLM client (reads OPENAI_API_KEY from env)
    llm_client = LLM(model=llm_model, api_key=OPENAI_API_KEY)

    def _safe_call(prompt_text: str):
        """
        Try a few common invocation styles for the AI LLM client and return the raw response.
        Raises RuntimeError if no known call signature works.
        """
        # 1) call(prompt=...)
        try:
            return llm_client.call(prompt=prompt_text)
        except TypeError:
            pass
        except Exception:
            raise

        # 2) call({'input': prompt}) or call(prompt_text) positional
        try:
            return llm_client.call({"input": prompt_text})
        except Exception:
            pass
        try:
            return llm_client.call(prompt_text)
        except Exception:
            pass

        # 3) give a helpful error so you can inspect dir(llm_client)
        raise RuntimeError("LLM client did not accept known call signatures. Inspect dir(llm_client).")

    def _extract_text(resp: Any) -> str:
        """Best-effort extraction of textual output from common response shapes."""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            for k in ("text", "output_text", "content"):
                if k in resp and isinstance(resp[k], str):
                    return resp[k]
            if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                first = resp["choices"][0]
                if isinstance(first, dict):
                    if "message" in first and isinstance(first["message"], dict):
                        return first["message"].get("content") or first["message"].get("text") or str(first)
                    if "text" in first:
                        return first["text"]
                return str(first)
            if "generations" in resp and isinstance(resp["generations"], list) and resp["generations"]:
                gen = resp["generations"][0]
                if isinstance(gen, dict):
                    return gen.get("text") or gen.get("output_text") or str(gen)
                return str(gen)
            return str(resp)
        if hasattr(resp, "text"):
            try:
                return resp.text
            except Exception:
                pass
        if hasattr(resp, "content"):
            try:
                return resp.content
            except Exception:
                pass
        return str(resp)

    # Batch ambiguous items
    for i in range(0, len(ambiguous_items), max_items_per_call):
        batch = ambiguous_items[i:i + max_items_per_call]
        prompt_lines = [
            "You are a careful assistant. For each item below (a possibly-messy URL-like string),",
            "return a JSON array where each element is an object with keys:",
            "  - llm_suggestion: cleaned URL or empty string if none",
            "  - explanation: 1-2 short sentences explaining why this may or may not match.",
            "Respond ONLY with valid JSON (an array).",
            "Items:"
        ]
        for it in batch:
            prompt_lines.append(f"- column: {it.get('source_column','')} | value: {it.get('source_value','')}")
        prompt = "\n".join(prompt_lines)

        # call LLM defensively
        try:
            raw = _safe_call(prompt)
        except Exception as err:
            # If LLM isn't callable, attach error as explanation to each item and continue
            for it in batch:
                out = it.copy()
                out["llm_suggestion"] = None
                out["llm_explanation"] = f"LLM call failed: {err}"
                results.append(out)
            continue

        text = _extract_text(raw)

        # parse JSON from LLM
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                parsed = [parsed]
            for item, suggestion in zip(batch, parsed):
                out = item.copy()
                if isinstance(suggestion, dict):
                    out["llm_suggestion"] = suggestion.get("llm_suggestion")
                    out["llm_explanation"] = suggestion.get("explanation")
                else:
                    out["llm_suggestion"] = None
                    out["llm_explanation"] = str(suggestion)
                results.append(out)
        except Exception:
            # fallback: store raw text as explanation
            for it in batch:
                out = it.copy()
                out["llm_suggestion"] = None
                out["llm_explanation"] = text[:1000]
                results.append(out)

    return results



# In[130]:


# ParserAgent: orchestrates reading, parsing, comparing, and optional LLM review
class ParserAgent:
    def __init__(self, llm_enabled: bool = True, llm_model: str = 'gpt-4o'):
        self.llm_enabled = llm_enabled and bool(OPENAI_API_KEY)
        self.llm_model = llm_model
        self.priority_cols = ["LogoURL","BannerImage","Mod2","Mod3"]

    def run(self, xlsx_path: str, eml_path: str, base_url: str = None) -> pd.DataFrame:
        # 1) Read filtered excel rows (only rows where MainEntityName == 'All Other')
        excel_rows = read_excel_tool.func(xlsx_path)
        if not excel_rows:
            # return empty DataFrame with expected columns
            cols = ['row_index','source_column','source_value','normalized_value','matched','match_symbol','match_locations','comparison_method','llm_suggestion','llm_explanation']
            return pd.DataFrame(columns=cols)

        # 2) Parse EML
        eml_struct = parse_eml_tool.func(eml_path)

        # 3) Compare: evaluate each of the 4 columns and get records for every non-empty column
        compare_records = compare_tool.func(excel_rows, eml_struct, base_url=base_url)
        compare_df = pd.DataFrame(compare_records) if compare_records else pd.DataFrame(columns=['row_index','source_column','source_value','normalized_value','matched','match_symbol','match_locations','comparison_method'])

        # 4) Build final outputs: here we keep one record per non-empty column (no early break)
        outputs = []
        for row_idx, row in enumerate(excel_rows):
            for col in self.priority_cols:
                # find compare entries for this row_idx and column
                matches = compare_df[(compare_df['row_index'] == row_idx) & (compare_df['source_column'] == col)] if not compare_df.empty else pd.DataFrame()
                if matches.empty:
                    # If compare didn't produce any record (maybe column was missing), but the excel has value, create an unmatched entry
                    val = row.get(col)
                    if val is None or str(val).strip() == '':
                        continue
                    outputs.append({
                        'row_index': row_idx,
                        'source_column': col,
                        'source_value': str(val).strip(),
                        'normalized_value': deterministic_normalize_url(val, base=base_url),
                        'matched': False,
                        'match_symbol': '❌',
                        'match_locations': [],
                        'comparison_method': 'none'
                    })
                else:
                    # there may be multiple compare rows (but compare_tool emits single per column)
                    for _, rec in matches.iterrows():
                        outputs.append({
                            'row_index': int(rec['row_index']),
                            'source_column': rec['source_column'],
                            'source_value': rec['source_value'],
                            'normalized_value': rec['normalized_value'],
                            'matched': bool(rec['matched']),
                            'match_symbol': rec['match_symbol'],
                            'match_locations': rec.get('match_locations', []),
                            'comparison_method': rec.get('comparison_method', 'none')
                        })

        result_df = pd.DataFrame(outputs)

        # 5) Optional LLM review for unmatched entries
        if self.llm_enabled:
            ambiguous = []
            for _, r in result_df.iterrows():
                if not r['matched']:
                    ambiguous.append({'source_column': r['source_column'], 'source_value': r['source_value'], 'normalized_value': r['normalized_value']})
            if ambiguous:
                llm_results = llm_review_tool.func(ambiguous, llm_model=self.llm_model)
                lookup = {(it['source_column'], it['source_value']): it for it in llm_results}
                llm_sugg, llm_expl = [], []
                for _, r in result_df.iterrows():
                    key = (r['source_column'], r['source_value'])
                    item = lookup.get(key)
                    llm_sugg.append(item.get('llm_suggestion') if item else None)
                    llm_expl.append(item.get('llm_explanation') if item else None)
                result_df['llm_suggestion'] = llm_sugg
                result_df['llm_explanation'] = llm_expl

        # Ensure columns exist
        expected_cols = ['row_index','source_column','source_value','normalized_value','matched','match_symbol','match_locations','comparison_method','llm_suggestion','llm_explanation']
        for c in expected_cols:
            if c not in result_df.columns:
                result_df[c] = None

        return result_df


# In[133]:


# Example usage (no CSV output). Print results
if __name__ == '__main__':
    example_xlsx = 'ARS00118_DE__Global_Matrix_v36.xlsx'
    example_eml = '[ERS00925_Sep_MNL-MainEntity=All Other-M1=Young Saver-M2=Prospect_ProAccount-proof] Kristin, grow your retirement savings___.eml'
    agent = ParserAgent(llm_enabled=False)
    results = agent.run(example_xlsx, example_eml, base_url=None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 300)
    print("Final results:")
    print(results[['row_index','source_column','source_value','match_symbol']])

