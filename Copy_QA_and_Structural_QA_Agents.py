# %%
# Import libraries

import os
import json
from bs4 import BeautifulSoup
from openai import OpenAI
from crewai import Agent, Task, Crew
from crewai.tools import tool

# %%
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o"

# %%
# Structural QA Function

def structural_qa(html_content: str):
    prompt = """
You are an HTML Structural QA Agent. 
Check the email for structural problems ONLY.
Return JSON in this structure:
{
 "status": "pass/fail",
 "issues": [
    {"severity": "critical/major/minor/info", "issue": "...", "element": "..."}
 ]
}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": html_content}
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


# %%
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o" 

# %%
from bs4 import BeautifulSoup

MAX_EMAIL_CHARS = 12000   # adjust if needed
MAX_CHECKLIST_CHARS = 8000

# %%
def html_to_text(html_content: str) -> str:
    """Convert HTML to visible text to reduce tokens."""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

# %%
def truncate_text(text: str, max_chars: int) -> str:
    """Truncate very long text to keep within token limits."""
    if len(text) <= max_chars:
        return text
    # Keep beginning and end (email headers/footer may matter)
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n...\n[TRUNCATED]\n...\n" + tail

# %%
# Copy QA Function

def copy_qa(html_content: str, checklist_text: str):
    # 1) Convert HTML to plain text (massive token saver)
    email_text = html_to_text(html_content)

    # 2) Truncate both sides so we never send a huge blob
    email_text = truncate_text(email_text, MAX_EMAIL_CHARS)
    checklist_text = truncate_text(checklist_text, MAX_CHECKLIST_CHARS)

    # 3) Build the prompt (no unescaped {} issues)
    prompt = (
        "You are a Copy QA Agent.\n"
        "Compare the email copy with the checklist rules.\n"
        "Only use the content provided below.\n\n"
        "Return JSON strictly in this structure:\n"
        "{\n"
        '  \"tone\": \"...\",\n'
        '  \"grammar_issues\": [],\n'
        '  \"missing_content\": [],\n'
        '  \"recommendations\": []\n'
        "}\n\n"
        "CHECKLIST:\n"
        f"{checklist_text}\n\n"
        "EMAIL TEXT:\n"
        f"{email_text}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a precise QA assistant that outputs strict JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content

# %%
from langchain_openai import ChatOpenAI

# %%
def safe_read(path):
    """Safely read text files using multiple fallback encodings."""
    encodings_to_try = ["utf-8", "cp1252", "latin-1", "iso-8859-1", "utf-16"]
    
    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            continue
    
    # As a last resort: ignore invalid bytes
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# %%
# Define Tools

@tool("structural_qa_tool")
def structural_qa_tool(html_path: str):
    """Run structural QA on an HTML file and return issues as JSON."""
    if not os.path.exists(html_path):
        return json.dumps({"error": "html file not found", "path": html_path})

    html = safe_read(html_path)
    return structural_qa(html)
    
@tool("copy_qa_tool")
def copy_qa_tool(html_path: str, checklist_path: str):
    """Run copy QA based on checklist and return structured JSON."""
    if not os.path.exists(html_path):
        return json.dumps({"error": "html file not found", "path": html_path})
    if not os.path.exists(checklist_path):
        return json.dumps({"error": "checklist file not found", "path": checklist_path})

    html = safe_read(html_path)
    checklist = safe_read(checklist_path)

    return copy_qa(html, checklist)

# %%
# Initialize LLM

llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# %%
# Declare Agents

copy_agent = Agent(
    role="Copy QA Agent",
    goal="Perform copy QA and return structured findings.",
    backstory="Expert email QA specialist.",
    tools=[copy_qa_tool],
    llm=llm,
    verbose=True
)

struct_agent = Agent(
    role="Structural QA Agent",
    goal="Find structural issues in HTML.",
    backstory="Expert in email rendering.",
    tools=[structural_qa_tool],
    llm=llm,
    verbose=True
)

summary_agent = Agent(
    role="Summary Agent",
    goal="Summarize results from both agents.",
    backstory="Documentation expert.",
    llm=llm,
    verbose=True
)

# %%
# Declare Tasks

task_copy = Task(
    description="Run Copy QA on {html_path} using checklist {checklist_path}.",
    agent=copy_agent,
    expected_output="Copy QA JSON"
)

task_struct = Task(
    description="Run Structural QA on {html_path}.",
    agent=struct_agent,
    expected_output="Structural QA JSON"
)

task_summary = Task(
    description=(
        "You are a QA summarizer. You will receive the outputs of the Copy QA task "
        "and the Structural QA task as context.\n\n"
        "Your job is to combine these into a single JSON object with this structure:\n"
        "{\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"source\": \"copy_qa\" | \"structural_qa\",\n"
        "      \"area\": \"subject line\" | \"body\" | \"layout\" | \"images\" | \"links\" | \"overall\" | \"other\",\n"
        "      \"severity\": \"critical\" | \"major\" | \"minor\" | \"info\",\n"
        "      \"issue\": \"short description of the issue\",\n"
        "      \"recommendation\": \"clear recommendation to fix it\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Only include entries for actual issues (do not include items that are fully OK).\n"
        "Return ONLY valid JSON – no extra commentary, no markdown, no explanation."
    ),
    agent=summary_agent,
    context=[task_copy, task_struct],
    expected_output="JSON with a list of issues and recommendations."
)


# %%
# Summarizer Function

import csv
import json
import os

def save_summary_to_csv(summary_result, csv_path: str = "qa_issues_summary.csv"):
    """
    Save issues + recommendations from summarizer JSON into a CSV file.
    `summary_result` can be:
      - a JSON string
      - a CrewOutput object
      - anything convertible to string containing JSON
    """
    # 1) Normalize to string
    if isinstance(summary_result, (str, bytes, bytearray)):
        summary_json_str = summary_result
    else:
        # Handle CrewOutput from CrewAI
        # Different versions may expose final text under different attributes
        if hasattr(summary_result, "raw"):
            summary_json_str = summary_result.raw
        elif hasattr(summary_result, "output"):
            summary_json_str = summary_result.output
        elif hasattr(summary_result, "final_output"):
            summary_json_str = summary_result.final_output
        else:
            # Fallback: whatever __str__ gives
            summary_json_str = str(summary_result)

    # 2) Try to parse JSON
    try:
        data = json.loads(summary_json_str)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse summary as JSON:", e)
        print("Raw summary was:\n", summary_json_str)
        return

    items = data.get("items", [])
    if not items:
        print("ℹ️ No 'items' found in summary JSON. Nothing to save.")
        return

    fieldnames = ["source", "area", "severity", "issue", "recommendation"]

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({
                "source":         item.get("source", ""),
                "area":           item.get("area", ""),
                "severity":       item.get("severity", ""),
                "issue":          item.get("issue", ""),
                "recommendation": item.get("recommendation", ""),
            })

    print(f"✅ Saved QA issues summary to: {csv_path}")


# %%
# Declare Crew

crew = Crew(
    agents=[copy_agent, struct_agent, summary_agent],
    tasks=[task_copy, task_struct, task_summary]
)

# %%
# Output CSV Path

csv_path = r"C:\Users\ks25\OneDrive - dentsu\NXT 24\outputs_qa_issues_summary.csv"

# %%
# Calling the Agents

inputs = {
    "html_path": r"C:\Users\ks25\OneDrive - dentsu\NXT 24\[ERS00925_Sep_MNL-MainEntity=All Other-M1=Young Saver-M2=Prospect_ProAccount-proof] Kristin, grow your retirement savings___.html",
    "checklist_path": r"C:\Users\ks25\OneDrive - dentsu\NXT 24\NW Checklist - September Monthly Newsletter_2025.xlsx"
    # or whichever checklist you want
}

# Run the crew – this will execute Copy QA, Structural QA, then Summary
result = crew.kickoff(inputs=inputs)

# Optional: see the raw JSON summary
print("Final summary from summary_agent:\n", result)

# Save one output of the summarizer into a CSV file
save_summary_to_csv(
    summary_result=result,
    csv_path=r"C:\Users\ks25\OneDrive - dentsu\NXT 24\outputs\qa_issues_summary.csv"
)

# %%
# inputs = {
#     "html_path": r"C:\Users\ks25\OneDrive - dentsu\NXT 24\[ERS00925_Sep_MNL-MainEntity=All Other-M1=Young Saver-M2=Prospect_ProAccount-proof] Kristin, grow your retirement savings___.html",
#     "checklist_path": r"C:\Users\ks25\OneDrive - dentsu\NXT 24\NW Checklist - September Monthly Newsletter_2025.xlsx"
#     # "checklist_path": r"C:\Users\ks25\OneDrive - dentsu\NXT 24\NW Checklist - LIB Awareness.csv"
# }

# result = crew.kickoff(inputs=inputs)
# print(result)



