"""
Intelligent Email QA Agent for Promotional Content
Analyzes email HTML, compares with reference images, and validates against brand guidelines
"""

import os
import base64
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
import json
import streamlit as st

# ======================================================================
# 1. LLM CLIENT INITIALIZATION
# ======================================================================

class LLMClient:
    """Initialize and manage OpenAI client for GPT-4o"""

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"

    def call(self, messages: list[dict], temperature: float = 0.3) -> str:
        """Call GPT-4o model"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=2000
        )
        return response.choices[0].message.content


# ======================================================================
# 2. COPY QA AGENT
# ======================================================================

@dataclass
class QAReport:
    tone_analysis: Dict
    grammar_issues: List[Dict]
    message_deviations: List[Dict]
    visual_consistency: Dict
    overall_score: float
    recommendations: List[str]


class CopyQAAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_text_from_html(self, html_content: str) -> Dict[str, str]:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        return {
            "subject": soup.find('title').get_text() if soup.find('title') else "",
            "headings": [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])],
            "body_text": soup.get_text(separator='\n', strip=True),
            "cta_buttons": [a.get_text().strip() for a in soup.find_all('a', href=True)
                            if 'button' in str(a.get('class', [])).lower()],
            "links": [a.get_text().strip() for a in soup.find_all('a', href=True)]
        }

    def encode_image(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def analyze_copy_quality(self, text_content: Dict, brand_guidelines: str) -> Dict:
        prompt = f"""You are an expert email QA analyst. Analyze the following email content against the brand guidelines.

EMAIL CONTENT:
- Subject: {text_content['subject']}
- Headings: {', '.join(text_content['headings'])}
- Body: {text_content['body_text'][:500]}...
- CTAs: {', '.join(text_content['cta_buttons'])}

BRAND GUIDELINES:
{brand_guidelines}

Provide a detailed analysis in JSON format with:
1. tone_analysis: {{consistency: score(0-100), issues: [list], brand_voice_match: score}}
2. grammar_issues: [{{location: str, issue: str, severity: str, suggestion: str}}]
3. message_deviations: [{{element: str, deviation: str, impact: str}}]
4. recommendations: [list of actionable improvements]
"""

        messages = [
            {"role": "system", "content": "You are an expert QA analyst for marketing emails."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.call(messages)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            return json.loads(response)
        except json.JSONDecodeError:
            return {"tone_analysis": {}, "grammar_issues": [], "message_deviations": [], "recommendations": [response]}

    def compare_with_reference_image(self, html_content: str, reference_image_bytes: bytes, text_content: Dict) -> Dict:
        base64_image = self.encode_image(reference_image_bytes)

        prompt = f"""Compare this reference email design image with the actual content below.

ACTUAL EMAIL CONTENT:
- Subject: {text_content['subject']}
- Headings: {', '.join(text_content['headings'])}
- CTAs: {', '.join(text_content['cta_buttons'])}

Analyze:
1. Visual consistency
2. Element placement
3. Content accuracy
4. Missing elements

Provide JSON output:
{{
    "visual_match_score": score(0-100),
    "discrepancies": [{{element: str, issue: str, severity: str}}],
    "missing_elements": [list],
    "layout_consistency": score(0-100)
}}"""

        messages = [
            {"role": "system", "content": "You are an expert at comparing email designs."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]

        response = self.llm.call(messages)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            return json.loads(response)
        except json.JSONDecodeError:
            return {"visual_match_score": 0, "discrepancies": [], "missing_elements": [], "layout_consistency": 0}

    def generate_report(self, html_content: str, reference_image_bytes: Optional[bytes], brand_guidelines: str) -> QAReport:
        text_content = self.extract_text_from_html(html_content)
        copy_analysis = self.analyze_copy_quality(text_content, brand_guidelines)

        visual_analysis = {}
        if reference_image_bytes:
            visual_analysis = self.compare_with_reference_image(html_content, reference_image_bytes, text_content)
        else:
            visual_analysis = {"visual_match_score": 0, "discrepancies": [], "missing_elements": [], "layout_consistency": 0}

        tone_score = copy_analysis.get('tone_analysis', {}).get('consistency', 0)
        visual_score = visual_analysis.get('visual_match_score', 0)
        overall_score = tone_score if visual_score == 0 else (tone_score * 0.6 + visual_score * 0.4)

        return QAReport(
            tone_analysis=copy_analysis.get('tone_analysis', {}),
            grammar_issues=copy_analysis.get('grammar_issues', []),
            message_deviations=copy_analysis.get('message_deviations', []),
            visual_consistency=visual_analysis,
            overall_score=overall_score,
            recommendations=copy_analysis.get('recommendations', [])
        )


# ======================================================================
# 3. STREAMLIT FRONTEND
# ======================================================================

def main():
    st.set_page_config(page_title="📧 Email QA Agent", layout="wide")
    st.title("📧 Email QA Agent (Streamlit Interface)")
    st.write("Upload your HTML, reference image, and brand checklist to run the QA analysis.")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    # api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
    # if not api_key:
    #     st.warning("Please enter your OpenAI API key to continue.")
    #     st.stop()

    html_file = st.file_uploader("📄 Upload Email HTML File", type=["html", "htm"])
    reference_image = st.file_uploader("🖼️ Upload Reference Image (optional)", type=["png", "jpg", "jpeg"])
    checklist_file = st.file_uploader("📋 Upload Brand Checklist / Guidelines", type=["txt", "md"])

    if st.button("🚀 Run QA Analysis"):
        if not html_file:
            st.error("Please upload an HTML file.")
            st.stop()

        with st.spinner("Initializing LLM client..."):
            llm_client = LLMClient(api_key=api_key)
            qa_agent = CopyQAAgent(llm_client)

        html_content = html_file.read().decode("utf-8")
        brand_guidelines = checklist_file.read().decode("utf-8") if checklist_file else "No checklist provided."
        image_bytes = reference_image.read() if reference_image else None

        with st.spinner("Running QA analysis... This might take a few minutes ⏳"):
            report = qa_agent.generate_report(html_content, image_bytes, brand_guidelines)

        st.success("✅ QA Analysis Complete!")
        st.subheader("📊 Overall Score")
        st.metric("Overall QA Score", f"{report.overall_score:.1f}/100")

        st.subheader("📝 Tone Analysis")
        st.json(report.tone_analysis)

        st.subheader("✏️ Grammar Issues")
        st.json(report.grammar_issues if report.grammar_issues else "✅ No issues detected")

        st.subheader("📋 Message Deviations")
        st.json(report.message_deviations if report.message_deviations else "✅ None found")

        st.subheader("🖼️ Visual Consistency")
        st.json(report.visual_consistency)

        st.subheader("💡 Recommendations")
        for i, rec in enumerate(report.recommendations, 1):
            st.write(f"**{i}.** {rec}")


# ======================================================================
# 4. ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
