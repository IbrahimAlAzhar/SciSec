import json
import re
from typing import Dict, List, Any


class PromptTemplate:
    
    def __init__(
        self,
        aspect_tag_types: List[str] = None,
        aspect_score_types: List[str] = None,
        explain: bool = False,
        dataset: str = None
    ):
        self.explain = explain
        self.dataset = dataset
        self.aspect_tag_types = aspect_tag_types or [
            "NONE",
            "SUMMARY",
            "MOTIVATION POSITIVE",
            "MOTIVATION NEGATIVE",
            "SUBSTANCE POSITIVE",
            "SUBSTANCE NEGATIVE",
            "ORIGINALITY POSITIVE",
            "ORIGINALITY NEGATIVE",
            "SOUNDNESS POSITIVE",
            "SOUNDNESS NEGATIVE",
            "CLARITY POSITIVE",
            "CLARITY NEGATIVE",
            "REPLICABILITY POSITIVE",
            "REPLICABILITY NEGATIVE",
            "MEANINGFUL COMPARISON POSITIVE",
            "MEANINGFUL COMPARISON NEGATIVE",
        ]
        
        self.aspect_score_types = aspect_score_types or [
            "OVERALL",
            "SUBSTANCE",
            "APPROPRIATENESS",
            "MEANINGFUL_COMPARISON",
            "SOUNDNESS_CORRECTNESS",
            "ORIGINALITY",
            "CLARITY",
            "IMPACT",
        ]
    
    def load_paper_json(self, json_path: str) -> Dict[str, Any]:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def format_paper_content(self, paper_data: Dict[str, Any], num_figures: int = 0) -> str:
        content_parts = []
        
        if paper_data.get('metadata', {}).get('title'):
            content_parts.append(f"TITLE: {paper_data['metadata']['title']}")
        
        sections = paper_data.get('metadata', {}).get('sections', [])
        for section in sections:
            heading = section.get('heading', 'SECTION')
            text = section.get('text', '')
            if text.strip():
                content_parts.append(f"\n{heading}:\n{text}")
        
        if num_figures > 0:
            content_parts.append(f"\n[Note: This paper includes {num_figures} figure(s) shown in the images provided.]")
        
        return "\n".join(content_parts)
    
    def generate_prompt(
        self,
        paper_content: str,
    ) -> str:
        
        if self.explain:
            input = (
                "You are a professional reviewer. "
                "Based on the given content of a research paper, you need to write a review in ICLR style "
                "  "
            )
            input += (
                "tags types: "
                + ", ".join([f"[{tag}]" for tag in self.aspect_tag_types])
                + ". "
            )
            input += (
                "Your total output should not surpass 500 tokens, make sure to include both positive and negative aspects. "
                # explanation
                "Also, you need to predict the review score for several aspects based on the generated review, providing an explanation of each aspect in less than 30 tokens. "
                "Choose a integer score from 1 to 10, higher score means better paper quality. "
                "To assign 8–10, the paper must pass all gates: "
                "(G1) clear novelty, (G2) strong evidence/rigor, (G3) meaningful comparisons, (G4) clarity & reproducibility, (G5) significance beyond a niche. "
                "If any notable weakness or missing information exists, cap the affected aspect scores at 7. For each additional weakness, decrease the relevant score by 1."
                "When uncertain between two possible scores, choose the lower one."
            )
        else:
            input = (
                "You are a professional reviewer. "
                "Based on the given content of a research paper, you need to write a review in ICLR style "
                "and tag sentences with the corresponding tag type at the beginning of sequence: "
            )
            input += (
                "tags types: "
                + ", ".join([f"[{tag}]" for tag in self.aspect_tag_types])
                + ". "
            )
            input += (
                "Your total output should not surpass 500 tokens, make sure to include both positive and negative aspects. "
                "Also, you need to predict the review score in several aspects without explanation. "
                "Choose a integer score from 1 to 10, higher score means better paper quality. "
                "To assign 8–10, the paper must pass all gates: "
                "(G1) clear novelty, (G2) strong evidence/rigor, (G3) meaningful comparisons, (G4) clarity & reproducibility, (G5) significance beyond a niche. "
                "If any notable weakness or missing information exists, cap the affected aspect scores at 7. For each additional weakness, decrease the relevant score by 1."
                "When uncertain between two possible scores, choose the lower one."
                )
            
        output_format = (
            "Please strictly follow the format of Example output:\n"
            "1. REVIEW: tagged sequences.\n"
            "2. REVIEW SCORE: "
            + ", ".join([f"{label}: score" for label in self.aspect_score_types])
            + ".\n")

        if self.explain:
            output_format += (
            "3. REVIEW SCORE EXPLANATION: "
            + ", ".join([f"{label}: explanation" for label in self.aspect_score_types])
            + ".\n")
        
        prompt = f"{input}\n\n{paper_content}\n\n{output_format}"
        
        return prompt
    
    @staticmethod
    def parse_review_output(output_text: str) -> Dict[str, Any]:
        result = {
            "review": {},
            "scores": {},
            "explanation": {},
            "raw_output": output_text
        }
        
        try:
            # Extract REVIEW content (numbered format: 1. 2. 3.)
            review_match = re.search(r"1\.\s*REVIEW:\s*(.*?)(?:\n\n2\.|\Z)", output_text, re.DOTALL)
            review_text = review_match.group(1).strip() if review_match else ""

            score_match = re.search(r"2\.\s*REVIEW SCORE:\s*(.*?)(?:\n\n3\.|\Z)", output_text, re.DOTALL)
            score_text = score_match.group(1).strip() if score_match else ""
            
            explanation_match = re.search(r"3\.\s*REVIEW SCORE EXPLANATION:\s*(.*?)$", output_text, re.DOTALL)
            explanation_text = explanation_match.group(1).strip() if explanation_match else ""
            review_dict = {
                key.replace("*", ""): value
                for key, value in re.findall(
                    r"\[(.*?)\]\s*(.*?)(?=\s*\[|\Z)", review_text
                )
            }
            review_dict = {
                key.replace("*", "").strip(): value.strip()
                for key, value in re.findall(
                    r"\[(.*?)\]\s*(.*?)(?=\s*-?\s*\[|\Z)", review_text
                )
            }
            
            score_dict = {
                    key.replace("*", "").strip(): int(value)
                    for key, value in re.findall(r"([\w\s]+):\s*(\d+)", score_text)
                }
            explanation_dict = {
                key.replace("*", "").strip(): value.strip()
                for key, value in re.findall(
                    r"([\w\s]+):\s*(.*?)(?=\s*\w[\w\s]*:|$)", explanation_text
                )
            }

            result["review"] = review_dict
            result["scores"] = score_dict
            result["explanation"] = explanation_dict
        except Exception as e:
            print(f"Error parsing review output: {e}")
        
        return result

