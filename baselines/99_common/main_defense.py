
import os
import sys
import json
import argparse
import random
import glob
from typing import Dict, Any, List, Tuple
from pathlib import Path

from defense.emb_search import EmbSearchDefense
from defense.llm_judge import LLMJudge
from defense.prompts import split_prompts_into_groups, sample_prompt_from_group
from defense.metrics import calculate_metrics_from_defense_results, print_metrics
from prompts import PromptTemplate
# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip


def find_paper_directories(base_dir: str = "/data") -> List[str]:
   
    pattern = os.path.join(base_dir, "rejected_papers_*_with_figures")
    directories = glob.glob(pattern)
    return directories


def load_all_papers(data_dirs: List[str]) -> List[Tuple[str, str, Dict[str, Any], List[str]]]:
    
    all_papers = []
    prompt_template = PromptTemplate(explain=False)
    
    for data_dir in data_dirs:
        parsed_pdfs_dir = os.path.join(data_dir, "parsed_pdfs")
        figures_dir = os.path.join(data_dir, "figures")
        
        if not os.path.exists(parsed_pdfs_dir):
            print(f"Warning: {parsed_pdfs_dir} does not exist, skipping...")
            continue
        
        # Find all JSON files in parsed_pdfs
        json_files = glob.glob(os.path.join(parsed_pdfs_dir, "*.pdf.json"))
        
        for json_path in json_files:
            paper_id = os.path.basename(json_path).replace(".pdf.json", "")
            
            try:
                # Load paper data
                with open(json_path, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                # Find figures
                figure_paths = []
                for i in [1, 2]:
                    fig_path = os.path.join(figures_dir, f"{paper_id}-{i}.png")
                    if os.path.exists(fig_path):
                        figure_paths.append(fig_path)
                
                all_papers.append((paper_id, data_dir, paper_data, figure_paths))
                
            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}")
                continue
    
    return all_papers


def inject_prompt(
    paper_content: str,
    payload_text: str,
    position: str = None
) -> str:

    if position is None:
        position = random.choice(["beginning", "end"])
    
    if position == "beginning":
        return payload_text + "\n\n" + paper_content
    else:
        return paper_content + "\n\n" + payload_text


def split_papers_into_groups(
    papers: List[Tuple[str, str, Dict[str, Any], List[str]]],
    seed: int = None
) -> Tuple[List[Tuple[str, str, Dict[str, Any], List[str]]], 
           List[Tuple[str, str, Dict[str, Any], List[str]]], 
           List[Tuple[str, str, Dict[str, Any], List[str]]]]:
    """
    Split papers into three groups: B1, B2, B3.
    
    Args:
        papers: List of paper tuples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (B1, B2, B3) paper lists
        - B1: No injection (clean papers)
        - B2: Sample prompts from A2 (A1 + same class prompts)
        - B3: Sample prompts from A3 (rest prompts)
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle papers
    shuffled_papers = papers.copy()
    random.shuffle(shuffled_papers)
    
    # Split into three roughly equal groups
    n = len(shuffled_papers)
    n_per_group = n // 3
    
    b1 = shuffled_papers[:n_per_group]
    b2 = shuffled_papers[n_per_group:2*n_per_group]
    b3 = shuffled_papers[2*n_per_group:]
    
    return b1, b2, b3


def evaluate_defense(
    papers: List[Tuple[str, str, Dict[str, Any], List[str]]],
    paper_group: str,
    prompt_group: List[Tuple[str, str, bool]],
    defense_method: Any,
    defense_type: str,
    prompt_template: PromptTemplate,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Evaluate defense on a group of papers.
    
    Args:
        papers: List of paper tuples
        paper_group: "B1", "B2", or "B3"
        prompt_group: List of prompts to sample from (only used for B2/B3)
        defense_method: Defense instance (EmbSearchDefense or LLMJudge)
        defense_type: "embed" or "llm"
        prompt_template: PromptTemplate instance
        verbose: Whether to print verbose output
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    for idx, (paper_id, data_dir, paper_data, figure_paths) in enumerate(papers, 1):
        if verbose:
            print(f"Processing {paper_group} paper {idx}/{len(papers)}: {paper_id}")
        
        try:
            # Format paper content
            paper_content = prompt_template.format_paper_content(
                paper_data,
                num_figures=len(figure_paths)
            )
            
            # Inject prompt if in B2 or B3
            injected_prompt_name = None
            if paper_group == "B2" or paper_group == "B3":
                # Sample a prompt from the prompt group
                prompt_name, payload_text, _ = sample_prompt_from_group(prompt_group)
                paper_content = inject_prompt(paper_content, payload_text)
                injected_prompt_name = prompt_name
                ground_truth = 1  # Attack present
            else:
                ground_truth = 0  # No attack
            
            final_prompt = prompt_template.generate_prompt(paper_content)
            # defense_result = defense_method.detect(final_prompt)
            
            # Run defense
            if defense_type == "embed":
                defense_result = defense_method.detect(paper_content)
                # defense_result = defense_method.detect(final_prompt)

                prediction = 1 if defense_result['is_attack'] else 0
                score = defense_result.get('confidence', defense_result.get('top_similarity', 0.0))
            elif defense_type == "llm":
                defense_result = defense_method.detect(paper_content)
                # defense_result = defense_method.detect(final_prompt)

                prediction = 1 if defense_result['is_attack'] else 0
                score = defense_result.get('confidence', 0.0)
            else:
                raise ValueError(f"Unknown defense type: {defense_type}")
            
            # Store result
            result = {
                "paper_id": paper_id,
                "data_dir": data_dir,
                "paper_group": paper_group,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "score": score,
                "injected_prompt": injected_prompt_name,
                "defense_result": defense_result
            }
            results.append(result)
            
            if verbose and (idx % 10 == 0 or idx == len(papers)):
                print(f"  Processed {idx}/{len(papers)} papers...")
                
        except Exception as e:
            print(f"Error processing paper {paper_id}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Defense evaluation against prompt injection attacks"
    )
    
    # Data directory options
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/data",
        help="Base directory to search for rejected_papers_*_with_figures directories"
    )
    
    parser.add_argument(
        "--data-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Specific data directories to use (overrides --base-dir search)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file for results"
    )
    
    # Defense method selection
    parser.add_argument(
        "--defense-method",
        type=str,
        choices=["llm", "embed"],
        default="llm",
        help="Defense method to use: 'llm', or 'embed'"
    )
    
    # embed defense options
    parser.add_argument(
        "--model-name",
        type=str,
        default="text-embedding-3-large",
        help="Azure OpenAI embedding model name for embed"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top similar prompts to check (embed defense)"
    )
    
    # LLM judge options
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="LLM model for judgment"
    )
    
    # Prompt splitting options
    parser.add_argument(
        "--a1-size",
        type=int,
        default=10,
        help="Number of prompts in A1 (few known prompts)"
    )
    
    # Paper splitting options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (for testing)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
    
    # Find paper directories
    if args.data_dirs:
        data_dirs = args.data_dirs
    else:
        print(f"Searching for paper directories in {args.base_dir}...")
        data_dirs = find_paper_directories(args.base_dir)
        if not data_dirs:
            print(f"Error: No rejected_papers_*_with_figures directories found in {args.base_dir}")
            sys.exit(1)
    
    print(f"Found {len(data_dirs)} data directories:")
    for dd in data_dirs:
        print(f"  - {dd}")
    
    # Load all papers
    print(f"\nLoading all papers...")
    all_papers = load_all_papers(data_dirs)
    
    if args.max_papers:
        all_papers = all_papers[:args.max_papers]
        print(f"Limited to {len(all_papers)} papers (--max-papers={args.max_papers})")
    
    print(f"Loaded {len(all_papers)} papers")
    
    # Split papers into groups
    print(f"\nSplitting papers into B1/B2/B3 groups (seed={args.seed})...")
    b1_papers, b2_papers, b3_papers = split_papers_into_groups(all_papers, seed=args.seed)
    print(f"  B1 (no injection): {len(b1_papers)} papers")
    print(f"  B2 (sample from A2): {len(b2_papers)} papers")
    print(f"  B3 (sample from A3): {len(b3_papers)} papers")
    
    # Split prompts into groups
    print(f"\nSplitting prompts into A1/A2/A3 groups (seed={args.seed})...")
    a1_prompts, a2_prompts, a3_prompts = split_prompts_into_groups(
        seed=args.seed,
        a1_size=args.a1_size
    )
    print(f"  A1 (few known prompts): {len(a1_prompts)} prompts")
    print(f"  A2 (A1 + same class prompts): {len(a2_prompts)} prompts")
    print(f"  A3 (rest prompts): {len(a3_prompts)} prompts")
    
    # Initialize defense method
    print(f"\nInitializing defense method: {args.defense_method}")
    if args.defense_method == "llm":
        defense = LLMJudge(
            model_type=args.judge_model,
            temperature=0.0,
            known_prompts=a1_prompts
        )
        print(f"  LLM Judge initialized with {len(a1_prompts)} known prompts (A1)")
    elif args.defense_method == "embed":
        from defense.emb_search import EmbSearchDefense
        defense = EmbSearchDefense(
            known_prompts=a1_prompts,
            top_k=args.top_k,
            embed_model=args.model_name
        )
        print(f"  Embedding search initialized with {len(a1_prompts)} known prompts (A1)")
    else:
        raise ValueError(f"Unknown defense method: {args.defense_method}")
    
    # Initialize prompt template
    prompt_template = PromptTemplate(explain=False)
    
    # Evaluate on all groups
    print(f"\n{'='*80}")
    print("EVALUATING DEFENSE")
    print(f"{'='*80}")
    
    all_results = []
    
    # B1: No injection
    print(f"\nEvaluating B1 (no injection, {len(b1_papers)} papers)...")
    b1_results = evaluate_defense(
        b1_papers,
        "B1",
        [],
        defense,
        args.defense_method,
        prompt_template,
        verbose=not args.quiet
    )
    all_results.extend(b1_results)
    
    # B2: Sample from A2
    print(f"\nEvaluating B2 (sample from A2, {len(b2_papers)} papers)...")
    b2_results = evaluate_defense(
        b2_papers,
        "B2",
        a2_prompts,
        defense,
        args.defense_method,
        prompt_template,
        verbose=not args.quiet
    )
    all_results.extend(b2_results)
    
    # B3: Sample from A3
    print(f"\nEvaluating B3 (sample from A3, {len(b3_papers)} papers)...")
    b3_results = evaluate_defense(
        b3_papers,
        "B3",
        a3_prompts,
        defense,
        args.defense_method,
        prompt_template,
        verbose=not args.quiet
    )
    all_results.extend(b3_results)
    
    # Calculate metrics
    print(f"\n{'='*80}")
    print("CALCULATING METRICS")
    print(f"{'='*80}")
    
    metrics = calculate_metrics_from_defense_results(all_results)
    print_metrics(metrics)
    
    # Prepare output
    output_data = {
        "defense_method": args.defense_method,
        "defense_config": {
            "model_name": args.model_name if args.defense_method == "embed" else None,
            "judge_model": args.judge_model if args.defense_method == "llm" else None,
            "top_k": args.top_k if args.defense_method == "embed" else None,
        },
        "prompt_config": {
            "a1_size": args.a1_size,
            "a2_size": len(a2_prompts),
            "a3_size": len(a3_prompts),
        },
        "paper_config": {
            "total_papers": len(all_papers),
            "b1_papers": len(b1_papers),
            "b2_papers": len(b2_papers),
            "b3_papers": len(b3_papers),
        },
        "seed": args.seed,
        "metrics": metrics,
        "results": all_results
    }
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
