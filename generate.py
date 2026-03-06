import os
import json
import re
import sys
import io
import warnings
import textwrap
import argparse
from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
import math

llm = None
sampler = None

def init_vllm(model_id: str, tensor_parallel_size: int = 1):
    global llm, sampler
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )
    sampler = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=2048,
        repetition_penalty=1.05
    )

def query_vllm(prompt: str) -> str:
    return llm.generate([prompt], sampler)[0].outputs[0].text

# ============================================
# PROMPTS
# ============================================

PSEUDOCODE_SPEC = """Operations:
- SELECT: Extract specific values or subsets from the table (e.g., SELECT F1 scores for BERT)
- LOOP: Iterate over items (e.g., LOOP for each model)
- COMPUTE: Numerical operations - can be detailed or high-level:
    * Detailed: COMPUTE average = (score_A + score_B) / 2
    * High-level: COMPUTE average across all datasets
    * With assignment: COMPUTE best_model = argmax(accuracy)
    * Increment: COMPUTE increment count
    * Comparison: COMPUTE difference between model A and model B
- IF: Conditional logic (e.g., IF score > threshold)
- RETURN: State what to return for the final answer (e.g., RETURN model with highest accuracy)

Rules:
- Use exact syntax: "LOOP for each", "SELECT", "COMPUTE", "IF", "RETURN"
- Indent after LOOP and IF
- Use actual entity names (e.g., "model", "dataset", "F1 score"), not generic "row" or "column"
- For multi-table questions, specify which table (e.g., SELECT from Table 1)
- No LOOP needed for simple operations like argmax/argmin
- No "END LOOP" needed"""

PSEUDOCODE_READING_GUIDE = """How to read the plan:
- SELECT: Look up and extract the specified values from the table
- LOOP for each X: Repeat the indented steps for every X
- COMPUTE: Perform the calculation described (may be a formula or high-level operation like "average", "argmax", "difference")
- IF: Only do the indented steps when the condition is true
- RETURN: This is the final answer to provide"""

def get_table_text(relevant_tables):
    return "\n\n".join("".join(table) for table in relevant_tables)

def create_plan_prompt(question, relevant_tables):
    """Prompt to generate plan (only for auto_plan mode)."""
    return f"""You are a helpful assistant that creates step-by-step plans for answering questions about tables.

{PSEUDOCODE_SPEC}

Example plans:

Example 1:
LOOP for each model
    SELECT accuracy on Dataset A
    SELECT accuracy on Dataset B
    COMPUTE average across datasets
COMPUTE best_model = argmax(average)
RETURN model with highest average accuracy

Example 2:
SELECT results for GPT-4 and BERT
COMPUTE difference = GPT-4 score - BERT score
RETURN the difference

Example 3:
LOOP for each dataset
    SELECT score for model A
    SELECT score for model B
    IF model A > model B
        COMPUTE increment count_A_wins
RETURN total count of A wins

Table:
{get_table_text(relevant_tables)}

Question: {question}

Write a concise step-by-step plan using the operations above. Do not solve it, just write the plan:
"""

def create_language_prompt(question, relevant_tables):
    """Language prompt WITHOUT plan."""
    return f"""You are a helpful science assistant who answers questions about information in tables.

Here is the relevant tabular data:

{get_table_text(relevant_tables)}

You may think through the question step by step. Your final response should be "Answer:" followed by the answer.

Question: {question}
"""

def create_language_prompt_with_plan(question, relevant_tables, plan):
    """Language prompt WITH plan."""
    return f"""You are a helpful science assistant who answers questions about information in tables.

Here is the relevant tabular data:

{get_table_text(relevant_tables)}

Here is a step-by-step plan to follow:

{plan}

{PSEUDOCODE_READING_GUIDE}

Follow this plan carefully step by step to answer the question. Your final response should be "Answer:" followed by the answer.

Question: {question}
"""

def create_code_prompt(question, relevant_tables):
    """Code prompt WITHOUT plan."""
    return f"""Write Python code to analyze the table and answer the question.

Table:
{get_table_text(relevant_tables)}

Question: {question}

Write Python code that prints the final answer.
"""

def create_code_prompt_with_plan(question, relevant_tables, plan):
    """Code prompt WITH plan."""
    return f"""Write Python code to analyze the table and answer the question.

Table:
{get_table_text(relevant_tables)}

Here is a step-by-step plan to implement:

{plan}

{PSEUDOCODE_READING_GUIDE}

Question: {question}

Write Python code that implements this plan and prints the final answer.
"""

# ============================================
# ANSWER EXTRACTION
# ============================================

def extract_answer_language(raw_response: str) -> str:
    m = re.search(r'(?:^|\n)\s*(?:final\s*)?answer\s*:\s*(.*)$', raw_response, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else raw_response.strip()

def extract_answer_code(raw_response: str) -> str:
    code = raw_response.strip()
    
    # Extract code from ```python blocks
    code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", code, flags=re.S | re.I)
    if code_blocks:
        code = code_blocks[0]
    code = textwrap.dedent(code).strip()
    
    # Execute
    stdout_backup = sys.stdout
    sys.stdout = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(code, "<string>", "exec"), {
                "pd": pd, "json": json, "re": re, "np": np,
                "numpy": np, "math": math, "__builtins__": __builtins__
            })
        stdout = sys.stdout.getvalue().strip()
        lines = [l.strip() for l in stdout.splitlines() if l.strip()]
        return lines[-1] if lines else ""
    except:
        return ""
    finally:
        sys.stdout = stdout_backup

def extract_plan(raw_response: str) -> str:
    plan = raw_response.strip()
    blocks = re.findall(r"```(?:plan|text)?\n?(.*?)```", plan, flags=re.S | re.I)
    return blocks[0].strip() if blocks else plan

# ============================================
# UTILITY
# ============================================

def extract_model_tag(model_id: str) -> str:
    tag = model_id.split('/')[-1].lower()
    return re.sub(r'[^a-zA-Z0-9_-]', '_', tag)

def load_existing_results(output_file: str):
    path = os.path.join('generations', os.path.basename(output_file))
    try:
        with open(path, 'r') as f:
            return json.load(f).get('results', [])
    except FileNotFoundError:
        return []

def save_results(output_file: str, results: list, metadata: dict):
    path = os.path.join('generations', os.path.basename(output_file))
    os.makedirs('generations', exist_ok=True)
    with open(path, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2, ensure_ascii=False)

# ============================================
# MAIN
# ============================================

def main(args):
    init_vllm(args.model_id, tensor_parallel_size=args.tensor_parallel_size)
    
    model_tag = args.model_tag or extract_model_tag(args.model_id)
    output_file = args.output_json or f"{model_tag}_{args.plan_mode}_{args.exec_mode}.json"
    
    print(f"Model: {model_tag} | Plan: {args.plan_mode} | Exec: {args.exec_mode}")
    
    if args.use_hf:
        print(f"Loading dataset from Hugging Face: jhu-clsp/SciTaRC")
        from datasets import load_dataset
        dataset = load_dataset("jhu-clsp/SciTaRC", split="test")
    else:
        print(f"Loading local dataset: {args.dataset_json}")
        with open(args.dataset_json) as f:
            dataset = json.load(f)
    
    existing_results = load_existing_results(output_file)
    completed = {r['row_id'] for r in existing_results}
    results = existing_results.copy()
    
    metadata = {
        "model_name": model_tag,
        "model_id": args.model_id,
        "plan_mode": args.plan_mode,
        "exec_mode": args.exec_mode,
        "total_questions": len(dataset)
    }
    
    for i, item in enumerate(dataset):
        if i in completed:
            continue
        
        question = item['question']
        tables = item['relevant_tables']
        
        # Step 1: Get plan (if needed)
        plan = None
        generated_plan = None
        
        if args.plan_mode == "oracle":
            plan = item.get('plan', '')
        elif args.plan_mode == "auto":
            plan_prompt = create_plan_prompt(question, tables)
            generated_plan = extract_plan(query_vllm(plan_prompt))
            plan = generated_plan
        
        # Step 2: Create execution prompt
        if args.exec_mode == "language":
            if plan:
                exec_prompt = create_language_prompt_with_plan(question, tables, plan)
            else:
                exec_prompt = create_language_prompt(question, tables)
            extract_answer = extract_answer_language
        else:  # code
            if plan:
                exec_prompt = create_code_prompt_with_plan(question, tables, plan)
            else:
                exec_prompt = create_code_prompt(question, tables)
            extract_answer = extract_answer_code
        
        # Step 3: Execute and extract answer
        raw_response = query_vllm(exec_prompt)
        prediction = extract_answer(raw_response)
        
        print(f"[{i}] Pred: {prediction[:80]}... | GT: {item.get('answer', '')}")
        
        # Step 4: Build result
        result = {
            "row_id": i,
            "question": question,
            "ground_truth": item.get('answer', ''),
            "prediction": prediction,
            "raw_response": raw_response
        }
        
        if args.plan_mode == "oracle":
            result["gold_plan"] = plan
        elif args.plan_mode == "auto":
            result["gold_plan"] = item.get('plan', '')
            result["generated_plan"] = generated_plan
        
        results.append(result)
        save_results(output_file, results, metadata)
    
    print(f"\nDone! {len(results)} results saved to generations/{os.path.basename(output_file)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run inference on the SciTaRC benchmark.")
    
    p.add_argument("--dataset-json", default="scitarc_dataset.json", 
                   help="Path to the local SciTaRC dataset JSON file.")
    p.add_argument("--use-hf", action="store_true", default=True,
                   help="Load dataset from Hugging Face instead of local JSON.")
    p.add_argument("--model-id", required=True, 
                   help="Hugging Face model ID")
    p.add_argument("--model-tag", default=None, 
                   help="Short custom name for the model, used for naming the output file.")
    p.add_argument("--tensor-parallel-size", type=int, default=1, 
                   help="Number of GPUs to use for tensor parallelism.")
    p.add_argument("--output-json", default=None, 
                   help="Custom filename/path to save the generation results.")
    p.add_argument("--plan-mode", required=True, choices=["none", "auto", "oracle"], 
                   help="Planning strategy: 'none' (Direct QA), 'auto' (Model generates plan), 'oracle' (Uses ground-truth plan).")
    p.add_argument("--exec-mode", required=True, choices=["language", "code"], 
                   help="Execution strategy: 'language' (Chain-of-Thought) or 'code' (Program-of-Thought).")

    args = p.parse_args()
    main(args)