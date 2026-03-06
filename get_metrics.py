import json
import re
import csv
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

def calculate_robust_depth(lines):
    raw_indents = []
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            raw_indents.append(len(line) - len(stripped))
    
    if not raw_indents: return 0
    non_zero_indents = [n for n in raw_indents if n > 0]
    if not non_zero_indents: return 0
    indent_unit = min(non_zero_indents)
    return max(raw // indent_unit for raw in raw_indents)

def extract_task_signature(pseudocode_text):
    defaults = {
        'plan_length': 0, 'max_depth': 0, 'retrieval_count': 0,
        'arithmetic_count': 0, 'loop_count': 0, 'if_count': 0, 'logic_complexity': 0
    }
    if not pseudocode_text: return defaults

    lines = [l for l in pseudocode_text.split('\n') if l.strip()]
    if not lines: return defaults

    plan_length = len(lines)
    max_depth = calculate_robust_depth(lines)
    
    n_select = n_compute = n_loop = n_if = 0
    for line in lines:
        clean_line = line.strip().upper()
        if clean_line.startswith('SELECT'): n_select += 1
        elif clean_line.startswith('COMPUTE'): n_compute += 1
        elif clean_line.startswith('LOOP'): n_loop += 1
        elif clean_line.startswith('IF'): n_if += 1
    
    return {
        'plan_length': plan_length,
        'max_depth': max_depth,
        'retrieval_count': n_select + n_if,
        'arithmetic_count': n_compute,
        'loop_count': n_loop,
        'if_count': n_if,
        'logic_complexity': n_loop + n_if + max_depth
    }

def detect_answer_type(answer_text):
    """
    Robustly detects if an answer is Numerical (involves values) or Categorical (entities).
    Handles mixed cases like 'Model A, 67.5'.
    """
    if not answer_text: return 'unknown'
    s = str(answer_text).strip()
    
    # Rule 1: No digits at all? -> Categorical (e.g., "Yes", "English", "Biomed")
    if not re.search(r'\d', s):
        return 'categorical'
        
    # Rule 2: Purely numeric chars? -> Numerical (e.g., "0.45", "1,200", "-5%")
    # Allowing spaces, commas, dots, signs, %, $
    if re.match(r'^[\d\.\,\-\+\s%\$]+$', s):
        return 'numerical'

    # Rule 3: Mixed Case Handling (The tricky part)
    # We want to catch "WizardLM, 67.6" but ignore "Llama-3"
    
    # A. If it contains a comma (,) or colon (:), it's likely a list/mapping of values
    # e.g., "Llama: 0.5, GPT: 0.6"
    if re.search(r'[\:,]', s):
        return 'numerical'
        
    # B. If it contains a Float (decimal point followed by digit), treat as Numerical
    # e.g., "Model X 64.71" -> Numerical
    # e.g., "Llama-3" -> Categorical (No decimal)
    # Note: "NLLB-1.3B" might trigger this, but parameter size is arguably numerical reasoning.
    if re.search(r'\.\d', s):
        return 'numerical'
    
    # C. Fallback: If it has text + integer but no separators, assume Entity Name
    # e.g., "Llama 3", "GPT-4", "Version 2"
    return 'categorical'

def count_cells_in_table(latex_table):
    table = latex_table.replace(r'\%', '__PCT__').replace(r'\&', '__AMP__')
    table = re.sub(r'%.*', '', table)
    rows = re.findall(r'(.*?)\\\\', table, re.DOTALL)
    total_cells = 0
    for row in rows:
        row = re.sub(r'\\(toprule|midrule|bottomrule|hline|cmidrule|addlinespace)', '', row)
        row = row.strip()
        if not row: continue
        base_cells = row.count('&') + 1
        multicolumns = re.findall(r'\\multicolumn\{(\d+)\}', row)
        extra_cells = sum(int(n) - 1 for n in multicolumns)
        total_cells += base_cells + extra_cells
    return total_cells

def analyze_tables(tables_list, tokenizer):
    total_cells = total_tokens = 0
    for table in tables_list:
        table_text = "".join(table) if isinstance(table, list) else str(table)
        total_cells += count_cells_in_table(table_text)
        total_tokens += len(tokenizer.tokenize(table_text))
    return {
        'num_tables': len(tables_list),
        'total_cells': total_cells,
        'total_tokens': total_tokens
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json')
    parser.add_argument('output_csv')
    args = parser.parse_args()
    
    # Safe tokenizer loading
    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    except:
        print("Warning: Defaulting to whitespace tokenization.")
        class SimpleTokenizer:
            def tokenize(self, t): return t.split()
        tokenizer = SimpleTokenizer()
    
    with open(args.input_json, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} items")
    
    results = []
    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        plan_text = item.get('plan') or item.get('pseudocode', '')
        tables = item.get('relevant_tables', [])
        answer = item.get('answer', '')
        
        plan_metrics = extract_task_signature(plan_text)
        table_metrics = analyze_tables(tables, tokenizer)
        answer_type = detect_answer_type(answer)
        
        row = {
            'row_id': i,
            'question_id': item.get('question_id', f'q_{i}'),
            'answer': answer,         # Save answer text for verification
            'answer_type': answer_type,
            **plan_metrics,
            **table_metrics
        }
        results.append(row)
    
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
        # Stats
        n_num = sum(1 for r in results if r['answer_type'] == 'numerical')
        n_cat = sum(1 for r in results if r['answer_type'] == 'categorical')
        print(f"\nAnswer Type Stats:")
        print(f"  Numerical:   {n_num} ({n_num/len(results)*100:.1f}%)")
        print(f"  Categorical: {n_cat} ({n_cat/len(results)*100:.1f}%)")
        print(f"Saved to {args.output_csv}")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()