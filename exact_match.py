import json
import argparse
import os
import glob

def normalize_text(s):
    """
    Simple normalization: strip whitespace and convert to string.
    You can add .lower() here if you want case-insensitive EM.
    """
    if s is None:
        return ""
    return str(s).strip()

def process_file(file_path, save_inplace=False):
    """
    Calculates Exact Match (EM) scores for a single file.
    """
    print(f"Processing: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Handle different JSON structures (list vs dict with 'individual_results')
    if isinstance(data, list):
        results = data
        is_list_root = True
        metadata = {} # No metadata if root is list
    else:
        results = data.get('individual_results', data.get('results', []))
        metadata = data.get('metadata', {})
        is_list_root = False

    if not results:
        print(f"  -> No results found in {file_path}. Skipping.")
        return

    em_correct_count = 0
    total_questions = len(results)

    # --- Calculation Loop ---
    for item in results:
        # 1. Get Fields (Handle 'answer' vs 'ground_truth' key variations)
        pred = normalize_text(item.get('prediction', ''))
        gt = normalize_text(item.get('ground_truth') or item.get('answer', ''))

        # 2. Compare (Exact Match)
        is_em = 1.0 if pred == gt else 0.0
        
        # 3. Store result in the item
        item['em_score'] = is_em
        
        # (Optional) Add a debug field to see why it matched/didn't
        # item['em_debug'] = f"'{pred}' vs '{gt}'" 

        if is_em == 1.0:
            em_correct_count += 1

    # --- Summary Calculation ---
    em_accuracy = em_correct_count / total_questions if total_questions > 0 else 0.0

    print(f"  -> Total: {total_questions}")
    print(f"  -> Exact Matches: {em_correct_count}")
    print(f"  -> EM Accuracy: {em_accuracy:.2%}")

    # Update Summary in JSON (if it exists or create it)
    if not is_list_root:
        if 'summary' not in data:
            data['summary'] = {}
        
        # Preserve existing LLM-judge accuracy if present
        llm_acc = data['summary'].get('accuracy', 'N/A')
        if isinstance(llm_acc, float):
            print(f"  -> vs LLM Accuracy: {llm_acc:.2%}")
            
        data['summary']['em_accuracy'] = em_accuracy
        data['summary']['em_correct_count'] = em_correct_count

    # --- Saving ---
    if save_inplace:
        output_path = file_path
        print(f"  -> Updating existing file.")
    else:
        output_path = file_path.replace('.json', '_with_em.json')
        print(f"  -> Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Add Exact Match (EM) scores to evaluation files.')
    parser.add_argument('--files', nargs='+', help='Specific JSON files to process')
    parser.add_argument('--dir', help='Directory to process all .json files in')
    parser.add_argument('--inplace', action='store_true', help='Overwrite input files instead of creating new ones')
    args = parser.parse_args()

    files_to_process = []

    if args.files:
        files_to_process = args.files
    elif args.dir:
        files_to_process = glob.glob(os.path.join(args.dir, "*.json"))
    else:
        # Default to current directory or a specific folder if you prefer
        files_to_process = glob.glob("evaluations/*.json")

    if not files_to_process:
        print("No files found to process.")
        return

    print(f"Found {len(files_to_process)} files...")
    for f in files_to_process:
        process_file(f, save_inplace=args.inplace)
        print("-" * 30)

if __name__ == "__main__":
    main()