import os
import json
import re
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
from typing import Dict, List

def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

class TableQAEvaluator:
    def __init__(self, model_name: str, prompt_file: str, tensor_parallel_size: int = 1):
        print(f"Loading evaluator with vLLM: {model_name}")
        self.model_name = model_name
        
        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=4096,
        )
        
        # Set up sampling parameters for evaluation
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            stop=["[Evaluation End]"]
        )
        
        self.prompt_template = self.load_prompt(prompt_file)
        print(f"Loaded prompt from {prompt_file}")
        print("vLLM model loaded successfully!")

    def load_prompt(self, prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def parse_response(self, response: str) -> Dict:
        """Extract raw score and reasoning from LLM response"""
        
        # Initialize defaults
        raw_score = 0.0
        reasoning = ""
        
        # Clean up the response and truncate at stop token if present
        response = response.strip()
        if "[Evaluation End]" in response:
            response = response.split("[Evaluation End]")[0].strip()
        
        # Method 1: Try JSON parsing first
        try:
            json_pattern = r'\{[^}]*"score"[^}]*\}'
            json_match = re.search(json_pattern, response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                if "score" in data:
                    raw_score = float(data["score"])
                    reasoning = str(data.get("reasoning", "")).strip()
                    
                    return {
                        'raw_score': raw_score,
                        'reasoning': reasoning,
                        'raw_response': response
                    }
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Method 2: Fallback - search for score keywords
        score_patterns = [
            r'"score":\s*([\d.]+)',
            r'score["\']?\s*[:=]\s*([\d.]+)',
            r'\*\*final\s+score\*\*:?\s*([\d.]+)',
            r'\*\*score\*\*:?\s*([\d.]+)',
            r'\*\*scoring\*\*:?\s*([\d.]+)',
            r'final\s+score:?\s*([\d.]+)',
            r'score:?\s*([\d.]+)',
            r'scoring:?\s*([\d.]+)',
            r'FINAL\s+SCORE:?\s*([\d.]+)',
            r'SCORE:?\s*([\d.]+)',
            r'SCORING:?\s*([\d.]+)',
            r'my\s+score:?\s*([\d.]+)',
            r'the\s+score:?\s*([\d.]+)',
            r'overall\s+score:?\s*([\d.]+)',
        ]
        
        for text_to_search in [response, response.lower()]:
            for pattern in score_patterns:
                score_match = re.search(pattern, text_to_search, re.IGNORECASE)
                if score_match:
                    try:
                        raw_score = float(score_match.group(1))
                        break
                    except ValueError:
                        continue
            else:
                continue
            break
        
        # Method 3: Extract reasoning separately
        if not reasoning:
            reasoning_patterns = [
                r'"reasoning":\s*"([^"]*)"',
                r'"reasoning":\s*\'([^\']*)\'',
                r'reasoning["\']?\s*[:=]\s*["\']([^"\']*)["\']',
                r'\*\*reasoning\*\*:?\s*([^\n\*]+)',
                r'reasoning:?\s*([^\n]+)',
            ]
            
            for pattern in reasoning_patterns:
                reasoning_match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    break
        
        if not reasoning:
            reasoning = "No reasoning extracted from response"
        
        return {
            'raw_score': raw_score,
            'reasoning': reasoning,
            'raw_response': response
        }

    def evaluate_answers_batch(self, questions, ground_truths, predictions):
        """Evaluate a batch of answers using vLLM"""
        batch_prompts = [
            self.prompt_template.format(question=q, ground_truth=gt, prediction=pred)
            for q, gt, pred in zip(questions, ground_truths, predictions)
        ]
        
        outputs = self.llm.generate(batch_prompts, self.sampling_params)
        
        results = []
        for output in outputs:
            response = output.outputs[0].text
            result = self.parse_response(response)
            results.append(result)
        
        return results

    def run_evaluation(self, generation_file: str, output_file: str, batch_size: int = 32):
        """Run evaluation on generation results and save complete results with summary"""
        
        if not output_file:
            base_name = os.path.splitext(os.path.basename(generation_file))[0]
            output_file = f"{base_name}_eval.json"
        
        output_file = os.path.join('evaluations', os.path.basename(output_file))
        os.makedirs('evaluations', exist_ok=True)
        
        # Load generation results
        with open(generation_file, 'r', encoding='utf-8') as f:
            generation_data = json.load(f)
        
        results = generation_data['results']
        model_name = generation_data['metadata']['model_name']
        
        print(f"Evaluating {len(results)} responses from {model_name}")
        
        eval_results = []
        
        with tqdm(total=len(results), desc="Evaluating", unit="eval") as pbar:
            for batch_start in range(0, len(results), batch_size):
                batch_end = min(batch_start + batch_size, len(results))
                batch_data = results[batch_start:batch_end]
                
                # Extract batch data and filter out empty predictions
                questions = []
                ground_truths = []
                predictions = []
                valid_indices = []

                for i, item in enumerate(batch_data):
                    if item['prediction'].strip():  # Non-empty prediction
                        questions.append(item['question'])
                        ground_truths.append(item['ground_truth'])
                        predictions.append(item['prediction'])
                        valid_indices.append(i)

                # Evaluate only non-empty predictions
                if predictions:
                    batch_eval_results = self.evaluate_answers_batch(questions, ground_truths, predictions)
                else:
                    batch_eval_results = []
                
                # Combine with original data
                eval_idx = 0
                for i, original in enumerate(batch_data):
                    if i in valid_indices:  # Has LLM evaluation
                        eval_result = batch_eval_results[eval_idx]
                        eval_idx += 1
                        llm_score = eval_result['raw_score']
                        reasoning = eval_result['reasoning']
                        eval_raw_response = eval_result['raw_response']
                    else:  # Empty prediction - no LLM call
                        llm_score = 0.0
                        reasoning = ""
                        eval_raw_response = ""
                    
                    result = {
                        "row_id": original['row_id'],
                        "question": original['question'],
                        "ground_truth": original['ground_truth'],
                        "prediction": original['prediction'],
                        "llm_score": llm_score,
                        "reasoning": reasoning,
                        "evaluation_raw_response": eval_raw_response
                    }
                    eval_results.append(result)
                
                pbar.update(len(batch_data))
        
        # Calculate summary statistics (binary: only 1.0 = correct)
        total_questions = len(eval_results)
        correct_answers = sum(1 for r in eval_results if r['llm_score'] == 1.0)
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        # Prepare final output
        final_output = {
            "metadata": {
                "model_name": model_name,
                "evaluator_model": self.model_name,
                "total_questions": total_questions
            },
            "individual_results": eval_results,
            "summary": {
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "accuracy": accuracy
            }
        }
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation completed!")
        print(f"Accuracy: {accuracy:.2%} ({correct_answers}/{total_questions})")
        print(f"Results saved to: {output_file}")
        
        return final_output

def main():
    parser = argparse.ArgumentParser(description='Table Reasoning Evaluation with vLLM')
    parser.add_argument('--evaluator_model', default='meta-llama/Llama-3.3-70B-Instruct',
                       help='Evaluator model path/name')
    parser.add_argument('--prompt_file', default='eval_prompt.txt')
    parser.add_argument('--generation_file', required=True, 
                       help='JSON file with generation results')
    parser.add_argument('--output_file', default=None,
                       help='Output JSON file for evaluation results (default: [generation_file]_eval.json)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    args = parser.parse_args()

    evaluator = TableQAEvaluator(
        args.evaluator_model, 
        args.prompt_file,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    evaluator.run_evaluation(
        args.generation_file,
        args.output_file, 
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()