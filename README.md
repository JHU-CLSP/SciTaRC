# SciTaRC: Benchmarking QA on Scientific Tabular Data that Requires Language Reasoning and Complex Computation

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/JHU-CLSP/SciTaRC)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2603.08910)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Data License: CC BY-NC 4.0](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**SciTaRC** is an expert-authored benchmark of questions about tabular data in scientific papers requiring both deep language reasoning and complex computation. We show that current state-of-the-art AI models fail on at least 23% of these questions. 

## ⚙️ Setup

Clone the repository and install the minimal dependencies:

    git clone https://github.com/JHU-CLSP/SciTaRC.git
    cd SciTaRC
    pip install -r requirements.txt

## 📊 Dataset

The benchmark data is provided locally as `scitarc_dataset.json` and is also accessible via [Hugging Face](https://huggingface.co/datasets/JHU-CLSP/SciTaRC). The dataset consists of 371 expert-annotated questions. Every instance includes an expert-annotated **pseudo-code plan** to facilitate granular diagnosis of model failures.

### Quick Look (Demo Instance)

    {
      "paper": "2401.06769",
      "relevant_tables": [
        [
          "\\begin{table*}[h!]\n",
          "...",
          "\\end{table*}\n"
        ]
      ],
      "question": "Which model has the biggest difference in translation quality when translating into English versus from English, and what is the value of that difference?",
      "answer": "NLLB-200-1.3B. 64.71",
      "plan": "SELECT all models\nLOOP for each model\n    SELECT all language pair containing en(English)\n    LOOP for each language pair containing en (English)\n        COMPUTE diff = abs(score translating into English − score translating from English)\n..."
    }

### Data Fields

* `paper` *(string)*: The arXiv ID of the source scientific paper.
* `question` *(string)*: The complex, multi-step question.
* `answer` *(string)*: The ground-truth answer.
* `plan` *(string)*: The expert-authored pseudo-code blueprint (e.g., SELECT, LOOP, COMPUTE, IF).
* `relevant_tables` *(list)*: The exact LaTeX source code for the specific table(s) required.
* `tables` *(list)*: The LaTeX source code for all tables and figures extracted from the paper.
* `fulltext` *(string)*: The complete LaTeX source text of the original scientific paper.

## 🚀 Running Inference (`generate.py`)

Our unified inference script uses `vllm` and supports testing the execution bottleneck by separating reasoning plans from execution. 

If no `--output-file` is provided, outputs are **automatically named and saved** to `generations/[model_tag]_[plan_mode]_[exec_mode].json`.

**Key Arguments:**
* `--plan-mode`: `none` (Direct QA), `self` (Autonomous Planning), `gold` (Oracle Planning).
* `--exec-mode`: `language` (Chain-of-Thought), `code` (Program-of-Thought).
* `--use-hf`: Add this flag to stream the dataset directly from Hugging Face instead of the local JSON.

**Standard Direct QA (No Plan):**

    python generate.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --plan-mode none \
        --exec-mode language

**Oracle Code Execution (Gold Plan + Program of Thoughts):**

    python generate.py \
        --model-id Qwen/Qwen2.5-Coder-7B-Instruct \
        --plan-mode gold \
        --exec-mode code

*Outputs are saved to the `generations/` directory.*

## ⚖️ Evaluation

### 1. LLM-as-a-Judge (`evaluate.py`)
Because answers are free-form and complex, we use an **LLM-as-a-Judge** protocol (aligned >95% with human annotators) to robustly evaluate logical reasoning and mathematical accuracy.

If no `--output-json` is provided, results are **automatically named and saved** to `evaluations/[generation_filename]_eval.json`.

    python evaluate.py \
        --generation-json generations/YOUR_GENERATION_FILE.json \
        --evaluator-model meta-llama/Llama-3.3-70B-Instruct \
        --prompt-file eval_prompt.txt

### 2. Exact Match (`exact_match.py`)
We also provide an Exact Match (EM) script for strict baseline comparisons. You can run this on single files or entire directories. Use the `--inplace` flag to append the EM scores directly to your existing evaluation JSONs.

    python exact_match.py --files evaluations/[YOUR_FILE]_eval.json --inplace

## 📈 Complexity Metrics (`get_metrics.py`)

Calculate input and reasoning complexity metrics ($C_{flow}$, $I_{calc}$, $L_{plan}$, $S_{cell}$) to reproduce our findings on model performance degradation:

    python get_metrics.py scitarc_dataset.json complexity_metrics.csv

## 📖 Citation

If you use this dataset, please cite our paper:

```bibtex
@misc{wang2026scitarc,
      title={SciTaRC: Benchmarking QA on Scientific Tabular Data that Requires Language Reasoning and Complex Computation}, 
      author={Hexuan Wang and Yaxuan Ren and Srikar Bommireddypalli and Shuxian Chen and Adarsh Prabhudesai and Rongkun Zhou and Elina Baral and Philipp Koehn},
      year={2026},
      eprint={2603.08910},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.08910}, 
}
```