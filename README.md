# SciTaRC: Benchmarking QA on Scientific Tabular Data that Requires Language Reasoning and Complex Computation

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/mattwang123/SciTaRC)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/YOUR_ARXIV_ID_HERE)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Data License: CC BY-NC 4.0](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**SciTaRC** is an expert-authored benchmark of questions about tabular data in scientific papers requiring both deep language reasoning and complex computation. We show that current state-of-the-art AI models fail on at least 30% of these questions. 
<!-- 
## 🏆 Key Findings

Our analysis reveals a universal **"execution bottleneck"**: both code and language models struggle to faithfully execute plans, even when provided with correct strategies. Specifically, code-based methods prove brittle on raw scientific tables, while natural language reasoning primarily fails due to initial comprehension issues and calculation errors.

<p align="center">
  <img src="assets/agreement_matrix.png" alt="Model Agreement Matrix" width="800"/>
</p>
<em>Figure 1: <strong>The Hard Ceiling.</strong> The Model Agreement Matrix illustrates that while stronger models perform better overall, a solid band of questions remains unsolved by any system (gray band at the bottom), indicating a hard limit on current composite reasoning capabilities.</em>

<p align="center">
  <img src="assets/gain_curve.png" alt="Performance Gain Curves" width="800"/>
</p>
<em>Figure 2: <strong>The Execution Bottleneck.</strong> Performance Gain Curves show that while code-specialized models benefit from explicit planning on hard tasks, generalist models regress on easy tasks due to the "Cost of Compliance."</em>

*(Note to author: Save Figure 3 as `assets/agreement_matrix.png` and Figure 6 as `assets/gain_curve.png` to display these charts!)* -->

## ⚙️ Setup

Clone the repository and install the minimal dependencies:

    git clone [https://github.com/mattwang123/SciTaRC.git](https://github.com/mattwang123/SciTaRC.git)
    cd SciTaRC
    pip install -r requirements.txt

## 📊 Dataset

The benchmark data is provided locally as `scitarc_dataset.json` and is also accessible via [Hugging Face](https://huggingface.co/datasets/mattwang123/SciTaRC). The dataset consists of 371 expert-annotated questions. Every instance includes an expert-annotated **pseudo-code plan** to facilitate granular diagnosis of model failures.

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
      "plan": "SELECT all models\nLOOP for each mode\n    SELECT all language pair containing en(English)\n    LOOP for each language pair containing en (English)\n        COMPUTE diff = abs(score translating into English − score translating from English)\n..."
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

If you use this benchmark, please cite our paper:

    @misc{wang2026scitarc,
          title={SciTaRC: Benchmarking QA on Scientific Tabular Data that Requires Language Reasoning and Complex Computation}, 
          author={Wang, Hexuan and Ren, Yaxuan and Bommireddypalli, Srikar and Chen, Shuxian and Prabhudesai, Adarsh and Zhou, Rongkun and Baral, Elina and Koehn, Philipp},
          year={2026},
          eprint={YOUR_ARXIV_ID},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
    }