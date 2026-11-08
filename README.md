# Math Reasoning with Test-Time Compute

An interactive AI-powered mathematical reasoning demo built with **Qwen2.5-1.5B-Instruct**, a **LoRA fine-tuned adapter trained on GSM8K**, and **Test-Time Compute (TTC)** through Self-Consistency. The application generates step-by-step solutions and can sample multiple reasoning paths before selecting the most frequent extracted answer.

> **Project type:** Research / educational demo  
> **Interface:** Streamlit  
> **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`  
> **Fine-tuning:** LoRA adapter (GSM8K)  
> **Inference methods:** Greedy decoding and Self-Consistency

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Checkpoint Setup](#model-checkpoint-setup)
- [Run the Application](#run-the-application)
- [Using the Interface](#using-the-interface)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

Small language models can produce different answers to the same mathematical question depending on the generated reasoning path. This project demonstrates a simple inference-time strategy for making answers more robust: generate several candidate solutions and use **Self-Consistency** to select the answer that appears most often.

The app offers two modes:

1. **Greedy decoding:** Generate one solution using deterministic decoding.
2. **Self-Consistency:** Generate multiple sampled solutions, extract their final answers, and select the most common answer by majority vote.

The application is designed as an interactive demonstration. Majority voting can improve answer reliability in some cases, but it does not guarantee mathematical correctness.

## Features

- **Step-by-step mathematical reasoning** with a Qwen2.5-1.5B-Instruct model and a GSM8K LoRA adapter.
- **Two inference modes:** standard greedy generation and Self-Consistency.
- **Adjustable sampling:** configure the number of candidate solutions and sampling temperature.
- **Configurable generation length:** control the maximum number of new tokens.
- **Majority-vote answer selection:** compare extracted answers from generated samples.
- **Candidate inspection:** expand a panel to inspect every generated solution.
- **Built-in example questions** based on arithmetic and word-problem tasks.
- **Recent session history:** review recent questions, answers, methods, and generation times.
- **Streamlit interface:** lightweight browser-based UI with progress feedback.

## How It Works

### 1. Model loading

The app loads the base model and tokenizer:

```text
Qwen/Qwen2.5-1.5B-Instruct
        +
Local LoRA adapter
        |
        v
PEFT model for inference
```

The adapter is loaded from:

```text
./checkpoints/qwen-1.5b-gsm8k-lora
```

The model is cached with Streamlit's `st.cache_resource` so it is not reloaded on every interaction.

### 2. Prompt construction

The user's question is placed into a step-by-step instruction prompt. The model is asked to provide its reasoning and final answer.

### 3. Greedy decoding

When Self-Consistency is disabled, the app generates one solution with sampling turned off. The generated text is displayed along with the extracted final answer.

### 4. Self-Consistency (Test-Time Compute)

When enabled, the app:

1. Generates multiple candidate solutions using sampling.
2. Uses the configured temperature and top-p sampling.
3. Extracts a candidate final answer from each solution.
4. Counts the extracted answers.
5. Selects the most frequent answer.
6. Displays the vote count, elapsed time, and all generated solutions.

Conceptually:

```text
                   Math problem
                        |
                   Prompt builder
                        |
             Generate N solutions
                /    |    \
             S1      S2    ... SN
                \    |    /
             Extract final answers
                        |
                  Count votes
                        |
              Most frequent answer
                        |
                  Display result
```

Self-Consistency uses additional inference calls. Increasing the sample count generally increases compute time and does not guarantee a better answer.

## System Architecture

| Component | Responsibility |
|---|---|
| Streamlit UI | Input, settings, examples, progress, output, and history |
| Transformers | Tokenization and autoregressive text generation |
| Qwen2.5-1.5B-Instruct | Base language model |
| PEFT | Loads the LoRA adapter onto the base model |
| PyTorch | Model execution and tensor operations |
| Answer extractor | Extracts a final numeric answer from generated text |
| Counter | Performs majority voting across candidate answers |
| Session State | Stores recent interaction history during the session |

## Repository Structure

The application expects the following layout:

```text
Math-Reasoning-with-Test-Time-Compute/
├── app.py
├── requirements.txt
├── README.md
└── checkpoints/
    └── qwen-1.5b-gsm8k-lora/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── ...                         # Other adapter/tokenizer files, if required
```

The checkpoint directory is a **runtime requirement** referenced by `app.py`. If it is not present in your clone, obtain or create the trained adapter and place it at the exact path above. The README does not assume that model weights are included in the repository.

## Requirements

- Python 3.10 or newer is recommended.
- pip and Git.
- Internet access on first run to download the base model from Hugging Face, unless it is already cached.
- The LoRA adapter checkpoint described in [Model Checkpoint Setup](#model-checkpoint-setup).
- Sufficient RAM and disk space for the model and its dependencies.

The current application loads the model on **CPU** using float32. A GPU is not required by the current code, but CPU inference—especially with multiple samples—can be slow.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Tauhid-Topu-007/Math-Reasoning-with-Test-Time-Compute.git
cd Math-Reasoning-with-Test-Time-Compute
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, you can use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch note:** If the default installation does not match your operating system or CUDA setup, install the appropriate PyTorch build by following the official [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining packages from `requirements.txt`.

## Model Checkpoint Setup

The application loads a local PEFT/LoRA adapter from:

```text
checkpoints/qwen-1.5b-gsm8k-lora/
```

Make sure this directory contains a valid adapter checkpoint, including at least the adapter configuration and adapter weights expected by PEFT (commonly `adapter_config.json` and `adapter_model.safetensors` or `adapter_model.bin`).

If you trained the adapter yourself, copy or export the adapter directory to this location. If the adapter is stored elsewhere, either move it to the expected path or update `lora_path` in `app.py`.

**Important:** The base model and adapter are separate. Downloading the base model does not provide the fine-tuned adapter. The app will fail during model loading if the adapter directory is missing or invalid.

## Run the Application

From the repository root, with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## Using the Interface

1. Start the app and wait for the model to load.
2. Choose an example question or enter your own math word problem.
3. In the sidebar, enable or disable **Self-Consistency**.
4. If Self-Consistency is enabled, choose the number of samples and temperature.
5. Set the maximum number of new tokens.
6. Select **Generate Solution**.
7. Review the final answer and, in Self-Consistency mode, inspect the candidate solutions.
8. Review recent session history or clear it from the sidebar.

### Sidebar settings

| Setting | Current default | Description |
|---|---:|---|
| Enable Self-Consistency | On | Switch between multi-sample voting and one greedy generation |
| Number of samples | 8 | Candidate solutions; slider range 4–16, step 2 |
| Temperature | 0.7 | Sampling temperature; slider range 0.1–1.0 |
| Max new tokens | 512 | Maximum generated tokens; slider range 256–1024 |

## Configuration

The main model and inference settings are currently defined in `app.py`:

| Variable / setting | Current value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| LoRA path | `./checkpoints/qwen-1.5b-gsm8k-lora` |
| Device map | `cpu` |
| Model dtype | `torch.float32` |
| Sampling top-p | `0.95` |
| Default temperature | `0.7` |
| Default sample count | `8` |
| Default maximum new tokens | `512` |

The application calls `load_dotenv()`, but the current implementation does not require an API key or external inference API. No `.env` file is necessary for the default workflow.

## Technical Details

### Answer extraction

The helper function `extract_answer()` first searches for a GSM8K-style final-answer marker:

```text
#### 42
```

If that marker is not found, it falls back to the last numeric value in the generated text. If no number is found, it returns `Not found`.

This is a lightweight heuristic, not a full mathematical parser. It may misread answers containing multiple numbers, decimals, units, fractions, or explanatory text.

### Majority voting

The app uses Python's `collections.Counter` to count extracted answer strings and selects the most common one. The displayed agreement is the winning vote count divided by the number of successfully extracted answers.

A tie is resolved by the order returned by `Counter.most_common()`; the app does not implement a separate tie-breaking or confidence-calibration strategy.

### History

Recent interactions are stored in Streamlit session state. The app displays up to the latest eight entries. This history is temporary and is not saved to a database.

## Limitations

- **No correctness guarantee:** majority voting can select a frequently repeated but incorrect answer.
- **Numeric extraction is heuristic:** the current parser is not robust to all answer formats.
- **CPU inference can be slow:** Self-Consistency performs several sequential generation calls.
- **No formal benchmark report is included in the UI:** do not interpret vote agreement as model accuracy.
- **Adapter dependency:** the local LoRA checkpoint must be supplied separately if it is not committed to the repository.
- **Session-only history:** clearing or restarting the session removes the stored history.
- **English-oriented examples:** the prompts and included examples are written in English.
- **No tool-based calculator:** the model generates solutions without executing arithmetic in a separate symbolic engine.

## Troubleshooting

### Adapter checkpoint not found

Confirm that this directory exists:

```text
checkpoints/qwen-1.5b-gsm8k-lora/
```

If your adapter is in another location, update `lora_path` in `app.py`.

### Out-of-memory or very slow inference

The current code loads the model on CPU in float32. Close other memory-intensive applications and try a smaller `max new tokens` value. Reduce the number of samples where possible; the current UI allows a minimum of four when Self-Consistency is enabled.

### PyTorch installation problems

Install a PyTorch build compatible with your Python version and hardware using the official [PyTorch guide](https://pytorch.org/get-started/locally/).

### Hugging Face download errors

Check your internet connection and confirm that the base model is accessible from your environment. The model identifier is `Qwen/Qwen2.5-1.5B-Instruct`.

### Streamlit command not found

Activate the virtual environment and reinstall dependencies:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Future Improvements

- Add a robust parser for integer, decimal, fraction, and structured final answers.
- Evaluate Greedy vs. Self-Consistency on a held-out GSM8K test set.
- Report exact-match accuracy, pass@k, latency, and token usage.
- Add answer normalization before majority voting.
- Explore weighted voting and confidence-aware aggregation.
- Add optional GPU / CUDA inference and configurable device selection.
- Add batch evaluation and exportable experiment logs.
- Add unit tests for answer extraction, voting, and generation utilities.
- Support persistent history and reproducible experiment configurations.
- Improve prompt templates and support more mathematical problem types.

## Acknowledgements

- [Qwen](https://huggingface.co/Qwen) for the Qwen2.5 model family.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for model loading and generation.
- [PEFT](https://huggingface.co/docs/peft) for parameter-efficient LoRA adapter loading.
- [GSM8K](https://github.com/openai/grade-school-math) for grade-school math word-problem data.
- [Streamlit](https://streamlit.io/) for the interactive application interface.

## License

No explicit license is currently specified in this repository. Unless a license file is added, all rights remain with the respective copyright holders and normal copyright restrictions apply. Add a `LICENSE` file before redistributing or reusing this project under an open-source license.
