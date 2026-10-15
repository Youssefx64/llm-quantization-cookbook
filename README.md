# 🧮 LLM Quantization — A Practical, Hands-On Reference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Enabled-orange.svg)](https://huggingface.co/docs/transformers)
[![Colab Ready](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

A comprehensive, self-contained, hands-on master reference for **every widely used Large Language Model (LLM) quantization technique**.

This project provides both the mathematical foundations and working code for quantizing real open-source language models (default: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)), benchmarking each approach across **memory footprint**, **perplexity (Wikitext-2)**, **inference speed (tokens/sec)**, and **peak GPU VRAM** against the unquantized FP16 baseline.

Everything is engineered to fit within a single **free Google Colab T4 GPU (15 GB VRAM)**.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Quantization Methods Covered](#-quantization-methods-covered)
- [Evaluation Harness & Metrics](#-evaluation-harness--metrics)
- [Decision Matrix & Guide](#-decision-matrix--guide)
- [Rules of Thumb](#-rules-of-thumb)
- [Repository Structure](#-repository-structure)
- [Quickstart & Setup](#-quickstart--setup)
  - [Google Colab](#google-colab)
  - [Local Installation](#local-installation)
- [Quantization Deep Dive](#-quantization-deep-dive)
  - [From Scratch (Affine, Group-wise, Bit-Packing)](#from-scratch)
  - [Weight-Only (W4A16 / W8A16)](#weight-only-w4a16--w8a16)
  - [Weight-and-Activation (W8A8 / SmoothQuant)](#weight-and-activation-w8a8--smoothquant)
  - [Extreme Compression (<4-bit)](#extreme-compression-4-bit)
- [References & Reading](#-references--reading)
- [Contributing & License](#-contributing--license)

---

## 🚀 Overview

Quantization maps continuous, high-precision floating-point weights and activations (e.g., FP32 or FP16) to discrete, low-bit integers (e.g., INT8, INT4, INT2) or specialized formats (e.g., NF4, FP8, FP4).

This repository is designed to answer three practical engineering questions:
1. **How does each algorithm work under the hood?** (From basic affine mapping to second-order Taylor expansions and Hessian inversion).
2. **How do you run, calibrate, and export it?** (Using standard libraries like `transformers`, `bitsandbytes`, `optimum`, `gptqmodel`, `autoawq`, `llama.cpp`, and `torchao`).
3. **What is the exact trade-off?** (Quantifying perplexity loss, generation speedup, and memory reduction).

---

## 📚 Quantization Methods Covered

The reference notebook covers 16 core sections:

| # | Topic / Method | Bits | Target | Key Characteristics & Libraries |
|---|---|---|---|---|
| **0** | **Setup & Harness** | — | — | Standardized benchmark: size, perplexity (Wikitext-2), tokens/s, peak VRAM |
| **1** | **FP16 Baseline** | 16-bit | Weights & Acts | Golden reference for speed, memory, and perplexity |
| **2** | **Quantization Theory** | — | — | Affine math, symmetric vs asymmetric, scales, zero-points, granularity (per-tensor, per-channel, per-group) |
| **3** | **From Scratch** | 8-bit & 4-bit | Weights | Custom PyTorch implementation: absmax, asymmetric, group-wise, int4 bit packing/unpacking, full LLM round-to-nearest |
| **4** | **bitsandbytes** | 8-bit & 4-bit | Weights | `LLM.int8()` (vector outlier decomposition), `NF4` (NormalFloat4), `FP4`, Double Quantization (`DQ`) |
| **5** | **GPTQ** | 4-bit & 8-bit | Weights | Hessian-driven second-order error minimization (`gptqmodel` / `optimum`) |
| **6** | **AWQ** | 4-bit | Weights | Activation-aware weight quantization, protects top 1% salient weights (`autoawq`) |
| **7** | **GGUF / llama.cpp** | 2-bit to 8-bit | Weights | K-quants (`Q2_K`, `Q3_K`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`) for CPU/Metal/CUDA inference |
| **8** | **HQQ** | 1-bit to 4-bit | Weights | Half-Quadratic Quantization; fast, zero calibration data needed (`hqq`) |
| **9** | **Quanto & TorchAO** | 2-bit to 8-bit | Weights & Acts | PyTorch-native quantization, composable PyTorch kernels (`optimum-quanto`, `torchao`) |
| **10** | **SmoothQuant** | 8-bit (W8A8) | Weights & Acts | Outlier smoothing: scales activation channels into weights for fast INT8 GEMM |
| **11** | **KV-Cache Quantization** | 4-bit & 8-bit | KV-Cache | FP8 / INT4 KV-cache compression to scale to massive context lengths |
| **12** | **QLoRA** | 4-bit | Weights | Fine-tuning PEFT LoRA adapters on top of frozen NF4 base models |
| **13** | **QAT** | Variable | Weights & Acts | Quantization-Aware Training with fake quantization and Straight-Through Estimators (STE) |
| **14** | **Frontier Methods** | 1-bit to 8-bit | All | Survey of FP8 (E4M3/E5M2), BitNet b1.58 (1.58-bit ternary), AQLM, QuIP#, Marlin kernels, vLLM serving |
| **15** | **Pareto Benchmark** | All | All | Side-by-side Pareto curves: Model Size vs Perplexity vs Throughput |
| **16** | **HF Hub Export** | All | Artifacts | Packaging, model card generation, and uploading quantized checkpoints to Hugging Face |

---

## 📊 Evaluation Harness & Metrics

All methods are evaluated consistently on an identical test harness:

```
+--------------------------------------------------------------------------+
|                          EVALUATION HARNESS                              |
+---------------------+----------------------------------------------------+
| Model Size (MB)     | Exact footprint in memory or on-disk artifact size |
| Perplexity (PPL)    | Wikitext-2 test set (512-token non-overlapping)   |
| Generation Speed    | Tokens/second (Greedy 128-token decode, bs=1)     |
| Peak GPU VRAM (MB)  | Measured via torch.cuda.max_memory_allocated()    |
| Sample Output       | Qualitative generation check on fixed prompt       |
+---------------------+----------------------------------------------------+
```

---

## 🧭 Decision Matrix & Guide

Which quantization scheme should you use? Use this table as a quick reference:

| Deployment Scenario | Recommended Format | Why |
|---|---|---|
| **Local Laptop / Mac / CPU / Ollama** | **GGUF `Q4_K_M`** (or `Q5_K_M`) | Single-file distribution, optimal CPU/Metal SIMD instructions, fast startup |
| **NVIDIA GPU Single-User Inference** | **AWQ 4-bit** or **GPTQ 4-bit** | Fast W4A16 GEMM / Marlin fused kernels, high decoding throughput |
| **High-Throughput Enterprise Serving** | **FP8** or **W8A8 (SmoothQuant)** | True Tensor Core compute acceleration (INT8/FP8 GEMM), not just bandwidth savings |
| **Fast Prototyping / No Calibration** | **bitsandbytes NF4** or **HQQ** | Zero calibration dataset required, single-line Hugging Face integration |
| **Fine-Tuning on Consumer GPUs** | **QLoRA (NF4 + Double Quant)** | Train adapters on frozen 4-bit weights with minimal VRAM |
| **Extreme Compression (< 3-bit)** | **HQQ**, **AQLM**, **QuIP#**, **GGUF IQ** | Advanced vector codebooks or optimization designed for sub-4-bit |
| **Near-Lossless Quality Retention** | **INT8 (bnb / GPTQ-8bit / Q8_0)** or **Q6_K** | Perplexity loss typically within < 1% of FP16 baseline |

---

## 💡 Rules of Thumb

1. **4-bit is the sweet spot:** Provides ~3.5×–4× memory reduction with negligible perplexity degradation when using group-wise scales (e.g., group size 128).
2. **Bigger quantized model > smaller unquantized model:** A 7B parameter model quantized to 4-bit almost always outperforms a 3B model in FP16, despite having similar VRAM requirements.
3. **Small models are more sensitive:** Quantizing a 1B model causes a larger relative quality degradation than quantizing a 70B model.
4. **Bandwidth vs Compute Bound:** At batch size 1 (greedy generation), LLM inference is memory-bandwidth bound. Weight-only quantization (W4A16) yields significant speedups by moving fewer bytes from VRAM. At high batch sizes, compute dominates, necessitating W8A8 or FP8 compute kernels.
5. **Match calibration data to your target domain:** Calibration datasets should reflect the distribution of downstream tasks (e.g., code datasets for code models).

---

## 📁 Repository Structure

```text
LLM-Quantization/
│
├── LLM_Quantization.ipynb      # Main master notebook (all 16 sections)
├── LICENSE                     # MIT License
├── .gitignore                  # Git exclusions for Python, caches, and weights
└── README.md                   # Project documentation and guide
```

---

## 🛠️ Quickstart & Setup

### Google Colab

The easiest way to run this project is on Google Colab with a free T4 GPU:

1. Upload `LLM_Quantization.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Navigate to **Runtime → Change runtime type** and select **T4 GPU**.
3. Run the cells sequentially.

### Local Installation

To run the notebook on a local Linux or Windows machine with an NVIDIA GPU:

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/LLM-Quantization.git
cd LLM-Quantization

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt # or install packages below:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes optimum datasets sentencepiece huggingface_hub

# 4. (Optional) Install specialized quantization tools as needed:
pip install gptqmodel --no-build-isolation
pip install autoawq
pip install llama-cpp-python
pip install torchao optimum-quanto
pip install git+https://github.com/dropbox/hqq.git

# 5. Launch Jupyter Lab or Notebook
jupyter lab LLM_Quantization.ipynb
```

---

## 🔍 Quantization Deep Dive

### From Scratch

Affine quantization maps a continuous range $[x_{\min}, x_{\max}]$ to discrete integers $[q_{\min}, q_{\max}]$:

$$q = \text{clip}\left(\left\lfloor \frac{x}{S} \right\rceil + Z, \; q_{\min}, \; q_{\max}\right)$$

$$\tilde{x} = S \cdot (q - Z)$$

Where:
- $S$ is the **Scale Factor**: $S = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$
- $Z$ is the **Zero-Point**: $Z = \left\lfloor - \frac{x_{\min}}{S} \right\rceil + q_{\min}$

The notebook implements symmetric absmax, asymmetric zero-point, group-wise scaling, and 4-bit byte-packing from first principles in PyTorch.

### Weight-Only (W4A16 / W8A16)

Weights are stored in 4-bit or 8-bit integers, dequantized on-the-fly to 16-bit float during matrix multiplication:
- **bitsandbytes NF4:** Information-theoretically optimal quantile distribution for normally-distributed weights.
- **GPTQ:** Uses the inverse Hessian matrix $H^{-1} = (2 X X^T + \lambda I)^{-1}$ to greedily compensate remaining unquantized weights for errors introduced by rounded weights.
- **AWQ:** Observes that only 0.1%–1% of activation channels contain high-magnitude outliers. By scaling these salient channels, weight truncation errors are sharply minimized without mixed-precision overhead.

### Weight-and-Activation (W8A8 / SmoothQuant)

Activations exhibit persistent systematic outlier channels across transformer layers. **SmoothQuant** applies a per-channel smoothing factor $s$ to migrate the activation dynamic range into the weights:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \hat{W}$$

This enables standard INT8 $\times$ INT8 GEMM hardware acceleration on Tensor Cores.

---

## 📖 References & Reading

- **LLM.int8():** Dettmers et al. (2022) — *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*
- **GPTQ:** Frantar et al. (2022) — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
- **SmoothQuant:** Xiao et al. (2022) — *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*
- **QLoRA:** Dettmers et al. (2023) — *QLoRA: Efficient Finetuning of Quantized LLMs*
- **AWQ:** Lin et al. (2023) — *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*
- **HQQ:** Badri & Shaji (2023) — *Half-Quadratic Quantization of Large Machine Learning Models*
- **AQLM:** Egiazarian et al. (2024) — *AQLM: Extreme Compression of LLMs via Additive Quantization*
- **QuIP#:** Tseng et al. (2024) — *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*
- **BitNet b1.58:** Ma et al. (2024) — *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*

---

## 📄 Contributing & License

Contributions, bug reports, and new quantization methods are welcome! Please open an issue or submit a pull request.

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
