# GPU-Accelerated LLM Inference & Performance Benchmarking

A practical benchmarking project for evaluating **LLM inference latency, throughput, batching behavior, device selection, and memory usage** across local execution environments.

This project currently runs on **Apple Silicon MPS**, with the architecture designed to support future extensions for **CUDA, TensorRT-LLM, vLLM, SGLang, quantization, and multi-GPU inference**.

---

## Overview

The goal of this project is to make LLM inference performance measurable and reproducible.

The benchmark pipeline:

1. Loads a Hugging Face causal language model
2. Detects the best available device
3. Tokenizes one or more prompts
4. Executes generation
5. Measures latency
6. Calculates tokens-per-second throughput
7. Tracks process memory usage
8. Stores benchmark results in CSV format

---

## Current Features

- PyTorch-based LLM inference
- Hugging Face Transformers integration
- Apple Silicon MPS support
- CPU fallback
- CUDA detection
- Batch-size benchmarking
- Latency measurement
- Tokens-per-second throughput measurement
- Process memory monitoring
- CSV benchmark logging
- Docker support
- Pytest-based tests

---

## Architecture

```text
Prompt
  ↓
Tokenizer
  ↓
Transformer Model
  ↓
Inference Runtime
  ↓
CPU / Apple MPS / CUDA
  ↓
Performance Measurement
  ↓
Latency | Throughput | Memory | Batch Size
  ↓
CSV Results
