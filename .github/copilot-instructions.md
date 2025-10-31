<!-- Copilot Instructions for contributors and AI coding agents -->
# Quick orientation for automated coding agents

This repository implements Shapley-style attribution for LLM context items (the SHapRAG toolset). The goal of this document is to give short, actionable guidance so an AI coding agent can be productive immediately.

Key components
- `SHapRAG/` — core library. Major files:
  - `rag_shap.py` — primary implementation (ContextAttribution class). Contains LLM interactions, utility caching, Shapley computations, sampling methods, and efficient approximations. Use this file as the first reference for changing attribution logic or LLM interfacing.
  - `utils.py` — supportive helpers used by experiments and scripts.
  - `evaluator.py` — evaluation glue (currently minimal/placeholder).
- `scripts/` — runnable experiments and dataset-specific scripts (e.g., `bioask_*`, `hotpotQA*`). Look here to see expected CLI inputs and experiment wiring.
- `Notebooks/` — exploratory analysis and example usages (data prep, dynamic ranking, synthetic experiments). Good for reference examples but not authoritative for production code.
- `data/` and `Experiment_data/` — datasets, synthetic examples, and experiment outputs. Tests and experiments expect particular CSV/JSON formats present here.

Important patterns and conventions
- LLM caller interface: code expects a function/method that returns a log-probability or generated text. In `ContextAttribution` the model/tokenizer pair (HuggingFace-style) are passed in and used directly. When changing LLM calls, preserve the contract:
  - Generation: _llm_generate_response(context_str) -> str (used to create a target response).
  - Utility: _compute_response_metric(context_str, mode, response=None) -> float (returns log-prob or other metric).
  - Keep tokenization/generation consistent with `tokenizer.apply_chat_template(...)` usage in `rag_shap.py`.
- Accelerator/Multiprocess: `ContextAttribution` uses `accelerate.Accelerator` and broadcasts cached objects. Make changes mindful of main-process guards: file reads/writes and generation occur only on `accelerator.is_main_process` then broadcast to other processes.
- Caching: `utility_cache` is a nested dict mapping subset tuples -> {mode: utility}. Persistence uses pickle and is only performed by main process via `save_utility_cache()`; follow same pattern when adding durable caching.
- Modes: utilities support modes like `log-prob`, `raw-prob`, `logit-prob`, `log-perplexity`, `divergence_utility`. If adding a new mode, implement in `_compute_response_metric` and ensure callers of `get_utility` pass a valid `mode` string.

Developer workflows and commands
- Python environment: `requirements.txt` lists core dependencies. Use a venv or devcontainer. Notable dependencies: `transformers`, `accelerate`, `torch`, `scipy`, `sklearn`, `fastFM`, `spectralexplain`.
- Docker Compose: `docker-compose.yaml` configures local services used in experiments (Ollama on port 11434 and Elasticsearch). Start services when reproducing retrieval or LLM-backed experiments:
  - Start: docker compose up -d
  - Healthchecks: Ollama API at http://localhost:11434/api, Elasticsearch at http://localhost:9200
- Running experiments: `scripts/` contains many runnable scripts. They are not unified under a single CLI — inspect the target script to know required args (common pattern: dataset path, cache path, model name/tokenizer). Example: `scripts/test.py` or `scripts/llm_eval.py` show how the code constructs model/tokenizer and calls `ContextAttribution`.

Testing and safety gates
- There are no formal unit tests present. When making changes that affect model I/O or tensor shapes, test locally with a small set of `data/musique/...` examples or synthetic subsets under `data/synthetic_data/` for quick smoke runs.
- Lint/format: follow existing file style (conservative imports at top of `rag_shap.py`, explicit device and accelerator usage, avoid reformatting large files unnecessarily).

Common pitfalls and gotchas (do not guess — check code)
- Tokenizer/chat template: `tokenizer.apply_chat_template(...)` is used in multiple places. Changing prompt composition must be done consistently for both generation and probability evaluation.
- Memory: the code uses explicit `torch.cuda.empty_cache()` in many places. When adding loops or batched computation, preserve or improve memory handling to avoid OOMs.
- Broadcast and main process I/O: file loading/saving and model.generate are done only on `accelerator.is_main_process`. If you add new side-effects (logs, temp files), guard them similarly and use `broadcast_object_list` where appropriate.
- Subset encoding: subsets are represented as tuples of 0/1 integers (length == n_items) and used as dict keys. Keep this exact shape for cache keys and for any external serialization.

Examples to copy/paste
- Computing Shapley (exact) uses:
  - Instantiate with a prepared HF model/tokenizer pair and call the public helper in `ContextAttribution` (see class header in `SHapRAG/rag_shap.py`).
- Persisting cache: call `save_utility_cache(path)` only from main process; the method already enforces this guard.

Where to look when things break
- If LLM outputs/token shapes are wrong: inspect `_compute_response_metric`, `_get_response_token_distributions`, and `_llm_generate_response` in `rag_shap.py`.
- If multiprocess hangs: check `Accelerator` usage, `broadcast_object_list`, and `wait_for_everyone()` calls.
- If experiments fail to reproduce retrieval: check `docker-compose.yaml` services (Ollama/Elasticsearch) and dataset paths under `data/`.

If you need more context
- Start with `README.md` and `SHapRAG/rag_shap.py` (this is the authoritative implementation). Then inspect `scripts/` for runnable examples.

Ask for clarifications or missing pieces (model config, exact experiment commands) so the instructions can be refined.
