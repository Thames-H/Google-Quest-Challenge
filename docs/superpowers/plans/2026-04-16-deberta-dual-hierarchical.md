# Google QUEST DeBERTa Dual Hierarchical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current RoBERTa dual-tower baseline with a DeBERTa-v3 dual-branch hierarchical pipeline that supports external data directories, Linux training, rank-aware loss, metadata features, and rank-based distribution matching.

**Architecture:** Keep the two-branch QUEST formulation, but upgrade each branch from single truncation to chunked document encoding plus hierarchical pooling. Add lightweight metadata features, mixed pointwise/ranking loss, and a post-processing stage that maps prediction ranks to the empirical target distributions.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, pandas, NumPy, scikit-learn, PyYAML, pytest

---

### Task 1: Lock Behavior With Tests

**Files:**
- Create: `F:\闈㈠challenge\google-quest-challenge\tests\test_hierarchical_pipeline.py`

- [ ] **Step 1: Write failing tests for the new behavior**
- [ ] **Step 2: Run the targeted pytest file and confirm failure**
- [ ] **Step 3: Cover external data-dir resolution, normalized GroupKFold keys, chunk batching, metadata extraction, mixed loss, and distribution matching**

### Task 2: Refactor Configuration And Data Loading

**Files:**
- Modify: `F:\闈㈠challenge\google-quest-challenge\quest\config.py`
- Modify: `F:\闈㈠challenge\google-quest-challenge\quest\data.py`

- [ ] **Step 1: Add DeBERTa-centric config fields for chunk sizes, chunk limits, metadata, ranking loss, and postprocess toggles**
- [ ] **Step 2: Add config resolution order for `--data-dir`, `QUEST_DATA_DIR`, then YAML default**
- [ ] **Step 3: Build normalized question grouping and external metadata extraction helpers**

### Task 3: Introduce Hierarchical Model And Losses

**Files:**
- Modify: `F:\闈㈠challenge\google-quest-challenge\quest\model.py`
- Create: `F:\闈㈠challenge\google-quest-challenge\quest\losses.py`
- Create: `F:\闈㈠challenge\google-quest-challenge\quest\postprocess.py`

- [ ] **Step 1: Add chunk-level masked pooling and branch-level attention pooling**
- [ ] **Step 2: Add metadata projection/embedding support**
- [ ] **Step 3: Implement pointwise plus margin-ranking mixed loss**
- [ ] **Step 4: Implement rank-based distribution matching helpers**

### Task 4: Rebuild Training And Prediction Pipeline

**Files:**
- Modify: `F:\闈㈠challenge\google-quest-challenge\quest\pipeline.py`
- Modify: `F:\闈㈠challenge\google-quest-challenge\train.py`
- Modify: `F:\闈㈠challenge\google-quest-challenge\predict.py`
- Modify: `F:\闈㈠challenge\google-quest-challenge\quest\__init__.py`

- [ ] **Step 1: Add collate support for variable chunk counts**
- [ ] **Step 2: Train with DeBERTa-v3-base, hierarchical pooling, metadata features, and mixed loss**
- [ ] **Step 3: Preserve checkpoint reuse and ensemble prediction**
- [ ] **Step 4: Add optional distribution matching during inference**

### Task 5: Add Linux-Focused Configs And Verify

**Files:**
- Modify: `F:\闈㈠challenge\google-quest-challenge\requirements.txt`
- Add: `F:\闈㈠challenge\google-quest-challenge\configs\deberta_dual_hierarchical_linux.yaml`
- Add: `F:\闈㈠challenge\google-quest-challenge\configs\deberta_dual_hierarchical_smoke.yaml`
- Add: `F:\闈㈠challenge\google-quest-challenge\docs\linux-training.md`

- [ ] **Step 1: Add tokenizer/runtime dependencies such as `sentencepiece`**
- [ ] **Step 2: Add smoke and Linux configs with external data-dir defaults**
- [ ] **Step 3: Run targeted pytest verification**
- [ ] **Step 4: Run a smoke train/predict loop locally if runtime allows**
