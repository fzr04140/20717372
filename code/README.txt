# Bayesian Belief Network + LLM Semantic CPT Fusion
# Supplementary Code Documentation

This repository contains all scripts used to preprocess the UK Road Safety dataset,
construct the Bayesian Belief Network (BBN), generate semantic CPTs using a Large
Language Model (LLM), perform selective CPT fusion, and evaluate predictive
performance using PySMILE.

The scripts are organized following the workflow described in the dissertation.

---

## 1. Environment Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- pysmile (with pysmile_license.py)
- openai / httpx (optional, for LLM API)
- tqdm

Place `pysmile.pyd` and `pysmile_license.py` in the same folder as the scripts.

---

## 2. File Overview

### 1. `data_process.py`
Preprocesses the raw UK accident datasets (collision, vehicle, casualty) and merges
them into a single modeling dataset.

### 2. `Rebinning.py`
Handles discretization, category merging, and binning of selected features.

### 3. `BBN_cpt.py`
Extracts CPTs from a trained GeNIe `.xdsl` file and exports them to CSV.

### 4. `llm_prompts.py`
Generates natural-language prompts for each CPT parent–state combination.

### 5. `llm_generate_cpt.py`
Sends prompts to the LLM (e.g., ChatGPT-4o), parses returned probabilities,
normalizes them, and outputs the Semantic CPT (SCPT).

### 6. `norm_results_llm.py`
Cleans, validates, and standardizes probability outputs from the LLM.

### 7. `n_smaples.py`
Counts empirical sample support `n(i)` for every CPT row using the 2024 dataset.

### 8. `norm_results_fused.py`
Performs percentile-based selective CPT fusion:
BBN_CPT_base, LLM_SCPT → fused CPT table.

### 9. `batch XDSL builder.py`
Rebuilds `.xdsl` files automatically for each fusion percentile.

### 10. `forecast.py`
Loads a fused BBN model using PySMILE and performs inference on the 2023 dataset.

### 11. `Visualization1.py`
Plots Macro-F1, Balanced Accuracy, and class recall across fusion percentiles.

### 12. `Visualization2.py`
Generates comparison charts .

### 13. `Gantt Table.py`
Produces the Gantt chart for the project timeline.

---

## 3. Running Order (Recommended Pipeline)

To fully reproduce the results:

1. `data_process.py`  
2. `Rebinning.py`  
3. `BBN_cpt.py`  
4. `llm_prompts.py`  
5. `llm_generate_cpt.py`  
6. `norm_results_llm.py`  
7. `n_smaples.py`  
8. `norm_results_fused.py`  
9. `batch XDSL builder.py`  
10. `forecast.py`  
11. `Visualization1.py`  
12. `Visualization2.py`  

---

## 4. Input Files Required

- `dft-road-casualty-statistics-collision-2024.csv`, `dft-road-casualty-statistics-vehicle-2024.csv`, `dft-road-casualty-statistics-casualty-2024.csv`
- `dft-road-casualty-statistics-collision-2023.csv`, etc. (for evaluation)
- `genie_ready_2024.xdsl` (baseline BBN)
- `pysmile.pyd` and `pysmile_license.py`

---

## 5. Notes

- Ensure API keys are configured before running LLM query scripts.
- Some scripts may take several minutes depending on CPT size.
- PySMILE must be properly licensed before evaluation.

---


