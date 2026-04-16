# CollapseTracker: An Empirical Dataset Documenting Progressive Model Collapse Across Recursive Generations of Synthetic Training Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19511599.svg)](https://doi.org/10.5281/zenodo.19511599)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Authors

- **RAMKUMAR R** — Department of Artificial Intelligence and Data Science (Medical Engineering), Amrita Vishwa Vidyapeetham, Coimbatore Campus
- **PRAGALYA M** — Department of Artificial Intelligence and Data Science (Medical Engineering), Amrita Vishwa Vidyapeetham, Coimbatore Campus

## Abstract

Model collapse — the progressive degradation of model quality when recursively trained on synthetic data — has emerged as a critical concern for the sustainability of AI development. While theoretical analyses and isolated experiments have demonstrated the phenomenon, no open-access dataset exists that systematically documents the progression of collapse across multiple generations, model sizes, and data mixing ratios.

We present **CollapseTracker**, an empirical dataset produced by recursively fine-tuning GPT-2 (124M) and DistilGPT-2 (82M) across 10 synthetic data generations on 3 seed datasets (Wikipedia abstracts, creative fiction, and technical Q&A). At each generation, we measure and record: text diversity (distinct n-grams, Self-BLEU), distributional drift (KL divergence from the original), vocabulary coverage, rare-token survival rates, repetition rates, and perplexity. We also vary the synthetic-to-real data mixing ratio (100%, 75%, 50%, 25%) to document how data contamination thresholds affect collapse velocity.

Our dataset comprises fine-tuned model checkpoints, generated text samples across all generations, and comprehensive quality metrics across all 240 experimental conditions. CollapseTracker is released under CC-BY 4.0 to provide the first reproducible, open-access empirical foundation for model collapse research.

## Research Gap

> "Everyone talks about model collapse. Nobody has released a clean, controlled dataset showing exactly HOW it happens, step by step."

The 2024 Nature paper by Shumailov et al. demonstrated that models trained on AI-generated data progressively degrade. Despite massive research interest, **no open dataset tracks the actual progression of collapse** across generations with controlled experiments. CollapseTracker fills this gap.

## Dataset Overview

### Experimental Design

| Dimension | Values | Count |
|-----------|--------|-------|
| **Seed Domains** | Wikipedia abstracts, Creative fiction, Technical Q&A | 3 |
| **Models** | GPT-2 (124M), DistilGPT-2 (82M) | 2 |
| **Mixing Ratios** | 100% synthetic, 75/25, 50/50, 25/75 | 4 |
| **Generations** | 0–10 recursive cycles | 10 |
| **Total Tracks** | 3 × 2 × 4 | 24 |
| **Total Data Points** | 24 tracks × 10 generations | 240 |

### Metrics Tracked Per Generation

| Metric | Description | Collapse Indicator |
|--------|-------------|-------------------|
| `distinct_1gram` | Unique unigram ratio | ↓ = collapse |
| `distinct_2gram` | Unique bigram ratio | ↓ = collapse |
| `distinct_3gram` | Unique trigram ratio | ↓ = collapse |
| `self_bleu` | Average BLEU between generated pairs | ↑ = collapse |
| `kl_divergence` | KL divergence from seed distribution | ↑ = collapse |
| `vocab_coverage` | Fraction of seed vocabulary preserved | ↓ = collapse |
| `rare_token_survival` | Fraction of rare tokens still generated | ↓ = collapse |
| `repetition_rate` | Fraction of text containing repeated n-grams | ↑ = collapse |
| `perplexity` | Model perplexity on generated text | ↑ = collapse |
| `mean_length` | Average text length (tokens) | varies |
| `length_std` | Standard deviation of text length | ↓ = collapse |

## File Structure

```
CollapseTracker/
├── README.md                          # This file
├── LICENSE                            # CC-BY 4.0
├── experiment_config.yaml             # Full reproducibility config
├── requirements.txt                   # Python dependencies
├── .zenodo.json                       # Zenodo metadata
│
├── seed_datasets/                     # Curated seed data (3 domains)
│   ├── wikipedia_abstracts.jsonl      # 1000 Wikipedia abstract samples
│   ├── creative_fiction.jsonl         # 1000 creative fiction samples
│   └── technical_qa.jsonl            # 1000 technical Q&A samples
│
├── training_scripts/                  # All experiment code
│   ├── 01_prepare_seed_data.py        # Seed data curation
│   ├── 02_collapse_loop.py           # Recursive fine-tuning loop
│   ├── 03_compute_metrics.py         # Metric computation
│   ├── 04_visualize.py               # Publication-quality plots
│   ├── 05_statistical_analysis.py    # Statistical tests
│   ├── utils.py                       # Shared utilities
│   └── CollapseTracker_Full.ipynb    # All-in-one Colab notebook
│
├── generated_samples/                 # Generated text at each generation
│   └── {domain}_{model}_{ratio}/
│       ├── gen_00.jsonl
│       ├── gen_01.jsonl
│       └── ...
│
├── model_checkpoints/                 # Saved model weights
│   └── {domain}_{model}_{ratio}/
│       ├── gen_00/
│       ├── gen_05/
│       └── gen_10/
│
├── collapse_metrics.csv               # Primary metrics (240 rows)
├── collapse_metrics_detailed.csv      # Per-sample detailed metrics
│
├── collapse_visualizations/           # Publication-quality figures
│   ├── collapse_trajectory_all.png
│   ├── collapse_by_domain.png
│   ├── collapse_by_model.png
│   ├── collapse_by_ratio.png
│   ├── heatmap_collapse_onset.png
│   └── ...
│
└── paper/
    ├── CollapseTracker_Paper.tex      # Full research paper source
    ├── references.bib                 # Bibliography
    └── figures/                       # Paper figures
```

## Quick Start

### Option 1: Run the Complete Experiment (Google Colab)

1. Open `training_scripts/CollapseTracker_Full.ipynb` in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially
4. Results are saved to Google Drive automatically

### Option 2: Run Individual Scripts (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Prepare seed datasets
python training_scripts/01_prepare_seed_data.py

# Step 2: Run recursive training loop (requires GPU)
python training_scripts/02_collapse_loop.py

# Step 3: Compute detailed metrics
python training_scripts/03_compute_metrics.py

# Step 4: Generate visualizations
python training_scripts/04_visualize.py

# Step 5: Statistical analysis
python training_scripts/05_statistical_analysis.py
```

### Option 3: Just Explore the Data

```python
import pandas as pd

# Load collapse metrics
metrics = pd.read_csv("collapse_metrics.csv")

# View collapse trajectory for Wikipedia + GPT-2 + 100% synthetic
track = metrics[
    (metrics["domain"] == "wikipedia") & 
    (metrics["model"] == "gpt2") & 
    (metrics["mixing_ratio"] == 1.0)
]
print(track[["generation", "distinct_2gram", "self_bleu", "vocab_coverage"]])
```

## Methodology

### Recursive Training Loop

```
For each (seed_dataset, model, mixing_ratio):
    model_g0 = fine_tune(base_model, seed_dataset)
    
    For generation g in 1..10:
        synthetic_data = generate(model_{g-1}, n=1000)
        mixed_data = mix(synthetic_data, seed_dataset, ratio)
        model_g = fine_tune(base_model, mixed_data)
        metrics[g] = compute_all_metrics(synthetic_data, seed_dataset)
        save(model_g, synthetic_data, metrics[g])
```

### Key Design Decisions

1. **Fresh fine-tuning from base**: At each generation, we fine-tune from the original base model (not the previous fine-tuned model) on the mixed data. This isolates the effect of data quality degradation from accumulated weight drift.

2. **Fixed generation parameters**: Temperature=1.0, top-k=50, top-p=0.95 across all experiments for fair comparison.

3. **Deterministic seeding**: All random seeds are fixed for full reproducibility.

## Citation

If you use CollapseTracker in your research, please cite:

```bibtex
RAMKUMAR R, & PRAGALYA M. (2026). CollapseTracker — A Time-Lapse Dataset of Progressive Model Collapse [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19511599
```

## Related Work

- Shumailov, I., Shumaylov, Z., Zhao, Y., et al. (2024). "AI models collapse when trained on recursively generated data." *Nature*, 631, 755–759.
- Hataya, R., Bao, H., & Arai, H. (2023). "Will Large-scale Generative Models Corrupt Future Datasets?" *ICCV 2023*.
- Alemohammad, S., et al. (2024). "Self-Consuming Generative Models Go MAD." *ICLR 2024*.

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this dataset for any purpose, provided you give appropriate credit.

## Acknowledgments

This work was conducted as part of undergraduate research at the Department of Artificial Intelligence and Data Science (Medical Engineering), Amrita Vishwa Vidyapeetham, Coimbatore Campus. Compute resources provided by Google Colab.
