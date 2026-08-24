# Preference Pair Bias Report

- Input: `results/preference_diagnostics/current_rank_pairs/pairs.jsonl`
- SHA-256: `320a3f53664a6e171e02c17009764eb80452d413e04beb6e3e6e85796a9a905f`
- Pairs: 485
- Pair types: `{"H2": 485}`

## Chosen versus rejected features

| Feature | Chosen mean | Rejected mean | Difference |
|---|---:|---:|---:|
| chars | 47.6289 | 46.9216 | +0.7072 |
| tokens_whitespace | 9.1340 | 9.0082 | +0.1258 |
| words | 9.1649 | 9.0289 | +0.1361 |
| emoji_count | 0.0000 | 0.0000 | +0.0000 |
| exclamation_count | 0.0454 | 0.0660 | -0.0206 |
| question_count | 0.1423 | 0.1876 | -0.0454 |
| quote_count | 0.0412 | 0.0969 | -0.0557 |
| pov | 0.0000 | 0.0000 | +0.0000 |
| bro | 0.0000 | 0.0000 | +0.0000 |
| meanwhile | 0.0000 | 0.0000 | +0.0000 |
| internet_slang | 0.0000 | 0.0000 | +0.0000 |
| lexical_diversity | 0.9676 | 0.9763 | -0.0087 |

## Potential shortcuts

No heuristic feature crossed the configured relative-difference threshold.

These checks diagnose surface confounders; they do not establish that a pair is visually grounded or humorously valid.

## Authoritative methodological references

- Rafailov et al. (2023), Direct Preference Optimization, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
- Wang et al. (2024), mDPO: Conditional Preference Optimization for Multimodal Large Language Models: https://arxiv.org/abs/2406.11839
