# Data and code for the paper "Learning to see the system: Perceived learning and perception language in Steam reviews of cooperative commercial games".

This repository includes:
- a data folder containing the English Steam reviews of the six analysed games.
- an `extraction_and_analysis.py` script that loads the data (or downloads it from Steam), parses it, and creates a pre-annotated corpus.
- a `LLM_annotation` Jupyer notebook, designed to be used on Google Colab, that queries Unsloth's quantized version of Qwen3-14B to annotate the data and download the updated corpus.
- an `annotated_corpus.json` which contains the final data.

Please cite as:
> Ciletti, M. (2026). Learning to see the system: Perceived learning and perception language in Steam reviews of cooperative commercial games [Data Set]. GitHub.
