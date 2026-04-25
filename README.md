# NeuroSpell: Translating EEG Signals into Text Using Machine Learning

## What it Does

We were curious about one question: how can brain signals actually be turned into something useful? Specifically, we wanted to see for ourselves whether EEG, the electrical signals your brain produces, could be used to spell words, just by thinking about letters on a screen. This led us to the P300 BCI speller, a classic paradigm in neuroscience where a person watches a grid of letters flashing on a screen, and their brain produces a small signal called the P300 at around 300ms after they see a letter that they want to spell. We built a full pipeline that takes raw EEG recordings, cleans and processes the signals, trains machine learning models to detect that P300 response, and ultimately decodes which character the person intended, turning brain waves into text.

## Research Question/Goal

Can machine learning models reliably detect the P300 brainwave response from noisy EEG data well enough to decode intended characters and how do classical approaches like LDA compare to deep learning ones like CNNs in this setting?
This question is grounded in BCI research, most notably Farwell & Donchin (1988), who first demonstrated that P300 signals could be used to spell words non-invasively. We wanted to reproduce and explore this pipeline ourselves using a modern open dataset (bigP3BCI, PhysioNet 2025) and compare how well different ML approaches hold up across participants with varying signal quality.

## Quick Start

Please see [SETUP.md](./SETUP.md) for full installation instructions and requirements.

Once set up, you will see the following project structure: 
 
 `src/` — Core source code shared across the project. Contains `Dataset.py` for data loading and preprocessing logic, and `Speller.py` for spelling decoding utility  used in the pipeline.
 
`models/` — Standalone Python implementations of each classifier. Includes `Baseline.py`, `LDA.py`, and `P300NN.py`, along with `_noClassBalance` variants of LDA and the P300 neural network that skip class rebalancing for preprocessing impact verification.
 
`notebooks/` — Jupyter notebooks for exploration, experimentation, and final evaluation. Start with `Init_Data_Analysis.ipynb` to understand how data is loaded and visualized. From there, explore `LDAFinal.ipynb` and `P300NNFinal.ipynb` for per-model results, the hyperparameter tuning and preprocessing test notebooks for experiment history, `ErrorAnalysis.ipynb` for a breakdown of failure cases, and finally `FinalEvaluation.ipynb` for the full end-to-end results of our experiment.
 
`data/` — Contains the raw and labelled EEG dataset files used for training and evaluation.

## Video Links

Video Links are included below as the files exceeded GitHub's filesize limit. 

| Video | Link |
|---|---|
| Demo Video | https://youtu.be/lsSXlFjFkb4 |
| Technical Walkthrough | <link> |

## Evaluation

We evaluated our models using three metrics that directly reflect whether the system can decode brain signals into the correct letters:

- AUC-ROC — how well the model separates target from non-target EEG epochs
- Model Classification Accuracy — percentage of correctly identified P300 events
- Character/Spelling Accuracy — percentage of characters correctly decoded end-to-end

We compared three approaches — a constant baseline, an LDA classifier, and a custom CNN (P300NN) — all trained and tested on the same data and preprocessing pipeline. Full results are in `notebooks/FinalEvaluation.ipynb`.

## Individual Contributions

| Contributor | Contributions |
|---|---|
| Arnav Gupta (ag796) | Custom CNN (P300NN), GPU training, hyperparameter tuning, optimizer selection, preprocessing pipeline, error analysis |
| Arjun Saha Choudhury (as1572) | LDA model, baseline model, Dataset class, Speller decoding pipeline, evaluation notebooks, qualitative analysis |
