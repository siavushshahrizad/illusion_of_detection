# The Illusion of Detection

Code accompanying "The Illusion of Detection: Why Machine Learning Can Overestimate Mental Disorder Recognition from Audio". Across a re-evaluation of 25 published studies and our own robust analysis of a large, naturalistic dataset, we found no reliable evidence that mental disorders can be detected from acoustic markers in audio, once common methodological flaws were corrected for.

📄 Manuscript: [TODO Add link]

## Repo Structure

```
constants.py                # Paths, hyperparameters, variable names
correlations.py             # Spearman's correlations for outcomes and static vars
knn.py                      # Data leakage simulation
loaders.py                  # Data loading for outcomes and inputs
loaders_torch.py            # PyTorch Dataset/DataLoader 
lstm.py                     # Implements a long short-term memory neural net
processing.py               # Normalisation and class weighting, KNN leakage simulation helpers
README.md                   # Project info
requirements.txt            # Dependencies
static_classifier.py        # Static classifiction, e.g. random forest, for outcomes
time_series_classifier.py   # Time-series classifcation of outcomes
tests/                      # Unit tests
```
## Usage

Each script accepts `-o {depression, anxiety, loneliness}` to select the outcome, and potentially some other params, e.g. input task.

```bash
pip install -r requirements.txt

# Static classifiers
python static_classifier.py -o depression

# Time-series (LSTM) model
python time_series_classifier.py -o depression -i rainbow

# Tests
pytest tests/
```

## Data

Our robust analysis uses the [Bridge2AI-Voice dataset](https://physionet.org/content/b2ai-voice/3.0.0/) (v3.0). This dataset is **not included in this repository** and requires a separate credentialed access request via PhysioNet or [healthdatanexus.ai](https://healthdatanexus.ai). This code expects the data to be in a `data/` folder at the root level. See `constants.py` for exact filenames. The 25 re-evaluated studies from the systematic review are listed with our audit results in the manuscript (Tables 1–4); no raw data from those studies is used or redistributed here.

## Citation

```
[TODO: Add BibTeX once preprinted]
```

## License

MIT — code only. The Bridge2AI dataset is governed by its own separate data use agreement.
