# The Illusion of Detection

Code accompanying "The Illusion of Detection: Why Machine Learning Can Overestimate Mental Disorder Recognition from Audio". Across a re-evaluation of 25 published studies and our own robust analysis of a large, naturalistic dataset, we found no reliable evidence that mental disorders can be detected from acoustic markers in audio, once common methodological flaws were corrected for.

📄 [Manuscript](https://doi.org/10.31224/8100)

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
@misc{shaIllusionDetectionWhy2026,
  title = {The {{Illusion}} of {{Detection}}: {{Why Machine Learning Can Overestimate Mental Disorder Recognition}} from {{Audio}}},
  shorttitle = {The {{Illusion}} of {{Detection}}},
  author = {Sha, Sia and Qualter, Pamela and Huang, Yongchao and Shahrizad, Nina and Krpan, Dario and Galizzi, Matteo},
  year = 2026,
  month = sep,
  doi = {10.31224/8100},
  urldate = {2026-09-07},
  abstract = {Machine learning (ML) research has attempted to detect mental disorders from people's audio recordings. Despite seemingly impressive reported results, it remains unclear whether those results reflect genuine clinical signal or artefacts related to how ML models are trained and evaluated. We re-examine 25 studies included in a recent systematic review and identify three major recurring methodological problems that are likely to inflate the ability of ML models to detect mental disorders: (1) data leakage, (2) lack of multiple seeds, and (3) lack of large, naturalistic datasets. We show via an experiment how ML practices can produce apparently strong results even in the absence of real, predictive signals. We also conduct robust analyses using a large, novel dataset, where we find no reliable evidence that mental disorders can be detected from audio across multiple modelling approaches. Our findings suggest that current evidence may substantially overestimate the real-world capability of audio-based detection of mental health disorders, highlighting the need for evaluation standards aligned with best practices for clinical and policy use.},
  copyright = {https://creativecommons.org/licenses/by/4.0},
}

```

## License

MIT — code only. The Bridge2AI dataset is governed by its own separate data use agreement.
