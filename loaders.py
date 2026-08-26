"""File contains various data loaders."""

import pandas as pd
import numpy as np
from constants import (
    MFCC_FILE, 
    PHQ_FILE,
    STATIC_FILE,
    LONELY_FILE,
    ANXIETY_FILE,
    TASK,
    MFCC, 
    ID,
    DEPRESSION,
    LONELINESS_MODERATE,
    LONELINESS_SEVERE,
    ANXIETY_MODERATE,
    ANXIETY_SEVERE,
)


MAPPING = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3
}

def load_mfcc(tasks: list, min_frames: int = 500) -> pd.DataFrame:
    """Loads mfcc data with respondent deduplication."""
    df = pd.read_parquet(MFCC_FILE)

    assert not df[ID].isna().any(), "Missing ID"

    # Oddly, in v3 of the datset ID is a str
    df[ID] = df[ID].astype(int)

    DURATION = "n_frames"

    # tasks = [
    df = df[df[TASK].isin(tasks)]

    # Filter out less than min_durations 
    # e.g. if min_duration is 1000, this means 
    # recoding must have thousand frames, which 
    # is roughly 10s
    # Numb of 500 frames seemed right based on 
    # distribution in rainbow task.
    df = df[df[DURATION] >= min_frames]

    # Drop any missing value rows
    df.dropna(inplace=True)

    # Transpose
    df[MFCC] = df[MFCC].apply(lambda x: np.stack(x).T)

    # Remove duplicates
    df = df.drop_duplicates(subset=ID)

    return df

def load_phq() -> pd.DataFrame:
    """Loads depression data with respondent deduplication."""
    df = pd.read_csv(PHQ_FILE, sep='\t')

    PHQ_Q_1 = "no_interest"       
    PHQ_Q_2 = "feeling_depressed"
    PHQ_Q_3 = "trouble_sleeping"
    PHQ_Q_4 = "no_energy"
    PHQ_Q_5 = "no_appetite"
    PHQ_Q_6 = "feeling_bad_self"
    PHQ_Q_7 = "trouble_concentrate"
    PHQ_Q_8 = "move_speak_slow"
    PHQ_Q_9 = "thoughts_death"

    PHQ_measures = [
        PHQ_Q_1, 
        PHQ_Q_2, 
        PHQ_Q_3, 
        PHQ_Q_4, 
        PHQ_Q_5, 
        PHQ_Q_6, 
        PHQ_Q_7,
        PHQ_Q_8,
        PHQ_Q_9
    ]
    
    # Check no missing values
    assert not df[PHQ_measures].isna().any().any(), "Missing PHQ responses"
    assert not df[ID].isna().any(), "Missing ID"
    
    # Convert to numeric
    for measure in PHQ_measures:
        df[measure] = df[measure].map(MAPPING)
    
    # Not enough people to divide by severe depression
    SUM = "sum_phq"
    MODERATE_THRESHOLD = 9 
    
    # Create binary outcome
    df[SUM] = df[PHQ_measures].sum(axis=1)  # Colwise
    df[DEPRESSION] = (df[SUM] > MODERATE_THRESHOLD).astype(int)

    # Remove any participant with duplicate PHQ reading
    duplicates = df[ID].duplicated(keep=False)
    df = df[~duplicates]

    # Removed unwanted cols
    unwanted = [
        'phq_9_duration', 
        'phq_9_session_id',
        'hard_to_work'
    ]

    df.drop(columns=unwanted, inplace=True)
    df.drop(columns=PHQ_measures, inplace=True)

    return df

def load_static(tasks: list, static_vars: list) -> pd.DataFrame:
    """Loads static data with respondent deduplication."""
    df = pd.read_csv(STATIC_FILE, sep='\t')
    assert not df[ID].isna().any(), "Missing ID"

    DURATION = "duration"

    df = df[df[TASK].isin(tasks)]

    static_vars_cp = static_vars[:]     # Else mutates constant and error later
    static_vars_cp.extend((ID, TASK, DURATION))

    df = df[static_vars_cp]

    # Filter out less than 10s recordings 
    # Note no one above 80s
    df = df[df[DURATION] >= 10]     # Comment: Seems roughly right
    df.drop(columns=[DURATION], inplace=True)

    # Drop any missing value rows
    df.dropna(inplace=True)

    # Deduplicate
    df = df.drop_duplicates(subset=ID)

    return df
 
def load_loneliness() -> pd.DataFrame:
    """Loads loneliness data."""
    df = pd.read_csv(LONELY_FILE, sep='\t')
    df.dropna(inplace=True)
    LONELY = "lonely"
    
    duplicates = df[ID].duplicated(keep="first")            # Note first duplicate kept as similar scores
    df = df[~duplicates]
    MODERATE_THRESHOLD = 5 
    SEVERE_THRESHOLD = 7 

    df[LONELINESS_MODERATE] = (df[LONELY] >= MODERATE_THRESHOLD).astype(int)
    df[LONELINESS_SEVERE] = (df[LONELY] >= SEVERE_THRESHOLD).astype(int)
    
    df = df[[ID, LONELY, LONELINESS_MODERATE, LONELINESS_SEVERE]]
    
    return df

def load_anxiety() -> pd.DataFrame:
    """Loads anxiety data."""
    df = pd.read_csv(ANXIETY_FILE, sep='\t')

    GAD_Q_1 = "afraid_of_things"       
    GAD_Q_2 = "cant_control_worry"
    GAD_Q_3 = "easily_agitated"
    GAD_Q_4 = "hard_to_sit_still"
    GAD_Q_5 = "nervous_anxious"
    GAD_Q_6 = "trouble_relaxing"
    GAD_Q_7 = "worry_too_much"

    GAD_measures = [
        GAD_Q_1, 
        GAD_Q_2, 
        GAD_Q_3, 
        GAD_Q_4, 
        GAD_Q_5, 
        GAD_Q_6, 
        GAD_Q_7,
    ]
    
    # Check no missing values
    assert not df[GAD_measures].isna().any().any(), "Missing GAD responses"
    assert not df[ID].isna().any(), "Missing ID"
    
    # Convert to numeric
    for measure in GAD_measures:
        df[measure] = df[measure].map(MAPPING)
    
    # Not enough people to divide by severe depression
    SUM = "sum_gad"
    MODERATE_THRESHOLD = 9
    SEVERE_THRESHOLD = 15
    
    # Create binary outcome
    df[SUM] = df[GAD_measures].sum(axis=1)  # Colwise
    df[ANXIETY_MODERATE] = (df[SUM] > MODERATE_THRESHOLD).astype(int)
    df[ANXIETY_SEVERE] = (df[SUM] > SEVERE_THRESHOLD).astype(int)

    # Remove any participant with duplicate GAD reading
    duplicates = df[ID].duplicated(keep=False)
    df = df[~duplicates]

    # Removed unwanted cols
    unwanted = [
        'gad_7_duration', 
        'gad_7_session_id',
        'tough_to_work'
    ]

    df.drop(columns=unwanted, inplace=True)
    df.drop(columns=GAD_measures, inplace=True)

    return df
