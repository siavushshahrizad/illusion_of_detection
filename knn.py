"""
File simulates speaker leakage. Specifically,
it does the following:
    1) Takes a random sample of train/test
       who are different people across sets.
       Model will be trained on first half
       of rainbow task mfcc, and in the test
       setting uses the B part to predict 
       depression. This is the non-leakage
       condition.
    2) In the leakage condition, model is 
       trained on A part, and the test set 
       is the B part of the same people.
"""

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from loaders import load_mfcc, load_phq
from processing import (
    compute_mean_and_std,
    normalise_df,
    compress_left,
    compress_right
)
from constants import (
    RAINBOW_TASK,
    ID, 
    LEFT,
    RIGHT,
    MFCC,
    DEPRESSION,
    SEEDS,
    TRAIN_SIZE
)


if __name__ == "__main__":
    # Load/process data
    phq = load_phq()
    mfcc = load_mfcc(RAINBOW_TASK)

    df_orig = pd.merge(
        phq,
        mfcc,
        how="inner",
        on=ID
    )

    # Results storage
    highest_acc = 0
    lowest_acc = float('inf')

    avg_acc = 0 
    avg_acc_leak = 0
    avg_sensitivity = 0
    avg_sensitivity_leak = 0
    avg_specificity = 0
    avg_specificity_leak = 0
    avg_ppv = 0
    avg_ppv_leak = 0
    avg_npv = 0
    avg_npv_leak = 0
    
    # Runs
    progress_bar = tqdm(total=SEEDS)

    for seed in range(SEEDS):
        df = df_orig.copy()

        # Train/test split
        train_idx, test_idx = train_test_split(
            df.index,
            train_size=TRAIN_SIZE,
            random_state=seed,
            stratify=df[DEPRESSION]
        )

        # Normalise - although here it doesn't matter much
        mean, std = compute_mean_and_std(df.iloc[train_idx]) # Only use train or leakage!!!
        normalise_df(df, mean, std)

        df[LEFT] = df[MFCC].apply(compress_left)
        df[RIGHT] = df[MFCC].apply(compress_right)

        # Condition I: no leakage simulation
        X_train = df[LEFT].loc[train_idx].tolist()  # Series struct causes problems at least in X
        y_train = df[DEPRESSION].loc[train_idx].tolist()
        rus = RandomUnderSampler(random_state=seed)
        X_res, y_res = rus.fit_resample(X_train, y_train)

        X_test = df[RIGHT].loc[test_idx].tolist()
        y_test = df[DEPRESSION].loc[test_idx].tolist()

        neigh = KNeighborsClassifier()
        neigh.fit(X_res, y_res)

        preds = neigh.predict(X_test)
        result = classification_report(
            y_test, 
            preds,
            output_dict=True
        )
        if result["accuracy"] > highest_acc:
            highest_acc = result["accuracy"]

        if result["accuracy"] < lowest_acc:
            lowest_acc = result["accuracy"]

        avg_acc += result["accuracy"]
        avg_sensitivity += result["1"]["recall"]
        avg_specificity += result["0"]["recall"]
        avg_ppv += result["1"]["precision"]
        avg_npv += result["0"]["precision"]

        # Condition II: leakage
        _, leak_idx = train_test_split(
           train_idx,
           test_size=len(test_idx),
           random_state=seed,
           stratify=df[DEPRESSION].loc[train_idx]
        )

        X_test = df[RIGHT].loc[leak_idx].tolist()
        y_test = df[DEPRESSION].loc[leak_idx].tolist()

        preds = neigh.predict(X_test)
        result = classification_report(
            y_test, 
            preds,
            output_dict=True
        )
        avg_acc_leak += result["accuracy"]
        avg_sensitivity_leak += result["1"]["recall"]
        avg_specificity_leak += result["0"]["recall"]
        avg_ppv_leak += result["1"]["precision"]
        avg_npv_leak += result["0"]["precision"]

        progress_bar.update(1)
    
    progress_bar.close()
    
    print("Highest acc:", highest_acc)
    print("Lowest acc:", lowest_acc)
    print()
    avg_acc /= SEEDS
    avg_acc_leak /= SEEDS
    avg_sensitivity /= SEEDS    
    avg_sensitivity_leak /= SEEDS    
    avg_specificity /= SEEDS
    avg_specificity_leak  /= SEEDS
    avg_ppv /= SEEDS
    avg_ppv_leak  /= SEEDS
    avg_npv /= SEEDS
    avg_npv_leak  /= SEEDS
    avg_youden = avg_sensitivity + avg_specificity - 1
    avg_youden_leak = avg_sensitivity_leak + avg_specificity_leak - 1

    print(f"AVG Accuracy: {avg_acc}")
    print(f"AVG Accuracy leak: {avg_acc_leak}")
    print(f"AVG Sensitivity: {avg_sensitivity}")
    print(f"AVG Sensitivity leak: {avg_sensitivity_leak}")
    print(f"AVG Specificity: {avg_specificity}")
    print(f"AVG Specificity leak: {avg_specificity_leak}")
    print(f"AVG PPV: {avg_ppv}")
    print(f"AVG PPV leak: {avg_ppv_leak}")
    print(f"AVG NPV: {avg_npv}")
    print(f"AVG NPV leak: {avg_npv_leak}")
    print(f"AVG Youden: {avg_youden}")
    print(f"AVG Youden leak: {avg_youden_leak}")
    print("--------------------")
