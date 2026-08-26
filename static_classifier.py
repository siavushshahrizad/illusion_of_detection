"""
File evaluates static models on mental
health outcomes.
"""

import sys
from pathlib import Path
root_folder = Path(__file__).parent.parent
sys.path.insert(0, str(root_folder))

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from loaders import (
    load_static, 
    load_phq,
    load_loneliness,
    load_anxiety
)
from constants import (
    ID, 
    FREE_TASKS,
    SEEDS,
    TRAIN_SIZE,
    DEPRESSION,
    ANXIETY_MODERATE,
    LONELINESS_MODERATE,
    STATIC_VARS
)


if __name__ == "__main__":
    # Parsing args
    parser = argparse.ArgumentParser(
        description="Run static classifiers on user defined outcome"
    )
    parser.add_argument(
        '-o', 
        '--outcome',
        type=str,
        default="depression",
        help="Specify what outcome classifiers should be evaluated on",
        choices=["depression", "loneliness", "anxiety"]
    )
    args = parser.parse_args()

    if args.outcome == "depression":
        outcome_data = load_phq()
        outcome_var = DEPRESSION
    elif args.outcome == "anxiety":
        outcome_data = load_anxiety()
        outcome_var = ANXIETY_MODERATE
    else:
        outcome_data = load_loneliness()
        outcome_var = LONELINESS_MODERATE

    # Load/process data
    static = load_static(FREE_TASKS, STATIC_VARS)

    df = pd.merge(
        outcome_data,
        static,
        how="inner",
        on=ID
    )

    progress_bar = tqdm(total=SEEDS)
    num_seed = 0        # To track results

    # Store results
    lr_acc = np.zeros(SEEDS)
    lr_sensitivity = np.zeros(SEEDS)
    lr_specificity = np.zeros(SEEDS)
    lr_ppv = np.zeros(SEEDS)
    lr_npv = np.zeros(SEEDS)

    rf_acc = np.zeros(SEEDS)
    rf_sensitivity = np.zeros(SEEDS)
    rf_specificity = np.zeros(SEEDS)
    rf_ppv = np.zeros(SEEDS)
    rf_npv = np.zeros(SEEDS)

    svm_acc = np.zeros(SEEDS)
    svm_sensitivity = np.zeros(SEEDS)
    svm_specificity = np.zeros(SEEDS)
    svm_ppv = np.zeros(SEEDS)
    svm_npv = np.zeros(SEEDS)

    for seed in range(SEEDS):
        # Train test split
        train_idx, test_idx = train_test_split(
            df.index,
            train_size=TRAIN_SIZE,
            random_state=seed,
            stratify=df[outcome_var]
        )

        X_train = df[STATIC_VARS].loc[train_idx]
        y_train = df[outcome_var].loc[train_idx]

        X_test = df[STATIC_VARS].loc[test_idx]        
        y_test = df[outcome_var].loc[test_idx]

        # Normalise
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_test_norm = scaler.transform(X_test)

        ### Logistic regression ###
        #                         #
        ###########################
        model = LogisticRegression(
            l1_ratio=0,    # L2 regularisation
            class_weight="balanced",
            random_state=seed,
            solver="liblinear"
        )

        model.fit(X_train_norm, y_train)
        preds = model.predict(X_test_norm)

        results = classification_report(
            y_test, 
            preds,
            output_dict=True
        )

        lr_acc[num_seed] = results["accuracy"]
        lr_sensitivity[num_seed] = results["1"]["recall"]
        lr_specificity[num_seed] = results["0"]["recall"]
        lr_ppv[num_seed] = results["1"]["precision"]
        lr_npv[num_seed] = results["0"]["precision"]

        ###    Random forest    ###
        #                         #
        ###########################
        model = RandomForestClassifier(
            min_samples_split=50,     
            min_samples_leaf=20,      
            max_features='sqrt',      
            class_weight='balanced',
            random_state=seed,
            n_estimators=100
        )

        model.fit(X_train_norm, y_train)
        preds = model.predict(X_test_norm)

        results = classification_report(
            y_test, 
            preds,
            output_dict=True
        )

        rf_acc[num_seed] = results["accuracy"]
        rf_sensitivity[num_seed] = results["1"]["recall"]
        rf_specificity[num_seed] = results["0"]["recall"]
        rf_ppv[num_seed] = results["1"]["precision"]
        rf_npv[num_seed] = results["0"]["precision"]

        ###    SVM      ###
        #                 #
        ###################
        model = SVC(
            kernel='rbf',
            class_weight='balanced',
            C=1.0,
            gamma='scale',
            random_state=seed,
        )

        model.fit(X_train_norm, y_train)
        preds = model.predict(X_test_norm)

        results = classification_report(
            y_test, 
            preds,
            output_dict=True
        )

        svm_acc[num_seed] = results["accuracy"]
        svm_sensitivity[num_seed] = results["1"]["recall"]
        svm_specificity[num_seed] = results["0"]["recall"]
        svm_ppv[num_seed] = results["1"]["precision"]
        svm_npv[num_seed] = results["0"]["precision"]

        progress_bar.update(1)
        num_seed += 1

    # Print results
    progress_bar.close()
    print(f"RESULTS for outcome: {args.outcome}")
    print()
    print("Logistic Regression")
    print(f"Acc: {lr_acc.mean():>10.2f} ({lr_acc.std():.2f})")
    print(f"Sensitivity: {lr_sensitivity.mean():>10.2f} ({lr_sensitivity.std():.2f})")
    print(f"Specificity: {lr_specificity.mean():>10.2f} ({lr_specificity.std():.2f})")
    print(f"PPV: {lr_ppv.mean():>10.2f} ({lr_ppv.std():.2f})")
    print(f"NPV: {lr_npv.mean():>10.2f} ({lr_npv.std():.2f})")
    lr_youden = lr_sensitivity + lr_specificity - 1
    print(f"Youden J: {lr_youden.mean():>10.2f} ({lr_youden.std():.2f})")
    print("-" * 30)
    print()

    print("Random forest")
    print(f"Acc: {rf_acc.mean():>10.2f} ({rf_acc.std():.2f})")
    print(f"Sensitivity: {rf_sensitivity.mean():>10.2f} ({rf_sensitivity.std():.2f})")
    print(f"Specificity: {rf_specificity.mean():>10.2f} ({rf_specificity.std():.2f})")
    print(f"PPV: {rf_ppv.mean():>10.2f} ({rf_ppv.std():.2f})")
    print(f"NPV: {rf_npv.mean():>10.2f} ({rf_npv.std():.2f})")
    rf_youden = rf_sensitivity + rf_specificity - 1
    print(f"Youden J: {rf_youden.mean():>10.2f} ({rf_youden.std():.2f})")
    print("-" * 30)
    print()

    print("SVM")
    print(f"Acc: {svm_acc.mean():>10.2f} ({svm_acc.std():.2f})")
    print(f"Sensitivity: {svm_sensitivity.mean():>10.2f} ({svm_sensitivity.std():.2f})")
    print(f"Specificity: {svm_specificity.mean():>10.2f} ({svm_specificity.std():.2f})")
    print(f"PPV: {svm_ppv.mean():>10.2f} ({svm_ppv.std():.2f})")
    print(f"NPV: {svm_npv.mean():>10.2f} ({svm_npv.std():.2f})")
    svm_youden = svm_sensitivity + svm_specificity - 1
    print(f"Youden J: {svm_youden.mean():>10.2f} ({svm_youden.std():.2f})")
    print("-" * 30)
    print()
