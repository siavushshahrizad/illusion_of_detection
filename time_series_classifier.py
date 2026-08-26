"""
File runs an LSTM on mental health outcome data.
Results are saved to disk.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lstm import LSTMModel
from loaders_torch import create_dataloader
from processing import (
    compute_mean_and_std,
    normalise_df,
    compute_class_weights
)
from constants import (
    DEPRESSION, 
    ANXIETY_MODERATE,
    LONELINESS_MODERATE,
    RAINBOW_TASK,
    STORY_TASK,
    ID, 
    MFCC,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    TS_SEEDS
)
from loaders import (
    load_phq,
    load_anxiety,
    load_loneliness,
    load_mfcc
)

TRAIN_SIZE = 0.7


if __name__ == "__main__":
    # Load data
    parser = argparse.ArgumentParser(
        description="Runs an LSTM on mental health outcomes."
    )
    parser.add_argument(
        "-o",
        "--outcome",
        type=str,
        default="depression",
        choices=["depression", "loneliness", "anxiety"],
        help="Choose an outcome with flag '-o' such as 'depression'"
    )

    parser.add_argument(
        "-i",
        "--inputs",
        type=str,
        default="rainbow",
        choices=["rainbow", "story"],
        help="Choose input data with flag '-i' such as 'rainbow' for rainbow task"
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

    if args.inputs == "rainbow":
        tasks = RAINBOW_TASK
        task_name = "rainbow"
    else:
        tasks = STORY_TASK
        task_name = "story"

    mfcc = load_mfcc(tasks)

    df_orig = pd.merge(
        outcome_data,
        mfcc,
        how="inner",
        on=ID
    )

    print(f"Running multiple {TS_SEEDS} seed(s) to train models")
    print(f"Task: {outcome_var}")
    print(f"Inputs: {tasks}")

    progress_bar = tqdm(total=TS_SEEDS)
    num_seed = 0        # To track results

    # Store results
    train_loss = np.zeros((TS_SEEDS, EPOCHS))
    train_acc = np.zeros((TS_SEEDS, EPOCHS))
    train_sensitivity = np.zeros((TS_SEEDS, EPOCHS))
    train_specificity = np.zeros((TS_SEEDS, EPOCHS))
    train_ppv = np.zeros((TS_SEEDS, EPOCHS))
    train_npv = np.zeros((TS_SEEDS, EPOCHS))

    val_loss = np.zeros((TS_SEEDS, EPOCHS))
    val_acc = np.zeros((TS_SEEDS, EPOCHS))
    val_sensitivity = np.zeros((TS_SEEDS, EPOCHS))
    val_specificity = np.zeros((TS_SEEDS, EPOCHS))
    val_ppv = np.zeros((TS_SEEDS, EPOCHS))
    val_npv = np.zeros((TS_SEEDS, EPOCHS))

    for seed in range(TS_SEEDS):
        # Random behaviour
        torch.manual_seed(seed)
        df = df_orig.copy()

        # Train test split
        train_idx, val_idx = train_test_split(      # Called val, but is the test split
            df.index,
            train_size=TRAIN_SIZE,
            random_state=seed,
            stratify=df[outcome_var]
        )

        # Normalise
        mean, std = compute_mean_and_std(df.loc[train_idx]) # Only use train or leakage!!!
        normalise_df(df, mean, std)

        # Dataloader expects np.ndarray rather than pd.Series
        X_train = df[MFCC].loc[train_idx].values
        y_train = df[outcome_var].loc[train_idx].values

        X_val = df[MFCC].loc[val_idx].values
        y_val = df[outcome_var].loc[val_idx].values

        # Data loaders
        train_dataloader = create_dataloader(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            device="GPU",
        )

        val_dataloader = create_dataloader(
            X_val,
            y_val,
            batch_size=BATCH_SIZE,
            shuffle=True,
            device="GPU",
        )

        weights = compute_class_weights(df.loc[train_idx], outcome_var)
        lstm = LSTMModel(
            weights=weights, 
            learning_rate=LEARNING_RATE,
            device="GPU"
        )

        for epoch in range(EPOCHS):
            train_result = lstm.model_train(train_dataloader)
            val_result = lstm.model_predict(val_dataloader)

            train_report = classification_report(
                train_result["labels"], 
                train_result["preds"],
                output_dict=True
            )

            val_report = classification_report(
                val_result["labels"], 
                val_result["preds"],
                output_dict=True
            )

            train_loss[seed][epoch] = train_result["loss"]
            train_acc[seed][epoch] = train_report["accuracy"]
            train_sensitivity[seed][epoch] = train_report["1"]["recall"]
            train_specificity[seed][epoch] = train_report["0"]["recall"]
            train_ppv[seed][epoch] = train_report["1"]["precision"]
            train_npv[seed][epoch] = train_report["0"]["precision"]
            
            val_loss[seed][epoch] = val_result["loss"]
            val_acc[seed][epoch] = val_report["accuracy"]
            val_sensitivity[seed][epoch] = val_report["1"]["recall"]
            val_specificity[seed][epoch] = val_report["0"]["recall"]
            val_ppv[seed][epoch] = val_report["1"]["precision"]
            val_npv[seed][epoch] = val_report["0"]["precision"]

        num_seed += 1
        progress_bar.update(1)

    progress_bar.close()
    train_youden = train_sensitivity + train_specificity - 1
    val_youden = val_sensitivity + val_specificity - 1

    # Results saving
    print(f"Saving results from {TS_SEEDS} seed(s)")
    time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    final_file = outcome_var + "_" + task_name + "_" + time + ".npy.npz" 
    np.savez(
        final_file,
        train_loss=train_loss,
        train_acc=train_acc,
        train_sensitivity=train_sensitivity,
        train_specificity=train_specificity,
        train_ppv=train_ppv,
        train_npv=train_npv,
        train_youden=train_youden,
        val_loss=val_loss,
        val_acc=val_acc,
        val_sensitivity=val_sensitivity,
        val_specificity=val_specificity,
        val_ppv=val_ppv,
        val_npv=val_npv,
        val_youden=val_youden
    )

    print("Done")
