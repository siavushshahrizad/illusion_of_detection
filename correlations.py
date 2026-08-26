"""File evaluates correlations of static vars with mental outcomes."""

import pandas as pd
from constants import (
    ID, 
    TASK, 
    FREE_TASKS,
    STATIC_VARS
)
from loaders import (
    load_static, 
    load_phq, 
    load_loneliness,
    load_anxiety
)


if __name__ == "__main__":
    ### Load/process data ###
    #                       #
    #########################

    static = load_static(FREE_TASKS, STATIC_VARS)

    # Depression
    phq = load_phq()

    df_depression = pd.merge(
        phq,
        static,
        how="inner",
        on=ID
    )

    df_depression = df_depression.drop(columns=[ID, TASK]) # Else str error in corr

    # Loneliness
    lonely = load_loneliness()

    df_lonely = pd.merge(
        lonely,
        static,
        how="inner",
        on=ID
    )

    df_lonely = df_lonely.drop(columns=[ID, TASK]) # Ditto 

    # Anxiety
    anxiety = load_anxiety()

    df_anxiety = pd.merge(
        anxiety,
        static,
        how="inner",
        on=ID
    )

    df_anxiety = df_anxiety.drop(columns=[ID, TASK]) # Ditto 

    ###   Correlations    ###
    #                       #
    #########################
    depression_corr = df_depression.corr(method="spearman")
    lonely_corr = df_lonely.corr(method="spearman")
    anxiety_corr = df_anxiety.corr(method="spearman")

    print("Depression correlations")
    print(depression_corr)
    print()
    
    print("Loneliness correlations")
    print(lonely_corr)
    print()

    print("Anxiety correlations")
    print(anxiety_corr)
    print()
