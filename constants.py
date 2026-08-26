# Randomness
SEED = 42
SEEDS = 50
TS_SEEDS = 20       # Just for LSTM 

# Files
# You will need to adjust the below strs, or create
# a similar folder structure if you want to use the code.
PHQ_FILE = "./data/phq9.tsv"
MFCC_FILE = "./data/torchaudio_mfcc.parquet"
STATIC_FILE = "./data/static_features.tsv"
LONELY_FILE = "./data/custom_affect_scale.tsv"
ANXIETY_FILE = "./data/gad7_anxiety.tsv" 

# Vars
DEPRESSION = "depression"
LONELINESS_MODERATE = "loneliness_moderate"
LONELINESS_SEVERE = "loneliness_severe"
ANXIETY_MODERATE = "anxiety_moderate"
ANXIETY_SEVERE = "anxiety_severe"

ID = "participant_id"  
MFCC = "mfcc"
TASK = "task_name"
LEFT = "left"
RIGHT = "right"
TRAIN_SIZE = 0.8

STATIC_VARS = [
    "mfcc1_sma3_stddevNorm",
    "mfcc2_sma3_stddevNorm",
    "mfcc3_sma3_stddevNorm",
    "mfcc4_sma3_stddevNorm",
    "F1frequency_sma3nz_amean",
    "F2frequency_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "cepstral_peak_prominence_mean",
    "local_jitter",
    "mean_hnr_db",
    "speaking_rate",
]

# Audio data src
FREE_TASKS = [
    'free-speech-1',
    'free-speech-2',
    'free-speech-3',
    'free-speech',
    'free-speech-v2-1',
    'free-speech-v2-2',
    'free-speech-v2-3',
]  

RAINBOW_TASK = ["rainbow-passage"]
STORY_TASK = ["story-recall", "story-recall-v2"]

# LSTM hyperparams
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
INPUT_SIZE = 60         # Num of MFCCs, i.e. 60
HIDDEN_SIZE = 32 
NUM_LAYERS_RNN = 1
EPOCHS = 10
OUTPUT_DIM = 2
