from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATASETS_DIR = PROJECT_ROOT / 'datasets'
RAW_ALPHABET_TRAIN_DIR = RAW_DATASETS_DIR / 'ASL_Alphabet_Dataset' / 'asl_alphabet_train'
RAW_ALPHABET_TEST_DIR = RAW_DATASETS_DIR / 'ASL_Alphabet_Dataset' / 'asl_alphabet_test'
RAW_DIGITS_DIR = RAW_DATASETS_DIR / 'asl_digits_images'

RUN_ROOT = PROJECT_ROOT / 'runs' / 'ASL_36'
DATASET_RUN_DIR = RUN_ROOT / 'dataset'
TRAIN_DIR = RUN_ROOT / 'training'
EVAL_DIR = RUN_ROOT / 'evaluation'
REALTIME_DIR = RUN_ROOT / 'realtime'
COMPARE_DIR = RUN_ROOT / 'comparison'
WEB_DIR = RUN_ROOT / 'web_demo'
LOG_DIR = RUN_ROOT / 'logs'

TRAIN_CSV = DATASET_RUN_DIR / 'train_39.csv'
VAL_CSV = DATASET_RUN_DIR / 'val_39.csv'
CLASS_NAMES_JSON = DATASET_RUN_DIR / 'class_names_39.json'

BEST_MODEL_PATH = TRAIN_DIR / 'best_model.pth'
LAST_MODEL_PATH = TRAIN_DIR / 'last_model.pth'

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NUM_CLASSES = len(CLASS_NAMES)

def ensure_run_dirs() -> None:
    for folder in [RUN_ROOT, DATASET_RUN_DIR, TRAIN_DIR, EVAL_DIR, REALTIME_DIR, COMPARE_DIR, WEB_DIR, LOG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
