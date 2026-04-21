# =============================================================
# config.py — Centralized configuration for BSL DL project
# =============================================================
import os

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR        = r"G:\HK2N4\HK2N4\TTNM"
NPY_DIR_ORIG    = os.path.join(BASE_DIR, "features_npy")       # original raw coordinates
NPY_DIR_ROOTREL = os.path.join(BASE_DIR, "features_npy_rootrel")  # root-relative normalized (auto only)
NPY_DIR         = os.path.join(BASE_DIR, "features_npy_rootrel")  # auto data + sqrt-freq sampling
TEST_SAMPLES    = os.path.join(BASE_DIR, "test_samples")
MANIFEST_JSON   = os.path.join(TEST_SAMPLES, "manifest.json")

TTNM_DIR        = BASE_DIR
DB_INDEX        = os.path.join(BASE_DIR, "bobsl_v1_4_videos_mp4.tar.index.sqlite")
SUBTITLE_TAR    = os.path.join(BASE_DIR, "bobsl_v1_4_auto_signing_aligned_subtitles_auto_sat_aligned.tar.gz")
TEMP_DIR        = os.path.join(BASE_DIR, "temp_videos")

DL_DIR          = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR  = os.path.join(DL_DIR, "checkpoints")
RUNS_DIR        = os.path.join(DL_DIR, "runs")
DEMO_OUTPUT_DIR = os.path.join(DL_DIR, "demo_output")

# ─── BOBSL Download ───────────────────────────────────────────
TAR_URL     = "https://thor.robots.ox.ac.uk/bobsl/v1.4/original_data/bobsl_v1_4_videos_mp4.tar"
AUTH_HEADER = "Authorization:Có cc t này m Auth"

# ─── Data ─────────────────────────────────────────────────────
# 36 classes — manual-annotation verified subset (≥20 expert-verified samples each)
# Dropped 20: address, aware, career, choose, college, computer, determine,
#   hearing, hello, language, meaning, project, school, student, study,
#   system, teacher, translate, university, video
CLASSES = sorted([
    'afternoon', 'always',    'and',       'animal',    'anything',
    'area',      'ask',       'because',   'best',      'better',
    'big',       'bird',      'building',  'but',       'call',
    'camera',    'chicken',   'die',       'different',  'education',
    'good',      'learn',     'make',      'me',        'morning',
    'my',        'name',      'one',       'our',       'show',
    'sign',      'technology','three',     'two',       'understand',
    'work',
])
NUM_CLASSES  = len(CLASSES)         # 56
CLASS2IDX    = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS    = {i: c for c, i in CLASS2IDX.items()}

# Anti-bias: cap max samples per class to reduce imbalance
MAX_SAMPLES_PER_CLASS = None   # disabled — using sqrt-frequency sampling instead     # None = no cap; 500 reduces 163x→19x imbalance

# Input tensor: (frames, landmarks, coords)
N_FRAMES     = 50
N_LANDMARKS  = 75
N_COORDS     = 2
INPUT_SIZE   = N_LANDMARKS * N_COORDS  # 150 (flattened per frame)

# Root-relative normalization constants
ROOTREL_CLIP_MIN = -3.0   # clip root-relative coords lower bound
ROOTREL_CLIP_MAX =  8.0   # clip root-relative coords upper bound
ROOTREL_IDX_LEFT_SHOULDER  = 11
ROOTREL_IDX_RIGHT_SHOULDER = 12
ROOTREL_FALLBACK_SHOULDER_W = 0.3068  # dataset median shoulder width
ROOTREL_MIN_SHOULDER_W      = 0.02    # prevent division by tiny values

# ─── Split ratios ─────────────────────────────────────────────
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
SEED         = 42

# ─── Training hyperparameters ─────────────────────────────────
BATCH_SIZE   = 64
NUM_EPOCHS   = 100
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 15          # early stopping
WARMUP_EPOCHS= 5
GRAD_CLIP    = 1.0

# ─── Model (Option 5 default) ─────────────────────────────────
CNN_CHANNELS = 128
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
ATTN_HEADS   = 4
DROP_CNN     = 0.1
DROP_LSTM    = 0.3
DROP_CLS     = 0.4
FC_HIDDEN    = 128

# ─── Demo videos ──────────────────────────────────────────────
DEMO_N_PER_CLASS  = 5
DEMO_WINDOW_SEC   = 1.5    # ±1.5s around annotation center → 3s clip
DEMO_VIDEO_FPS    = 25     # BOBSL native FPS

# ─── Real-time camera ─────────────────────────────────────────
CAM_BUFFER_SIZE   = N_FRAMES   # keep latest 50 frames
CAM_CONFIDENCE    = 0.5
PREDICT_EVERY_N   = 5          # predict every 5 new frames
TRAIN_FPS         = 25         # BOBSL training video FPS (50 frames = 2.0s)

# ─── Sign Spotting (Phase 2 – B1 heuristic) ──────────────────
SPOT_VELOCITY_THRESHOLD = 0.012   # normalized hand displacement per frame
SPOT_MIN_SIGN_FRAMES    = 15      # minimum frames to be a valid sign
SPOT_MAX_SIGN_FRAMES    = 80      # maximum frames before force-cut
SPOT_COOLDOWN_FRAMES    = 10      # idle frames after sign before next detect
SPOT_PRE_BUFFER         = 5       # keep N frames before motion onset
SPOT_POST_BUFFER        = 5       # keep N frames after motion stops
SPOT_SMOOTH_WINDOW      = 5       # moving-average window for velocity
SPOT_IDLE_THRESHOLD     = 0.005   # below this = definitely idle

# ─── Ensemble (Phase 2 – F) ──────────────────────────────────
ENSEMBLE_MODELS  = ["hybrid", "tcn", "bilstm", "transformer"]
ENSEMBLE_WEIGHTS = [0.15, 0.40, 0.10, 0.35]  # optimized on val: +0.23pp over equal (86.27% test)
ENSEMBLE_MODE    = "soft"         # "soft" = weighted prob avg, "hard" = majority vote
ENSEMBLE_USE_KD  = False          # disabled — old _kd.pt trained on non-rootrel data

# fix_worst_classes.py results (2026-03-02, rootrel data):
#   Best individual: method3/hybrid (73.93% overall, 46.4% worst avg)
#   All methods trade overall accuracy for modest worst-class improvement
#   In ensemble context: equal weights outperform individual method checkpoints
# Old optimized weights [0.20, 0.50, 0.10, 0.20] → 80.28% overall, 44.0% worst avg
# Equal weights [0.25, 0.25, 0.25, 0.25] → 80.10% overall, 45.5% worst avg (+1.5pp worst)
ENSEMBLE_USE_HARDMINE = False     # _hardmine.pt doesn't improve ensemble accuracy

# ─── Manual Dataset Flag ─────────────────────────────────────
# Set True to use the 21-class manual-annotations-only dataset
# instead of the 36-class auto dataset. When True, config_selector
# automatically routes to config_manual.py settings.
USE_MANUAL_DATASET = True
