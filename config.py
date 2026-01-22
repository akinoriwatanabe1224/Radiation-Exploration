# config.py
# 設定パラメータ・定数
import numpy as np
import os

# ---------------------------
#  乱数シード
# ---------------------------
np.random.seed(0)

# ---------------------------
#  線源・測定点の設定（グリッド座標系: 0〜49）
# ---------------------------
# 旧ワールド座標 [-3, 5], [3, 5] → グリッド座標に変換
sources = np.array([[9.5, 17.5], [13.5, 34.5], [27.5, 19.5]])   # 真の線源位置（グリッド座標）
# 各線源の放射能強度（sources と同じ順序で指定）
source_intensities = np.array([1e6, 1e6, 1e6])  # 各線源の強度
# 旧ワールド座標 [[-9, -4], [-5, -4], [5, -4], [9, -4]] → グリッド座標に変換
measurements = np.array([[33.5, 31.5], [44.5, 34.5], [33.5, 23.5], [44.5, 18.5]])  # 測定点（グリッド座標）

# ---------------------------
#  座標範囲（GIFと統一: 0〜49、グリッドインデックス範囲）
# ---------------------------
x_min, x_max = 0, 49
y_min, y_max = 0, 49
grid_resolution = 1.0  # ヒートマップ解像度（グリッド単位）

# ---------------------------
#  角度・レイのパラメータ
# ---------------------------
angle_noise_std_deg = 2
num_rays_per_pair = 100
num_env_rays_per_msr = 30
num_bins = 90
kappa = 800  # von Mises 集中度

# ---------------------------
#  ヒートマップピーク検出
# ---------------------------
peak_local_size = 3
peak_threshold_ratio = 0.55

# ---------------------------
#  コスト配分・ロボット設定
# ---------------------------
NUM_ROBOTS = 2
rho = 2.0
alpha = 1.0
beta = 1.0
gamma = 1.0

# ---------------------------
#  ロボット初期位置（ワールド座標）
# ---------------------------
# 各ロボットの初期位置を指定（NUM_ROBOTSと同じ数だけ指定）
# 例: [(x1, y1), (x2, y2), ...]
ROBOT_START_POSITIONS = [
    (37.5, 34.5),   # Robot 0 の初期位置
    (37.5, 35.5),  # Robot 1 の初期位置
]

# ---------------------------
#  D* および GPR 用グリッド
# ---------------------------
GRID_SIZE = 50   # D* グリッドサイズ

# ---------------------------
#  LiDAR センサパラメータ
# ---------------------------
SENSOR_RANGE = 5  # LiDARセンサの範囲（グリッドセル数）

# ---------------------------
#  GPR / ローカル探査パラメータ
# ---------------------------
SEARCH_RADIUS = 1           # cells (1 -> 3x3)
GPR_FINE = 150
PREDICT_MARGIN_CELLS = 3.0  # 測定範囲の拡張マージン
PREDICT_SCALE = 3.0         # 測定範囲の拡大倍率
MAX_GPR_ITERS = 8           # 各観測点での最大GPR反復回数

# GPRステップ上限設定
USE_GPR_STEP_LIMIT = True  # True: GPR探査中のステップ数に上限を設ける
MAX_GPR_STEPS = 30          # GPR探査中の最大ステップ数（USE_GPR_STEP_LIMIT=True時に有効）

# ---------------------------
#  TestMAP.csv パス設定
# ---------------------------
TESTMAP_CSV_PATH = "maps/Unit3_3F.csv"

# ---------------------------
#  GIF表示範囲設定
# ---------------------------
# None の場合はグリッド全体を表示（0 〜 GRID_SIZE-1）
# 座標を指定すると、その範囲のみを表示
GIF_DISPLAY_X_MIN = None  # 例: 0, 10, etc. (None = 自動)
GIF_DISPLAY_X_MAX = None  # 例: 49, 40, etc. (None = 自動)
GIF_DISPLAY_Y_MIN = None  # 例: 0, 10, etc. (None = 自動)
GIF_DISPLAY_Y_MAX = None  # 例: 49, 40, etc. (None = 自動)

# ---------------------------
#  実験バージョン名
# ---------------------------
EXPERIMENT_VERSION = "Unit3_3F_1.0_1.0_1.0"   # 実験バージョン名

# ---------------------------
#  出力ディレクトリ
# ---------------------------
PNG_DIR = "results/PNG"
GIF_DIR = "results/GIF"
TXT_DIR = "results/TXT"
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)
