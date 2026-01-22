# test/config.py
# テスト用設定パラメータ
import numpy as np
import os

# ---------------------------
#  乱数シード
# ---------------------------
np.random.seed(0)

# ---------------------------
#  線源・測定点の設定（グリッド座標系: 0〜49）
# ---------------------------
sources = np.array([[9.5, 17.5], [13.5, 34.5], [27.5, 19.5]])   # 真の線源位置（グリッド座標）
# 各線源の放射能強度（sources と同じ順序で指定）
source_intensities = np.array([1e6, 1e6, 1e6])  # 各線源の強度
measurements = np.array([[33.5, 31.5], [44.5, 34.5], [33.5, 23.5], [44.5, 18.5]])  # 測定点（グリッド座標）

# ---------------------------
#  座標範囲（グリッドインデックス範囲）
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
peak_local_size = 5
peak_threshold_ratio = 0.55

# ---------------------------
#  グリッドサイズ
# ---------------------------
GRID_SIZE = 50

# ---------------------------
#  放射線シミュレーション
# ---------------------------
RADIATION_INTENSITY = 1e6  # 線源の放射能

# ---------------------------
#  障害物CSV設定
# ---------------------------
# CSVファイルのパス（Noneの場合はDEFAULT_OBSTACLESを使用）
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(TEST_DIR)

# 使用するマップCSVのパス（任意のパスを指定可能）
# 例: TESTMAP_CSV_PATH = os.path.join(PARENT_DIR, "maps", "TestMAP3.csv")
# 例: TESTMAP_CSV_PATH = "G:/マイドライブ/大学/研究/00_program_main/maps/custom_map.csv"
TESTMAP_CSV_PATH = os.path.join(PARENT_DIR, "maps", "Unit1_1F.csv")

# CSVファイルが見つからない場合のデフォルト障害物
DEFAULT_OBSTACLES = [
    # (x0, x1, y0, y1, mu)
    (20, 25, 25, 35, 1.0),  # 中央の遮蔽物
    (35, 40, 20, 30, 2.0),  # 右側の遮蔽物（減衰係数大）
]

# ---------------------------
#  出力ディレクトリ
# ---------------------------
PNG_DIR = os.path.join(TEST_DIR, "PNG")
os.makedirs(PNG_DIR, exist_ok=True)
