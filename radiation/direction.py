# radiation/direction.py
# 方向データ生成・角度ヒストグラム
import numpy as np
from scipy.signal import find_peaks

from utils.geometry import find_intersection


def generate_direction_data(sources, measurements, x_min, x_max, y_min, y_max,
                            num_rays_per_pair=100, num_env_rays_per_msr=30,
                            angle_noise_std_deg=5):
    """
    各測定点から線源方向へのレイを生成し、角度データを収集する

    Parameters
    ----------
    sources : ndarray
        線源位置の配列 shape=(n_sources, 2)
    measurements : ndarray
        測定点位置の配列 shape=(n_measurements, 2)
    x_min, x_max, y_min, y_max : float
        ワールド座標の範囲
    num_rays_per_pair : int
        各測定点-線源ペアあたりのレイ数
    num_env_rays_per_msr : int
        各測定点あたりの環境ノイズレイ数
    angle_noise_std_deg : float
        角度ノイズの標準偏差（度）

    Returns
    -------
    dict
        各測定点インデックスをキーとする角度リストの辞書
    list
        可視化用のレイデータ [(msr, end, color, alpha), ...]
    """
    angle_dict = {i: [] for i in range(len(measurements))}
    ray_data = []

    for i, msr in enumerate(measurements):
        for src in sources:
            vec = src - msr
            if np.linalg.norm(vec) == 0:
                continue
            base_angle = np.degrees(np.arctan2(vec[1], vec[0])) % 360
            for _ in range(num_rays_per_pair):
                noisy = (base_angle + np.random.normal(0, angle_noise_std_deg)) % 360
                angle_dict[i].append(noisy)
                rad = np.radians(noisy)
                direction = np.array([np.cos(rad), np.sin(rad)])
                end = find_intersection(msr, direction, x_min, x_max, y_min, y_max)
                ray_data.append((msr, end, 'g', 0.03))

        # 環境ノイズ
        for _ in range(num_env_rays_per_msr):
            rnd = np.random.uniform(0, 360)
            angle_dict[i].append(rnd)
            rad = np.radians(rnd)
            direction = np.array([np.cos(rad), np.sin(rad)])
            end = find_intersection(msr, direction, x_min, x_max, y_min, y_max)
            ray_data.append((msr, end, 'orange', 0.03))

    return angle_dict, ray_data


def extract_peak_angles(angle_dict, num_bins=36):
    """
    各測定点の角度ヒストグラムからピーク角度を抽出する

    Parameters
    ----------
    angle_dict : dict
        測定点インデックスをキーとする角度リストの辞書
    num_bins : int
        ヒストグラムのビン数

    Returns
    -------
    dict
        測定点インデックスをキーとするピーク角度リストの辞書
    """
    peak_angle_dict = {}
    for i in angle_dict.keys():
        angles = angle_dict[i]
        counts, bins = np.histogram(angles, bins=num_bins, range=(0, 360))
        counts_ext = np.concatenate(([counts[-1]], counts, [counts[0]]))
        peaks_ext, props = find_peaks(counts_ext, height=10)
        peaks = []
        for p in peaks_ext:
            idx = p - 1
            if idx < 0:
                idx = num_bins - 1
            if idx >= num_bins:
                idx = 0
            if idx not in peaks:
                peaks.append(idx)
        peak_angles = [(bins[idx] + bins[idx + 1]) / 2.0 for idx in peaks]
        peak_angle_dict[i] = peak_angles
    return peak_angle_dict
