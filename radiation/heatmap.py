# radiation/heatmap.py
# von Mises ヒートマップ・ピーク検出
import numpy as np
from scipy.stats import vonmises
from scipy.ndimage import maximum_filter


def create_vonmises_heatmap(measurements, peak_angle_dict, x_vals, y_vals, kappa=200):
    """
    von Mises 分布を用いたヒートマップを作成する

    Parameters
    ----------
    measurements : ndarray
        測定点位置の配列 shape=(n_measurements, 2)
    peak_angle_dict : dict
        測定点インデックスをキーとするピーク角度リストの辞書
    x_vals : ndarray
        x座標の配列
    y_vals : ndarray
        y座標の配列
    kappa : float
        von Mises 分布の集中度パラメータ

    Returns
    -------
    ndarray
        正規化されたヒートマップ
    ndarray
        Xのメッシュグリッド
    ndarray
        Yのメッシュグリッド
    """
    Xc, Yc = np.meshgrid(x_vals, y_vals)
    heatmap_sum = np.zeros_like(Xc, dtype=float)

    for i, msr in enumerate(measurements):
        dir_angle = np.arctan2(Yc - msr[1], Xc - msr[0])
        for peak_deg in peak_angle_dict[i]:
            peak_rad = np.radians(peak_deg)
            diff = (dir_angle - peak_rad + np.pi) % (2 * np.pi) - np.pi
            heatmap_sum += vonmises.pdf(diff, kappa)

    # 正規化
    heatmap_sum /= np.nanmax(heatmap_sum)
    return heatmap_sum, Xc, Yc


def detect_peaks(heatmap, x_vals, y_vals, peak_local_size=3, peak_threshold_ratio=0.4):
    """
    ヒートマップからローカル最大値（ピーク）を検出する

    Parameters
    ----------
    heatmap : ndarray
        ヒートマップ
    x_vals : ndarray
        x座標の配列
    y_vals : ndarray
        y座標の配列
    peak_local_size : int
        ローカル最大値検出のウィンドウサイズ
    peak_threshold_ratio : float
        ピーク検出の閾値比率

    Returns
    -------
    ndarray
        ピーク座標の配列 shape=(n_peaks, 2)
    ndarray
        ピーク値の配列

    Raises
    ------
    RuntimeError
        ピークが検出されなかった場合
    """
    local_max = (heatmap == maximum_filter(heatmap, size=peak_local_size))
    thr = peak_threshold_ratio * np.nanmax(heatmap)
    peak_mask = local_max & (heatmap > thr)
    coords = np.argwhere(peak_mask)

    if coords.shape[0] == 0:
        raise RuntimeError("No peaks detected — lower peak_threshold_ratio or adjust parameters.")

    # ピークの連続座標
    peaks_xy = np.array([[x_vals[j], y_vals[i]] for (i, j) in coords])
    peaks_val = heatmap[coords[:, 0], coords[:, 1]]

    return peaks_xy, peaks_val
