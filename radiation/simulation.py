# radiation/simulation.py
# 放射線強度シミュレーション
import numpy as np

from utils.coordinate import world_to_grid


def generate_radiation_field(sources, x_min, x_max, y_min, y_max, grid_size,
                             intensity=1000, sigma=5):
    """
    ガウス分布に基づく放射線強度フィールドを生成する

    Parameters
    ----------
    sources : ndarray
        線源位置の配列 shape=(n_sources, 2)
    x_min, x_max, y_min, y_max : float
        ワールド座標の範囲
    grid_size : int
        グリッドサイズ
    intensity : float
        放射線強度の最大値
    sigma : float
        ガウス分布の標準偏差

    Returns
    -------
    ndarray
        放射線強度フィールド shape=(grid_size, grid_size)
    list
        線源のグリッド座標リスト [(gx, gy), ...]
    """
    true_intensity = np.zeros((grid_size, grid_size), dtype=float)
    sources_grid = []

    for src in sources:
        gx, gy = world_to_grid(src, x_min, x_max, y_min, y_max, grid_size)
        sources_grid.append((gx, gy))

        # ガウス分布で放射線強度を生成
        for y in range(grid_size):
            for x in range(grid_size):
                dist = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
                true_intensity[y, x] += intensity * np.exp(-dist ** 2 / (2 * sigma ** 2))

    return true_intensity, sources_grid


def generate_radiation_field_with_shielding(sources, x_min, x_max, y_min, y_max, grid_size,
                                            obstacles=None, intensity=1e6, softening=None):
    """
    遮蔽を考慮した放射線強度フィールドを生成する（ソフトコア関数モデル）

    ソフトコア関数モデル:
        intensity = I₀ / (4π(r² + ε²))

    特徴:
    - r = 0 でも発散しない（intensity = I₀ となる）
    - r >> ε で逆二乗則 I₀/(4πr²) に漸近
    - 境界での不連続性がなく、滑らかに減衰

    Parameters
    ----------
    sources : ndarray
        線源位置の配列 shape=(n_sources, 2) - グリッド座標
    x_min, x_max, y_min, y_max : float
        座標範囲
    grid_size : int
        グリッドサイズ
    obstacles : list
        RectangleObstacleのリスト
    intensity : float or ndarray
        線源の放射能（activity）。スカラーの場合は全線源に同じ値を適用。
        配列の場合は各線源に個別の強度を適用。
    softening : float, optional
        ソフトニングパラメータ ε
        デフォルト: 1/√(4π) ≈ 0.282（r=0で線源強度と等しくなる値）

    Returns
    -------
    ndarray
        放射線強度フィールド shape=(grid_size, grid_size)
    list
        線源のグリッド座標リスト [(gx, gy), ...]
    list
        各線源の強度リスト
    """
    if obstacles is None:
        obstacles = []

    # デフォルトのソフトニングパラメータ: r=0 で intensity = src_intensity_val となる値
    if softening is None:
        softening = 1.0 / np.sqrt(4 * np.pi)

    # 強度を配列に変換（スカラーの場合は全線源に同じ値）
    intensities = np.atleast_1d(intensity)
    if len(intensities) == 1:
        intensities = np.full(len(sources), intensities[0])

    # グリッド座標の配列を作成
    x_range = np.linspace(x_min, x_max, grid_size)
    y_range = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_range, y_range)

    intensity_total = np.zeros_like(X, dtype=float)
    sources_grid = []
    sources_intensity_list = []

    for idx, src in enumerate(sources):
        src_x, src_y = src[0], src[1]
        src_intensity_val = intensities[idx]
        sources_grid.append((src_x, src_y))  # 浮動小数点のまま保持（セル中央座標）
        sources_intensity_list.append(src_intensity_val)

        # 距離計算
        r = np.sqrt((X - src_x) ** 2 + (Y - src_y) ** 2)

        # 遮蔽による減衰係数を計算
        total_attenuation_coeff = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for obs in obstacles:
                    intersections = obs.get_intersections(
                        (src_x, src_y),
                        (X[i, j], Y[i, j])
                    )
                    if intersections:
                        (x0, y0), (x1, y1) = intersections
                        dist = np.hypot(x1 - x0, y1 - y0)
                        total_attenuation_coeff[i, j] += obs.mu * dist

        # 減衰を適用
        attenuation = np.exp(-total_attenuation_coeff)

        # ソフトコア関数による強度計算
        # intensity = I₀ / (4π(r² + ε²))
        src_intensity = src_intensity_val / (4 * np.pi * (r ** 2 + softening ** 2))

        intensity_total += src_intensity * attenuation

    return intensity_total, sources_grid, sources_intensity_list


def add_poisson_noise(intensity_field):
    """
    放射線強度フィールドにポアソンノイズを追加する

    Parameters
    ----------
    intensity_field : ndarray
        放射線強度フィールド

    Returns
    -------
    ndarray
        ポアソンノイズが追加された観測カウント
    """
    return np.random.poisson(intensity_field)


def calculate_intensity_at_point(measurement_pos, sources, intensities,
                                  obstacles=None, softening=None):
    """
    指定した測定位置での放射線強度を計算する（ソフトコア関数モデル）

    ソフトコア関数モデル:
        intensity = I₀ / (4π(r² + ε²))

    特徴:
    - r = 0 でも発散しない（intensity = I₀ となる）
    - r >> ε で逆二乗則 I₀/(4πr²) に漸近
    - 境界での不連続性がなく、滑らかに減衰

    Parameters
    ----------
    measurement_pos : tuple
        測定位置 (x, y) - 浮動小数点座標
    sources : list or ndarray
        線源位置のリスト [(x1, y1), (x2, y2), ...] - 浮動小数点座標
    intensities : list or ndarray
        各線源の放射能（activity）
    obstacles : list, optional
        RectangleObstacleのリスト（遮蔽計算用）
    softening : float, optional
        ソフトニングパラメータ ε
        デフォルト: 1/√(4π) ≈ 0.282（r=0で線源強度と等しくなる値）

    Returns
    -------
    float
        測定位置での総放射線強度
    """
    if obstacles is None:
        obstacles = []

    # デフォルトのソフトニングパラメータ: r=0 で intensity = src_intensity_val となる値
    # I₀ / (4π × ε²) = I₀  →  ε = 1/√(4π) ≈ 0.282
    if softening is None:
        softening = 1.0 / np.sqrt(4 * np.pi)

    mx, my = measurement_pos
    total_intensity = 0.0

    for idx, src in enumerate(sources):
        src_x, src_y = src[0], src[1]
        src_intensity_val = float(intensities[idx])

        # 距離計算
        r = np.sqrt((mx - src_x) ** 2 + (my - src_y) ** 2)

        # ソフトコア関数による強度計算
        # intensity = I₀ / (4π(r² + ε²))
        # r=0: I₀/(4πε²) = I₀ (ε = 1/√(4π) のとき)
        # r→∞: I₀/(4πr²) (逆二乗則に漸近)
        intensity = src_intensity_val / (4 * np.pi * (r ** 2 + softening ** 2))

        # 遮蔽による減衰を計算
        attenuation = 1.0
        for obs in obstacles:
            intersections = obs.get_intersections(
                (src_x, src_y),
                (mx, my)
            )
            if intersections:
                (x0, y0), (x1, y1) = intersections
                dist = np.hypot(x1 - x0, y1 - y0)
                attenuation *= np.exp(-obs.mu * dist)

        total_intensity += intensity * attenuation

    return total_intensity
