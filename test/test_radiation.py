# test/test_radiation.py
# 放射線関連機能のテストスクリプト
# - 放射線方向データ・レイ可視化
# - von Misesヒートマップ
# - 遮蔽を考慮した放射線強度マップ

import os
import sys
import numpy as np

# 親ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.config import (
    sources, measurements, source_intensities,
    x_min, x_max, y_min, y_max, grid_resolution,
    angle_noise_std_deg, num_rays_per_pair, num_env_rays_per_msr, num_bins, kappa,
    peak_local_size, peak_threshold_ratio,
    GRID_SIZE, RADIATION_INTENSITY, DEFAULT_OBSTACLES, TESTMAP_CSV_PATH,
    PNG_DIR
)
from radiation.direction import generate_direction_data, extract_peak_angles
from radiation.heatmap import create_vonmises_heatmap, detect_peaks
from radiation.simulation import generate_radiation_field_with_shielding, add_poisson_noise
from obstacles.rectangle import RectangleObstacle
from visualization.gif_generator import (
    save_direction_viz, save_heatmap_viz, save_heatmap_only_viz,
    save_heatmap_with_obstacles_viz, save_intensity_map_viz
)


def load_obstacles_from_csv_or_default():
    """
    CSVファイルから障害物を読み込む。
    CSVが見つからない場合はデフォルト障害物を使用。
    """
    obstacles = []

    if TESTMAP_CSV_PATH and os.path.exists(TESTMAP_CSV_PATH):
        print(f"Loading obstacles from CSV: {TESTMAP_CSV_PATH}")
        obstacles = RectangleObstacle.load_obstacles(TESTMAP_CSV_PATH)
    else:
        print(f"CSV not found: {TESTMAP_CSV_PATH}")
        print("Using default obstacles instead.")

    # CSVが空またはファイルがない場合、デフォルト障害物を使用
    if len(obstacles) == 0:
        print("Creating default obstacles...")
        for obs_data in DEFAULT_OBSTACLES:
            x0, x1, y0, y1, mu = obs_data
            obstacles.append(RectangleObstacle(x0, x1, y0, y1, mu=mu))

    return obstacles


def test_direction_data():
    """
    テスト1: 放射線方向データ生成・レイ可視化
    """
    print("=" * 60)
    print("TEST 1: Direction Data Generation")
    print("=" * 60)

    # 方向データ生成
    angle_dict, ray_data = generate_direction_data(
        sources, measurements,
        x_min, x_max, y_min, y_max,
        num_rays_per_pair=num_rays_per_pair,
        num_env_rays_per_msr=num_env_rays_per_msr,
        angle_noise_std_deg=angle_noise_std_deg
    )

    print(f"Generated direction data for {len(measurements)} measurement points")
    print(f"Total rays: {len(ray_data)}")

    # レイ可視化を保存
    output_path = os.path.join(PNG_DIR, "test_direction_viz.png")
    save_direction_viz(
        sources, measurements, ray_data,
        x_min, x_max, y_min, y_max,
        output_path
    )
    print(f"Saved: {output_path}")

    # ピーク角度を抽出
    peak_angle_dict = extract_peak_angles(angle_dict, num_bins=num_bins)
    print(f"Extracted peak angles:")
    for i, peaks in peak_angle_dict.items():
        print(f"  Measurement {i}: {[f'{p:.1f}' for p in peaks]} deg")

    return angle_dict, peak_angle_dict


def test_vonmises_heatmap(peak_angle_dict):
    """
    テスト2: von Misesヒートマップ生成・ピーク検出
    """
    print("\n" + "=" * 60)
    print("TEST 2: von Mises Heatmap Generation")
    print("=" * 60)

    # ヒートマップ用グリッド
    x_vals = np.arange(x_min, x_max + 1e-9, grid_resolution)
    y_vals = np.arange(y_min, y_max + 1e-9, grid_resolution)

    # von Misesヒートマップ作成
    heatmap_sum, Xc, Yc = create_vonmises_heatmap(
        measurements, peak_angle_dict, x_vals, y_vals, kappa=kappa
    )
    print(f"Created von Mises heatmap: shape={heatmap_sum.shape}")

    # ピーク検出
    try:
        peaks_xy, peaks_val = detect_peaks(
            heatmap_sum, x_vals, y_vals,
            peak_local_size=peak_local_size,
            peak_threshold_ratio=peak_threshold_ratio
        )
        print(f"Detected {len(peaks_xy)} peaks:")
        for i, (xy, val) in enumerate(zip(peaks_xy, peaks_val)):
            print(f"  Peak {i+1}: ({xy[0]:.1f}, {xy[1]:.1f}), value={val:.4f}")
    except RuntimeError as e:
        print(f"Warning: {e}")
        peaks_xy = np.array([[25, 37]])  # ダミーピーク
        peaks_val = np.array([1.0])

    # ヒートマップのみの図を保存（ピークや線源なし）
    output_path_only = os.path.join(PNG_DIR, "test_vonmises_heatmap_only.png")
    save_heatmap_only_viz(
        heatmap_sum, Xc, Yc,
        x_min, x_max, y_min, y_max,
        output_path_only
    )
    print(f"Saved: {output_path_only}")

    # ヒートマップ + ピーク + 線源の可視化を保存
    output_path = os.path.join(PNG_DIR, "test_vonmises_heatmap.png")
    save_heatmap_viz(
        heatmap_sum, Xc, Yc, peaks_xy, sources, measurements,
        x_min, x_max, y_min, y_max,
        output_path
    )
    print(f"Saved: {output_path}")

    return heatmap_sum, Xc, Yc, peaks_xy


def test_radiation_intensity_with_shielding(heatmap_data=None):
    """
    テスト3: 遮蔽を考慮した放射線強度マップ生成

    Parameters
    ----------
    heatmap_data : tuple or None
        (heatmap_sum, Xc, Yc, peaks_xy) - von Misesヒートマップ+障害物の可視化用
    """
    print("\n" + "=" * 60)
    print("TEST 3: Radiation Intensity Map with Shielding")
    print("=" * 60)

    # 障害物をCSVまたはデフォルトから読み込む
    obstacles = load_obstacles_from_csv_or_default()
    print(f"Loaded {len(obstacles)} obstacles:")
    for i, obs in enumerate(obstacles):
        print(f"  Obstacle {i+1}: x=[{obs.x0}, {obs.x1}], y=[{obs.y0}, {obs.y1}], mu={obs.mu}")

    # von Misesヒートマップ + 障害物の重ね図を保存
    if heatmap_data is not None:
        heatmap_sum, Xc, Yc, peaks_xy = heatmap_data
        output_path_hm_obs = os.path.join(PNG_DIR, "test_vonmises_heatmap_obstacles.png")
        save_heatmap_with_obstacles_viz(
            heatmap_sum, Xc, Yc, peaks_xy, sources, measurements,
            obstacles, x_min, x_max, y_min, y_max,
            output_path_hm_obs
        )
        print(f"Saved: {output_path_hm_obs}")

    # 遮蔽を考慮した放射線強度フィールドを生成（各線源の個別強度を使用）
    print("\nGenerating radiation field with shielding...")
    intensity_field, sources_grid, sources_intensity_list = generate_radiation_field_with_shielding(
        sources, x_min, x_max, y_min, y_max, GRID_SIZE,
        obstacles=obstacles, intensity=source_intensities
    )
    print(f"Generated intensity field: shape={intensity_field.shape}")
    print(f"Intensity range: [{intensity_field.min():.2e}, {intensity_field.max():.2e}]")
    print(f"Sources (grid): {sources_grid}")
    print(f"Source intensities: {sources_intensity_list}")

    # ポアソンノイズを追加
    observed_counts = add_poisson_noise(intensity_field)
    print(f"Added Poisson noise: range=[{observed_counts.min()}, {observed_counts.max()}]")

    # 強度マップを保存（ポアソンノイズなし）
    output_path = os.path.join(PNG_DIR, "test_intensity_map_shielding.png")
    save_intensity_map_viz(
        intensity_field, sources_grid, obstacles,
        x_min, x_max, y_min, y_max,
        output_path,
        title="Radiation Intensity Map (with Shielding)",
        use_log_scale=True
    )
    print(f"Saved: {output_path}")

    # 観測カウント（ポアソンノイズあり）も保存
    output_path_noisy = os.path.join(PNG_DIR, "test_intensity_map_observed.png")
    save_intensity_map_viz(
        observed_counts, sources_grid, obstacles,
        x_min, x_max, y_min, y_max,
        output_path_noisy,
        title="Observed Counts (with Poisson Noise)",
        use_log_scale=True
    )
    print(f"Saved: {output_path_noisy}")

    return intensity_field, observed_counts


def main():
    """テストメイン関数"""
    print("\n" + "=" * 60)
    print("RADIATION MODULE TEST")
    print("=" * 60)
    print(f"Output directory: {PNG_DIR}\n")

    # テスト1: 方向データ
    angle_dict, peak_angle_dict = test_direction_data()

    # テスト2: von Misesヒートマップ
    heatmap_sum, Xc, Yc, peaks_xy = test_vonmises_heatmap(peak_angle_dict)

    # テスト3: 遮蔽を考慮した強度マップ（ヒートマップデータも渡す）
    heatmap_data = (heatmap_sum, Xc, Yc, peaks_xy)
    intensity, observed = test_radiation_intensity_with_shielding(heatmap_data)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {PNG_DIR}")
    print("Files generated:")
    print("  - test_direction_viz.png")
    print("  - test_vonmises_heatmap_only.png")
    print("  - test_vonmises_heatmap.png")
    print("  - test_vonmises_heatmap_obstacles.png")
    print("  - test_intensity_map_shielding.png")
    print("  - test_intensity_map_observed.png")


if __name__ == "__main__":
    main()
