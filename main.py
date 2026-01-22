# main.py
# 統合パイプライン実行スクリプト
import os
import sys
import numpy as np
import imageio

# モジュールパスを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    sources, measurements, source_intensities,
    x_min, x_max, y_min, y_max, grid_resolution,
    angle_noise_std_deg, num_rays_per_pair, num_env_rays_per_msr, num_bins, kappa,
    peak_local_size, peak_threshold_ratio,
    NUM_ROBOTS, rho, alpha, beta, gamma, ROBOT_START_POSITIONS,
    GRID_SIZE, SENSOR_RANGE,
    SEARCH_RADIUS, MAX_GPR_ITERS, PREDICT_SCALE, PREDICT_MARGIN_CELLS,
    USE_GPR_STEP_LIMIT, MAX_GPR_STEPS,
    GIF_DISPLAY_X_MIN, GIF_DISPLAY_X_MAX, GIF_DISPLAY_Y_MIN, GIF_DISPLAY_Y_MAX,
    EXPERIMENT_VERSION as DEFAULT_EXPERIMENT_VERSION,
    TESTMAP_CSV_PATH, PNG_DIR, GIF_DIR, TXT_DIR
)
from utils.coordinate import grid_to_world
from radiation.direction import generate_direction_data, extract_peak_angles
from radiation.heatmap import create_vonmises_heatmap, detect_peaks
from radiation.simulation import generate_radiation_field, add_poisson_noise, generate_radiation_field_with_shielding
from environment.grid import load_obstacles_from_csv, create_occupancy_grid
from obstacles.rectangle import RectangleObstacle
# Note: Dynamic task allocation is now handled inside create_integrated_gif_with_unknown_env
from visualization.gif_generator import (
    save_direction_viz, save_heatmap_viz, save_heatmap_only_viz,
    save_heatmap_with_obstacles_viz, save_intensity_map_viz,
    create_integrated_gif_with_unknown_env
)


def main():
    """メインパイプライン実行"""

    # ---------------------------
    # EXPERIMENT_VERSION のターミナル入力
    # ---------------------------
    print("=" * 60)
    print("Multi-Robot GPR Exploration Pipeline")
    print("=" * 60)
    print(f"\nDefault EXPERIMENT_VERSION: {DEFAULT_EXPERIMENT_VERSION}")
    user_input = input("Enter EXPERIMENT_VERSION (press Enter to use default): ").strip()

    if user_input:
        EXPERIMENT_VERSION = user_input
        print(f"Using custom EXPERIMENT_VERSION: {EXPERIMENT_VERSION}")
    else:
        EXPERIMENT_VERSION = DEFAULT_EXPERIMENT_VERSION
        print(f"Using default EXPERIMENT_VERSION: {EXPERIMENT_VERSION}")
    print()

    # ---------------------------
    # STEP A: 方向データ生成
    # ---------------------------
    print("=" * 60)
    print("STEP A: Generating direction data...")
    print("=" * 60)

    angle_dict, ray_data = generate_direction_data(
        sources, measurements,
        x_min, x_max, y_min, y_max,
        num_rays_per_pair=num_rays_per_pair,
        num_env_rays_per_msr=num_env_rays_per_msr,
        angle_noise_std_deg=angle_noise_std_deg
    )

    # 方向レイの可視化を保存
    save_direction_viz(
        sources, measurements, ray_data,
        x_min, x_max, y_min, y_max,
        os.path.join(PNG_DIR, f"direction_viz_{EXPERIMENT_VERSION}.png")
    )

    # ピーク角度を抽出
    peak_angle_dict = extract_peak_angles(angle_dict, num_bins=num_bins)

    # ---------------------------
    # STEP B: von Mises ヒートマップ作成 & ピーク検出
    # ---------------------------
    print("\n" + "=" * 60)
    print("STEP B: Creating von Mises heatmap and detecting peaks...")
    print("=" * 60)

    x_vals = np.arange(x_min, x_max + 1e-9, grid_resolution)
    y_vals = np.arange(y_min, y_max + 1e-9, grid_resolution)

    heatmap_sum, Xc, Yc = create_vonmises_heatmap(
        measurements, peak_angle_dict, x_vals, y_vals, kappa=kappa
    )

    peaks_xy, peaks_val = detect_peaks(
        heatmap_sum, x_vals, y_vals,
        peak_local_size=peak_local_size,
        peak_threshold_ratio=peak_threshold_ratio
    )

    # ヒートマップの可視化を保存
    save_heatmap_viz(
        heatmap_sum, Xc, Yc, peaks_xy, sources, measurements,
        x_min, x_max, y_min, y_max,
        os.path.join(PNG_DIR, f"vonmises_heatmap_sum_peaks_{EXPERIMENT_VERSION}.png")
    )

    print(f"Detected {len(peaks_xy)} peaks on heatmap.")

    # ---------------------------
    # STEP E: Occupancy grid 作成
    # ---------------------------
    print("\n" + "=" * 60)
    print("STEP E: Creating occupancy grid...")
    print("=" * 60)

    obstacle_rects, csv_loaded = load_obstacles_from_csv(TESTMAP_CSV_PATH)
    true_grid = create_occupancy_grid(obstacle_rects, GRID_SIZE)

    # von Misesヒートマップのみの図を保存
    save_heatmap_only_viz(
        heatmap_sum, Xc, Yc,
        x_min, x_max, y_min, y_max,
        os.path.join(PNG_DIR, f"vonmises_heatmap_only_{EXPERIMENT_VERSION}.png")
    )

    # von Misesヒートマップ + 障害物の重ね図を保存
    save_heatmap_with_obstacles_viz(
        heatmap_sum, Xc, Yc, peaks_xy, sources, measurements,
        obstacle_rects, x_min, x_max, y_min, y_max,
        os.path.join(PNG_DIR, f"vonmises_heatmap_obstacles_{EXPERIMENT_VERSION}.png")
    )

    # ---------------------------
    # STEP G: ロボット開始位置の設定（動的タスク割り当て使用）
    # ---------------------------
    print("\n" + "=" * 60)
    print("STEP G: Setting up robot start positions (dynamic task allocation)...")
    print("=" * 60)

    # ロボットの初期位置（config.pyから取得）
    robot_start_world = ROBOT_START_POSITIONS[:NUM_ROBOTS]
    print(f"Robot start positions (world): {robot_start_world}")

    # ロボット開始位置をグリッド座標に変換
    from utils.coordinate import world_to_grid
    robot_start_grid = []
    for r in range(NUM_ROBOTS):
        start_pos = world_to_grid(robot_start_world[r], x_min, x_max, y_min, y_max, GRID_SIZE)
        sx, sy = start_pos
        # 開始位置が障害物でないことを確認
        if true_grid[sy, sx] == 1:
            print(f"Warning: Robot {r} start position {start_pos} is on obstacle, finding alternative...")
            found = False
            for offset in range(1, 10):
                for dx in range(-offset, offset + 1):
                    for dy in range(-offset, offset + 1):
                        nx, ny = sx + dx, sy + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and true_grid[ny, nx] == 0:
                            start_pos = (nx, ny)
                            found = True
                            print(f"  -> Moved to {start_pos}")
                            break
                    if found:
                        break
                if found:
                    break
        robot_start_grid.append(start_pos)
    print(f"Robot start positions (grid): {robot_start_grid}")

    # ---------------------------
    # STEP H: 放射線シミュレーション（遮蔽考慮）
    # ---------------------------
    print("\n" + "=" * 60)
    print("STEP H: Simulating radiation field with shielding...")
    print("=" * 60)

    # 遮蔽用障害物リストを作成（RectangleObstacle形式）
    shielding_obstacles = []
    for rect in obstacle_rects:
        shielding_obstacles.append(RectangleObstacle(
            rect.x0, rect.x1, rect.y0, rect.y1,
            mu=rect.mu, color=rect.color, alpha=rect.alpha
        ))

    # 遮蔽を考慮した放射線強度フィールドを生成（各線源の個別強度を使用）
    true_intensity, sources_grid, sources_intensity_list = generate_radiation_field_with_shielding(
        sources, x_min, x_max, y_min, y_max, GRID_SIZE,
        obstacles=shielding_obstacles, intensity=source_intensities
    )
    sim_observed_counts = add_poisson_noise(true_intensity)

    print(f"True sources (grid): {sources_grid}")
    print(f"Source intensities: {sources_intensity_list}")

    # マップ全体の放射線強度をPNGに保存
    save_intensity_map_viz(
        true_intensity, sources_grid, shielding_obstacles,
        x_min, x_max, y_min, y_max,
        os.path.join(PNG_DIR, f"radiation_intensity_map_{EXPERIMENT_VERSION}.png"),
        title="Radiation Intensity Map (with Shielding)",
        use_log_scale=True
    )
    print(f"Saved radiation intensity map to {PNG_DIR}")

    # ---------------------------
    # STEP I, J: 統合GIF生成
    # ---------------------------
    print("\n" + "=" * 60)
    print("Generating integrated Exploration GIF with unknown environment...")
    print("=" * 60)

    integrated_frames, integrated_gpr_peaks, robot_distances, task_assignment_log = create_integrated_gif_with_unknown_env(
        robot_start_grid, peaks_xy, peaks_val,
        true_grid, sim_observed_counts, sources_grid,
        obstacle_rects, SENSOR_RANGE, GRID_SIZE,
        x_min, x_max, y_min, y_max,
        search_radius=SEARCH_RADIUS,
        max_gpr_iters=MAX_GPR_ITERS,
        predict_scale=PREDICT_SCALE,
        predict_margin_cells=PREDICT_MARGIN_CELLS,
        use_gpr_step_limit=USE_GPR_STEP_LIMIT,
        max_gpr_steps=MAX_GPR_STEPS,
        rho=rho, alpha=alpha, beta=beta, gamma=gamma,
        display_x_min=GIF_DISPLAY_X_MIN, display_x_max=GIF_DISPLAY_X_MAX,
        display_y_min=GIF_DISPLAY_Y_MIN, display_y_max=GIF_DISPLAY_Y_MAX
    )

    if len(integrated_frames) > 0:
        integrated_gif_path = os.path.join(GIF_DIR, f"integrated_Exploration_{EXPERIMENT_VERSION}.gif")
        imageio.mimsave(integrated_gif_path, integrated_frames, fps=6, loop=0)
        print(f"\nSaved integrated Exploration GIF: {integrated_gif_path} ({len(integrated_frames)} frames)")

        # ゴーストと真の線源を分類して表示
        # タプル構造:
        # TRUE SOURCE: 7要素 (peak_x, peak_y, robot_id, obs_id, is_ghost, estimated_intensity, actual_intensity)
        # GHOST: 9要素 (peak_x, peak_y, robot_id, obs_id, is_ghost, orig_x, orig_y, estimated_intensity, actual_intensity)
        num_ghosts = sum(1 for peak_data in integrated_gpr_peaks if peak_data[4])
        num_true = len(integrated_gpr_peaks) - num_ghosts
        print(f"Detected {len(integrated_gpr_peaks)} peaks from GPR exploration ({num_true} true, {num_ghosts} ghost):")
        for peak_data in integrated_gpr_peaks:
            peak_x, peak_y, robot_id, obs_id, is_ghost = peak_data[:5]
            if is_ghost:
                # ゴースト: 9要素 - 元のタスク位置と強度も表示
                orig_x, orig_y = peak_data[5], peak_data[6]
                estimated_int, actual_int = peak_data[7], peak_data[8]
                print(f"  Task{obs_id + 1} (Robot {robot_id}): GHOST - Original task at ({orig_x:.2f}, {orig_y:.2f}), "
                      f"GPR peak at ({peak_x:.2f}, {peak_y:.2f}), estimated={estimated_int:.2e}, actual={actual_int:.2e}")
            else:
                # 真の線源: 7要素 - GPRピーク位置と強度を表示
                estimated_int, actual_int = peak_data[5], peak_data[6]
                print(f"  Task{obs_id + 1} (Robot {robot_id}): TRUE SOURCE - GPR peak at ({peak_x:.2f}, {peak_y:.2f}), "
                      f"estimated={estimated_int:.2e}, actual={actual_int:.2e}")

        print(f"\nRobot travel distances:")
        for robot_id, distance in robot_distances.items():
            print(f"  Robot {robot_id}: {distance:.2f} cells")

    # ---------------------------
    # 結果ログ保存
    # ---------------------------
    save_results_log(sources_grid, sources_intensity_list, integrated_gpr_peaks, robot_distances,
                     x_min, x_max, y_min, y_max, GRID_SIZE, EXPERIMENT_VERSION)

    print("\n=== Pipeline complete ===")


def save_results_log(sources_grid, sources_intensity_list, integrated_gpr_peaks, robot_distances,
                     x_min, x_max, y_min, y_max, grid_size, experiment_version):
    """結果ログを保存する"""
    log_path = os.path.join(TXT_DIR, f"gpr_results_log_{experiment_version}.txt")

    # ゴーストと真の線源を分類
    # TRUE SOURCE: 7要素 (peak_x, peak_y, robot_id, obs_id, is_ghost, estimated_int, actual_int)
    # GHOST: 9要素 (peak_x, peak_y, robot_id, obs_id, is_ghost, orig_x, orig_y, estimated_int, actual_int)
    true_peaks = [(pd[0], pd[1], pd[2], pd[3], pd[5], pd[6]) for pd in integrated_gpr_peaks if not pd[4]]
    ghost_peaks = [(pd[0], pd[1], pd[2], pd[3], pd[5], pd[6], pd[7], pd[8]) for pd in integrated_gpr_peaks if pd[4]]

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Multi-Robot GPR Exploration with Unknown Environment - Results Log\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Experiment Version: {experiment_version}\n")
        f.write(f"Number of Robots: {NUM_ROBOTS}\n")
        f.write(f"Grid Size: {GRID_SIZE} x {GRID_SIZE}\n")
        f.write(f"LiDAR Sensor Range: {SENSOR_RANGE} cells\n")
        f.write(f"Search Radius: {SEARCH_RADIUS} (measurement area: {2*SEARCH_RADIUS+1}x{2*SEARCH_RADIUS+1})\n")
        f.write(f"Max GPR Iterations: {MAX_GPR_ITERS}\n")
        f.write(f"GPR Step Limit: {'Enabled' if USE_GPR_STEP_LIMIT else 'Disabled'}\n")
        if USE_GPR_STEP_LIMIT:
            f.write(f"Max GPR Steps: {MAX_GPR_STEPS}\n")
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("ROBOT TRAVEL DISTANCES\n")
        f.write("-" * 70 + "\n")
        total_distance = 0.0
        for robot_id, distance in robot_distances.items():
            f.write(f"Robot {robot_id}: {distance:.2f} cells\n")
            total_distance += distance
        f.write(f"\nTotal distance (all robots): {total_distance:.2f} cells\n\n")

        f.write("-" * 70 + "\n")
        f.write("TRUE SOURCE POSITIONS AND INTENSITIES\n")
        f.write("-" * 70 + "\n")
        for i, (sx, sy) in enumerate(sources_grid):
            wx, wy = grid_to_world(sx, sy, x_min, x_max, y_min, y_max, grid_size)
            intensity = sources_intensity_list[i] if i < len(sources_intensity_list) else 0.0
            f.write(f"Source {i+1}:\n")
            f.write(f"  Grid: ({sx}, {sy})\n")
            f.write(f"  World: ({wx:.2f}, {wy:.2f})\n")
            f.write(f"  Intensity: {intensity:.2e}\n\n")

        # ゴースト検出結果のサマリー
        f.write("-" * 70 + "\n")
        f.write("GHOST DETECTION SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total peaks detected: {len(integrated_gpr_peaks)}\n")
        f.write(f"  True sources: {len(true_peaks)}\n")
        f.write(f"  Ghosts: {len(ghost_peaks)}\n\n")

        # 真の線源として検出されたピーク
        f.write("-" * 70 + "\n")
        f.write("DETECTED TRUE SOURCES\n")
        f.write("-" * 70 + "\n")
        if true_peaks:
            for idx, (peak_x, peak_y, robot_id, obs_id, estimated_int, actual_int) in enumerate(true_peaks):
                wx, wy = grid_to_world(peak_x, peak_y, x_min, x_max, y_min, y_max, grid_size)
                f.write(f"True Peak {idx+1} (Task{obs_id + 1}, Robot {robot_id}):\n")
                f.write(f"  Grid: ({peak_x:.3f}, {peak_y:.3f})\n")
                f.write(f"  World: ({wx:.3f}, {wy:.3f})\n")
                f.write(f"  GPR Estimated Intensity: {estimated_int:.2e}\n")
                f.write(f"  Actual Intensity at Peak: {actual_int:.2e}\n\n")
        else:
            f.write("No true sources detected.\n\n")

        # ゴーストとして検出されたピーク
        f.write("-" * 70 + "\n")
        f.write("DETECTED GHOSTS\n")
        f.write("-" * 70 + "\n")
        if ghost_peaks:
            for idx, (peak_x, peak_y, robot_id, obs_id, orig_x, orig_y, estimated_int, actual_int) in enumerate(ghost_peaks):
                gpr_wx, gpr_wy = grid_to_world(peak_x, peak_y, x_min, x_max, y_min, y_max, grid_size)
                orig_wx, orig_wy = grid_to_world(orig_x, orig_y, x_min, x_max, y_min, y_max, grid_size)
                f.write(f"Ghost {idx+1} (Task{obs_id + 1}, Robot {robot_id}):\n")
                f.write(f"  Original Task Position:\n")
                f.write(f"    Grid: ({orig_x:.3f}, {orig_y:.3f})\n")
                f.write(f"    World: ({orig_wx:.3f}, {orig_wy:.3f})\n")
                f.write(f"  Final GPR Peak Position:\n")
                f.write(f"    Grid: ({peak_x:.3f}, {peak_y:.3f})\n")
                f.write(f"    World: ({gpr_wx:.3f}, {gpr_wy:.3f})\n")
                f.write(f"  GPR Estimated Intensity: {estimated_int:.2e}\n")
                f.write(f"  Actual Intensity at Peak: {actual_int:.2e}\n")
                f.write(f"  Reason: GPR step limit reached (did not converge)\n\n")
        else:
            f.write("No ghosts detected.\n\n")

        # 真の線源との比較（真の検出のみ使用）
        f.write("-" * 70 + "\n")
        f.write("COMPARISON WITH TRUE SOURCES (excluding ghosts)\n")
        f.write("-" * 70 + "\n")
        for i, (sx, sy) in enumerate(sources_grid):
            wx, wy = grid_to_world(sx, sy, x_min, x_max, y_min, y_max, grid_size)
            intensity = sources_intensity_list[i] if i < len(sources_intensity_list) else 0.0
            f.write(f"True Source {i+1} at ({sx}, {sy}) [grid], ({wx:.2f}, {wy:.2f}) [world], intensity={intensity:.2e}:\n")

            if true_peaks:
                distances = [np.sqrt((sx-px)**2 + (sy-py)**2) for px, py, _, _, _, _ in true_peaks]
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                px, py, rid, oid, est_int, act_int = true_peaks[min_idx]
                pwx, pwy = grid_to_world(px, py, x_min, x_max, y_min, y_max, grid_size)

                f.write(f"  Closest detected peak: ({px:.3f}, {py:.3f}) [grid], ({pwx:.3f}, {pwy:.3f}) [world]\n")
                f.write(f"  Distance: {min_dist:.3f} (grid)\n")
                f.write(f"  Detected by: Robot {rid}, Task{oid + 1}\n")
                f.write(f"  GPR Estimated Intensity: {est_int:.2e}\n")
                f.write(f"  Actual Intensity at Peak: {act_int:.2e}\n")
            else:
                f.write("  No true peaks detected to compare.\n")
            f.write("\n")

    print(f"Saved results log: {log_path}")


if __name__ == "__main__":
    main()
