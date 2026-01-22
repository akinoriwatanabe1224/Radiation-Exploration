# visualization/gif_generator.py
# 統合GIF生成
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from path_planning.dstar_lite import DStarLite
from sensors.lidar import reveal_with_lidar
from radiation.gpr import fit_gpr
from task_allocation.auction import run_dynamic_auction
from utils.coordinate import world_to_grid, grid_to_world


def save_direction_viz(sources, measurements, ray_data, x_min, x_max, y_min, y_max, output_path):
    """
    方向レイの可視化を保存する

    Parameters
    ----------
    sources : ndarray
        線源位置
    measurements : ndarray
        測定点位置
    ray_data : list
        レイデータ [(msr, end, color, alpha), ...]
    x_min, x_max, y_min, y_max : float
        座標範囲
    output_path : str
        出力パス
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(sources[:, 0], sources[:, 1], c='k', marker='x', label='true sources')
    plt.scatter(measurements[:, 0], measurements[:, 1], c='r', marker='x', label='measure points')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)

    for msr, end, color, alpha in ray_data:
        plt.plot([msr[0], end[0]], [msr[1], end[1]], '-', color=color, alpha=alpha)

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def save_heatmap_viz(heatmap, Xc, Yc, peaks_xy, sources, measurements,
                     x_min, x_max, y_min, y_max, output_path):
    """
    ヒートマップの可視化を保存する

    Parameters
    ----------
    heatmap : ndarray
        ヒートマップ
    Xc, Yc : ndarray
        メッシュグリッド
    peaks_xy : ndarray
        ピーク座標
    sources : ndarray
        線源位置
    measurements : ndarray
        測定点位置
    x_min, x_max, y_min, y_max : float
        座標範囲
    output_path : str
        出力パス
    """
    plt.figure(figsize=(6, 6))
    plt.contourf(Xc, Yc, heatmap, levels=100, cmap='jet')
    plt.scatter(peaks_xy[:, 0], peaks_xy[:, 1], c='white', s=80, edgecolor='k', label='peaks')
    plt.scatter(sources[:, 0], sources[:, 1], c='k', marker='x', s=100, label='true sources')
    plt.scatter(measurements[:, 0], measurements[:, 1], c='r', marker='x', label='measure points')
    # plt.title("von Mises heatmap (SUM) + peaks")
    plt.colorbar()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def save_heatmap_only_viz(heatmap, Xc, Yc, x_min, x_max, y_min, y_max, output_path):
    """
    von Misesヒートマップのみを保存する（ピークや線源なし）

    Parameters
    ----------
    heatmap : ndarray
        ヒートマップ
    Xc, Yc : ndarray
        メッシュグリッド
    x_min, x_max, y_min, y_max : float
        座標範囲
    output_path : str
        出力パス
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # ヒートマップを表示（zorder=1: 最下層）
    contour = ax.contourf(Xc, Yc, heatmap, levels=100, cmap='jet', zorder=1)
    plt.colorbar(contour, ax=ax, label='von Mises score')

    # ax.set_title("von Mises heatmap")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_heatmap_with_obstacles_viz(heatmap, Xc, Yc, peaks_xy, sources, measurements,
                                     obstacle_rects, x_min, x_max, y_min, y_max, output_path):
    """
    von Misesヒートマップと障害物の重ね図を保存する
    レイヤー順序: ヒートマップ(最下層) -> 障害物(中間) -> ピーク/点(最上層)

    Parameters
    ----------
    heatmap : ndarray
        ヒートマップ
    Xc, Yc : ndarray
        メッシュグリッド
    peaks_xy : ndarray
        ピーク座標
    sources : ndarray
        線源位置
    measurements : ndarray
        測定点位置
    obstacle_rects : list
        RectangleObstacleのリスト
    x_min, x_max, y_min, y_max : float
        座標範囲
    output_path : str
        出力パス
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # ヒートマップを表示（zorder=1: 最下層）
    contour = ax.contourf(Xc, Yc, heatmap, levels=100, cmap='jet', zorder=1)
    plt.colorbar(contour, ax=ax, label='von Mises score')

    # 障害物を描画（zorder=5: 中間層）
    for obs in obstacle_rects:
        ax.fill(
            [obs.x0, obs.x1, obs.x1, obs.x0],
            [obs.y0, obs.y0, obs.y1, obs.y1],
            color=obs.color, alpha=1.0, zorder=5
        )

    # ピーク、線源、測定点を描画（zorder=10: 最上層）
    ax.scatter(peaks_xy[:, 0], peaks_xy[:, 1], c='white', s=80, edgecolor='k', label='peaks', zorder=10)
    ax.scatter(sources[:, 0], sources[:, 1], c='k', marker='x', s=100, label='true sources', zorder=10)
    ax.scatter(measurements[:, 0], measurements[:, 1], c='r', marker='x', label='measure points', zorder=10)

    # ax.set_title("von Mises heatmap + obstacles")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_intensity_map_viz(intensity_field, sources_grid, obstacle_rects,
                           x_min, x_max, y_min, y_max, output_path,
                           title="Radiation Intensity Map (with Shielding)",
                           use_log_scale=True):
    """
    放射線強度マップの可視化を保存する

    Parameters
    ----------
    intensity_field : ndarray
        放射線強度フィールド
    sources_grid : list
        線源のグリッド座標リスト [(gx, gy), ...]
    obstacle_rects : list
        RectangleObstacleのリスト
    x_min, x_max, y_min, y_max : float
        座標範囲
    output_path : str
        出力パス
    title : str
        タイトル
    use_log_scale : bool
        対数スケールを使用するか
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 強度マップを表示
    clipped = np.clip(intensity_field, 1, None)
    if use_log_scale:
        im = ax.imshow(clipped, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap='inferno', norm='log',
                       vmin=1, vmax=np.max(clipped))
    else:
        im = ax.imshow(intensity_field, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap='inferno')

    plt.colorbar(im, ax=ax, label='Intensity (counts)')

    # 障害物を描画（完全不透明: alpha=1.0）
    for obs in obstacle_rects:
        ax.fill(
            [obs.x0, obs.x1, obs.x1, obs.x0],
            [obs.y0, obs.y0, obs.y1, obs.y1],
            color=obs.color, alpha=1.0, zorder=5
        )

    # 線源位置を描画
    if sources_grid:
        sx = [p[0] for p in sources_grid]
        sy = [p[1] for p in sources_grid]
        ax.scatter(sx, sy, marker='*', s=300, color='cyan',
                   edgecolors='k', linewidths=2, zorder=10, label='Sources')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    # ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_integrated_gif_with_unknown_env(robot_start_grid, peaks_xy, peaks_val,
                                           true_grid, sim_observed_counts, sources_grid,
                                           obstacle_rects, sensor_range, grid_size,
                                           x_min, x_max, y_min, y_max,
                                           search_radius=1, max_gpr_iters=8,
                                           predict_scale=3.0, predict_margin_cells=3.0,
                                           use_gpr_step_limit=False, max_gpr_steps=50,
                                           rho=2.0, alpha=1.0, beta=0.2, gamma=0.5,
                                           display_x_min=None, display_x_max=None,
                                           display_y_min=None, display_y_max=None):
    """
    未知環境でLiDARを使いながら複数ロボットが同時に動き、GPR探査を行う統合GIF
    動的タスク割り当てを使用

    Parameters
    ----------
    robot_start_grid : list
        各ロボットの開始位置（グリッド座標）
    peaks_xy : ndarray
        タスク点（ピーク位置）の座標配列（ワールド座標）shape=(n_points, 2)
    peaks_val : ndarray
        各ピークの強度値
    true_grid : ndarray
        真の環境グリッド
    sim_observed_counts : ndarray
        放射線観測カウント
    sources_grid : list
        線源のグリッド座標
    obstacle_rects : list
        障害物リスト
    sensor_range : int
        センサ範囲
    grid_size : int
        グリッドサイズ
    x_min, x_max, y_min, y_max : float
        ワールド座標の範囲
    search_radius : int
        GPR探査半径
    max_gpr_iters : int
        最大GPR反復回数
    predict_scale : float
        予測範囲の拡大倍率
    predict_margin_cells : float
        予測範囲のマージン
    use_gpr_step_limit : bool
        GPR探査中のステップ数上限を有効にするか
    max_gpr_steps : int
        GPR探査中の最大ステップ数
    rho : float
        混雑度ペナルティの減衰パラメータ
    alpha : float
        重み係数
    beta : float
        移動コスト係数
    gamma : float
        混雑度係数
    display_x_min : float or None
        GIF表示範囲のX最小値（Noneの場合は自動）
    display_x_max : float or None
        GIF表示範囲のX最大値（Noneの場合は自動）
    display_y_min : float or None
        GIF表示範囲のY最小値（Noneの場合は自動）
    display_y_max : float or None
        GIF表示範囲のY最大値（Noneの場合は自動）

    Returns
    -------
    list
        フレーム画像のリスト
    list
        完了したGPRピークの情報リスト
    dict
        各ロボットの移動距離
    dict
        タスク割り当て結果（ロボットID -> 完了したタスクIDリスト）
    """
    H, W = true_grid.shape
    overall_frames = []
    completed_gpr_peaks = []
    num_robots = len(robot_start_grid)

    # GIF表示範囲の設定（Noneの場合はデフォルト値を使用）
    disp_x_min = display_x_min if display_x_min is not None else -0.5
    disp_x_max = display_x_max if display_x_max is not None else grid_size - 0.5
    disp_y_min = display_y_min if display_y_min is not None else -0.5
    disp_y_max = display_y_max if display_y_max is not None else grid_size - 0.5

    # 未知環境マスク（全ロボット共通）
    global_discovered = np.zeros_like(true_grid, dtype=bool)

    # 動的タスク管理
    # 全タスクを登録（障害物チェックは探査時に行う）
    num_tasks = len(peaks_xy)
    task_positions_grid = []
    for i in range(num_tasks):
        gpos = world_to_grid(peaks_xy[i], x_min, x_max, y_min, y_max, grid_size)
        task_positions_grid.append(gpos)

    # オークション用の配列
    task_positions_world = np.array(peaks_xy)  # ワールド座標
    task_values = np.array(peaks_val)  # ピーク値

    unassigned_tasks = set(range(num_tasks))  # 未割り当てタスク
    unreachable_tasks = set()  # 到達不能と判断されたタスク
    obstacle_tasks = set()  # 障害物内にあると発見されたタスク
    task_fail_count = {t: 0 for t in range(num_tasks)}  # 各タスクの到達失敗回数
    MAX_TASK_FAIL_COUNT = 3  # この回数失敗したら到達不能と判断
    task_assignment_log = {r: [] for r in range(num_robots)}  # 各ロボットが完了したタスク

    # Robot states
    robot_states = []
    for r in range(num_robots):
        pos = robot_start_grid[r]
        robot_states.append({
            'id': r,
            'position': pos,
            'start_pos': pos,
            'current_task_id': None,  # 現在担当中のタスクID
            'current_goal': None,     # 現在の目標位置（グリッド座標）
            'path': None,
            'path_idx': 0,
            'state': 'IDLE',
            'move_target': None,
            'gpr_cumulative': {},
            'gpr_center': None,
            'gpr_pred': None,
            'gpr_iter': 0,
            'gpr_targets': [],
            'gpr_target_idx': 0,
            'gpr_measuring': False,
            'replan_count': 0,
            'color': f'C{r}',
            'total_distance': 0.0,
            'gpr_steps': 0
        })
        # 初期位置でLiDAR観測
        reveal_with_lidar(pos, global_discovered, true_grid, obstacle_rects, sensor_range)

    def get_in_progress_positions():
        """他のロボットが担当中または向かっている位置を取得（ワールド座標）"""
        positions = []
        for rs in robot_states:
            if rs['current_task_id'] is not None:
                positions.append(peaks_xy[rs['current_task_id']])
        return positions

    def get_robot_world_position(rs):
        """ロボットの現在位置をワールド座標で取得"""
        gx, gy = rs['position']
        return grid_to_world(gx, gy, x_min, x_max, y_min, y_max, grid_size)

    def fit_gpr_for_robot(rs):
        """Fit GPR for a robot's cumulative measurements"""
        if len(rs['gpr_cumulative']) == 0:
            return None
        return fit_gpr(
            rs['gpr_cumulative'], W, H,
            predict_scale=predict_scale,
            predict_margin_cells=predict_margin_cells,
            gpr_fine=100
        )

    def make_overall_frame(frame_label=""):
        """Generate a frame showing all robots, unknown areas, and GPR states"""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')

        # 背景：未知領域=グレー(0.5)、既知の自由空間=白(1.0)、既知の障害物=黒(0.0)
        disp = np.full_like(true_grid, 0.5, dtype=float)
        disp[global_discovered] = 1.0
        disp[global_discovered & (true_grid == 1)] = 0.0

        ax.imshow(disp, cmap='gray', origin='lower',
                  extent=[0, grid_size, 0, grid_size], alpha=0.6, vmin=0, vmax=1)

        # グリッド線
        grid_interval = 1
        label_interval = 5
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(grid_interval))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(grid_interval))
        ax.grid(which="minor", color="lightgray", linewidth=0.5, zorder=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(label_interval))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(label_interval))
        ax.grid(which="major", color="gray", linewidth=1.0, zorder=0)

        # 障害物を描画
        for rect in obstacle_rects:
            rect.draw(ax, zorder=5)

        # GPRヒートマップ（アクティブなロボット）
        for rs in robot_states:
            if rs['gpr_pred'] is not None:
                pred = rs['gpr_pred']
                ax.imshow(
                    pred["Zi"],
                    extent=[pred["pred_min_x"], pred["pred_max_x"],
                            pred["pred_min_y"], pred["pred_max_y"]],
                    origin='lower',
                    cmap='inferno',
                    alpha=0.55,
                    vmin=0,
                    vmax=np.nanmax(pred["Zi"]),
                    zorder=5,
                    aspect='equal'
                )

                # ピーク
                ax.scatter(
                    pred["peak_x"], pred["peak_y"],
                    marker='x', s=150, linewidths=3,
                    color=rs['color'], edgecolors='white', zorder=20
                )

                # 測定領域
                if rs['gpr_center'] is not None:
                    cx, cy = rs['gpr_center']
                    # セル中央基準で描画（+0.5オフセット）
                    rect = Rectangle(
                        (cx + 0.5 - search_radius - 0.5, cy + 0.5 - search_radius - 0.5),
                        2 * search_radius + 1, 2 * search_radius + 1,
                        fill=False,
                        edgecolor=rs['color'], linewidth=2.5,
                        linestyle='-', alpha=0.9,
                        zorder=15
                    )
                    ax.add_patch(rect)

        # 完了したGPRピーク（ゴーストと真の線源を区別）
        if completed_gpr_peaks:
            for peak_data in completed_gpr_peaks:
                peak_x, peak_y, robot_id, obs_id, is_ghost = peak_data[:5]
                if is_ghost:
                    # ゴースト: 元のタスク位置に表示（グレー + ×マーカー）
                    orig_x, orig_y = peak_data[5], peak_data[6]
                    ax.scatter(orig_x, orig_y, marker='X', s=200,
                               color='gray', edgecolors='white', linewidths=2,
                               zorder=22, alpha=0.7)
                    ax.text(orig_x + 0.8, orig_y + 0.8, f'Task{obs_id + 1}\n(Ghost)', fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.25', facecolor='lightgray',
                                      edgecolor='gray', linewidth=1.5, alpha=0.85))
                else:
                    # 真の線源: GPRピーク位置に表示（ロボット色 + Pマーカー）
                    ax.scatter(peak_x, peak_y, marker='P', s=180,
                               color=f'C{robot_id}', edgecolors='white', linewidths=2,
                               zorder=22, alpha=0.9)
                    ax.text(peak_x + 0.8, peak_y + 0.8, f'Task{obs_id + 1}', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                      edgecolor=f'C{robot_id}', linewidth=1.5, alpha=0.85))

        # 観測点（未完了・進行中）
        completed_task_ids = set(peak_data[3] for peak_data in completed_gpr_peaks)
        in_progress_task_ids = set(rs['current_task_id'] for rs in robot_states if rs['current_task_id'] is not None)

        for task_id in range(num_tasks):
            gx, gy = task_positions_grid[task_id]
            # セル中央に描画（+0.5オフセット）
            gx_disp, gy_disp = gx + 0.5, gy + 0.5

            if task_id in completed_task_ids:
                # 完了済み - completed_gpr_peaksで描画済み
                continue
            elif task_id in in_progress_task_ids:
                # 進行中のタスク（ロボットが向かっている or GPR探査中）
                robot_color = None
                for rs in robot_states:
                    if rs['current_task_id'] == task_id:
                        robot_color = rs['color']
                        break
                ax.plot(gx_disp, gy_disp, 'o', color='orange', markersize=11,
                        markeredgecolor=robot_color if robot_color else 'k',
                        markeredgewidth=2, alpha=0.9, zorder=10)
                ax.text(gx_disp, gy_disp - 1.4, f'Task{task_id + 1}', fontsize=8, ha='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='orange',
                                  edgecolor=robot_color if robot_color else 'k',
                                  linewidth=1.5, alpha=0.9))
            else:
                # 未割り当てタスク
                ax.plot(gx_disp, gy_disp, 'o', color='gold', markersize=11,
                        markeredgecolor='k', markeredgewidth=2, alpha=0.7, zorder=10)
                ax.text(gx_disp, gy_disp - 1.4, f'Task{task_id + 1}', fontsize=8, ha='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='gold',
                                  edgecolor='k', linewidth=1.5, alpha=0.7))

        # ロボット描画
        for rs in robot_states:
            x, y = rs['position']
            # セル中央に描画（+0.5オフセット）
            x_disp, y_disp = x + 0.5, y + 0.5

            # 開始位置
            if 'start_pos' in rs:
                sx, sy = rs['start_pos']
                # セル中央に描画（+0.5オフセット）
                sx_disp, sy_disp = sx + 0.5, sy + 0.5
                ax.scatter(sx_disp, sy_disp, marker='x', s=110,
                           color='white', linewidths=2.5, zorder=5)
                ax.text(sx_disp, sy_disp + 1.2, f"Start R{rs['id']}", fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

            # ロボットマーカー
            if rs['state'] == 'COMPLETED':
                marker_style = 's'
                marker_size = 13
                alpha = 0.6
            elif rs['state'] == 'GPR_EXPLORING':
                marker_style = 'D'
                marker_size = 15
                alpha = 1.0
            else:
                marker_style = 'o'
                marker_size = 13
                alpha = 1.0

            ax.plot(x_disp, y_disp, marker_style, color=rs['color'],
                    markersize=marker_size, markeredgecolor='white',
                    markeredgewidth=2.5, alpha=alpha, zorder=30)

            # ロボットラベル
            ax.text(x_disp, y_disp + 1.5, f"R{rs['id']}", fontsize=11, ha='center',
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=rs['color'],
                              edgecolor='white', linewidth=2, alpha=0.95))

            # 計画経路
            if rs['path'] is not None and rs['state'] == 'MOVING':
                remaining_path = rs['path'][rs['path_idx']:]
                if len(remaining_path) > 1:
                    # セル中央に描画（+0.5オフセット）
                    xs = [p[0] + 0.5 for p in remaining_path]
                    ys = [p[1] + 0.5 for p in remaining_path]
                    ax.plot(xs, ys, '--', color=rs['color'], alpha=0.6, linewidth=2.5, zorder=8)

        # 真の線源
        if sources_grid:
            sx = [p[0] for p in sources_grid]
            sy = [p[1] for p in sources_grid]
            ax.scatter(sx, sy, marker='*', s=500, color='cyan',
                       edgecolors='k', linewidths=3, zorder=35)
            for i, (sx_i, sy_i) in enumerate(sources_grid):
                ax.text(sx_i, sy_i - 1.8, f'Source{i + 1}', fontsize=10, ha='center',
                        fontweight='bold', color='k',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='cyan',
                                  edgecolor='k', linewidth=2, alpha=0.95))

        # タイトルとステータス（設定された表示範囲を使用）
        ax.set_xlim(disp_x_min, disp_x_max)
        ax.set_ylim(disp_y_min, disp_y_max)
        ax.set_xlabel('x [m]', fontsize=12)
        ax.set_ylabel('y [m]', fontsize=12)
        ax.set_title(frame_label, fontsize=14, fontweight='bold', pad=20)

        # 凡例
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=10, label='Unknown', markeredgecolor='k'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
                   markersize=10, label='Known Free', markeredgecolor='k'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                   markersize=10, label='Known Obstacle', markeredgecolor='k')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(h, w, 4)[:, :, :3]
        plt.close(fig)
        return img

    # メインシミュレーションループ
    step = 0
    max_steps = 1000

    while step < max_steps:
        step += 1
        any_active = False

        for rs in robot_states:
            if rs['state'] == 'COMPLETED':
                continue

            any_active = True

            # IDLE: 動的オークションで次のタスクを取得
            if rs['state'] == 'IDLE':
                rs['replan_count'] = 0

                # 未割り当てタスクがなければ完了
                if len(unassigned_tasks) == 0:
                    rs['state'] = 'COMPLETED'
                    continue

                # 動的オークション実行
                robot_world_pos = get_robot_world_position(rs)
                in_progress_positions = get_in_progress_positions()

                task_id, bid = run_dynamic_auction(
                    rs['id'],
                    robot_world_pos,
                    list(unassigned_tasks),
                    task_positions_world,  # 有効タスクのワールド座標
                    in_progress_positions,
                    task_values,  # 有効タスクのピーク値
                    rho=rho, alpha=alpha, beta=beta, gamma=gamma
                )

                if task_id is None:
                    rs['state'] = 'COMPLETED'
                    continue

                # タスクを割り当て
                unassigned_tasks.discard(task_id)
                rs['current_task_id'] = task_id
                goal = task_positions_grid[task_id]
                rs['current_goal'] = goal

                # タスク位置が既知の障害物かチェック
                gx, gy = goal
                if global_discovered[gy, gx] and true_grid[gy, gx] == 1:
                    print(f"  [Auction] Robot {rs['id']}: Task{task_id + 1} at {goal} is on known obstacle, removing task")
                    obstacle_tasks.add(task_id)
                    rs['current_task_id'] = None
                    rs['current_goal'] = None
                    continue

                print(f"  [Auction] Robot {rs['id']} bid for Task{task_id + 1} (bid: {bid:.4f})")

                # 既知情報で経路計画
                known_grid = np.where(global_discovered, true_grid, 0).astype(int)
                try:
                    dstar = DStarLite(known_grid, rs['position'], goal)
                    dstar.compute_shortest_path()
                    path = dstar.extract_path()
                except:
                    path = None

                if path:
                    rs['path'] = path
                    rs['path_idx'] = 0
                    rs['move_target'] = ('obs', goal)
                    rs['state'] = 'MOVING'
                    print(f"  Robot {rs['id']}: Planning path to Task{task_id + 1} at {goal} (len {len(path)})")
                else:
                    # 到達失敗回数をインクリメント
                    task_fail_count[task_id] += 1
                    if task_fail_count[task_id] >= MAX_TASK_FAIL_COUNT:
                        print(f"  Robot {rs['id']}: Cannot reach Task{task_id + 1} at {goal}, marking as UNREACHABLE")
                        unreachable_tasks.add(task_id)
                        unassigned_tasks.discard(task_id)
                    else:
                        print(f"  Robot {rs['id']}: Cannot reach Task{task_id + 1} at {goal}, skipping (fail {task_fail_count[task_id]}/{MAX_TASK_FAIL_COUNT})")
                        unassigned_tasks.add(task_id)
                    rs['current_task_id'] = None
                    rs['current_goal'] = None

            # MOVING: 経路に沿って移動
            elif rs['state'] == 'MOVING':
                if rs['path_idx'] < len(rs['path']):
                    next_pos = rs['path'][rs['path_idx']]
                    nx, ny = next_pos

                    # 障害物チェック
                    if true_grid[ny, nx] == 1:
                        if not global_discovered[ny, nx]:
                            global_discovered[ny, nx] = True
                            print(f"  Robot {rs['id']}: Discovered obstacle at {next_pos}, replanning")
                        else:
                            print(f"  Robot {rs['id']}: Path includes known obstacle at {next_pos}, replanning")

                        # ゴール位置自体が障害物かチェック
                        goal = rs['move_target'][1]
                        gx, gy = goal
                        if global_discovered[gy, gx] and true_grid[gy, gx] == 1:
                            tid = rs['current_task_id']
                            if tid is not None:
                                print(f"  Robot {rs['id']}: Task{tid + 1} position is on obstacle, removing task")
                                obstacle_tasks.add(tid)
                                unassigned_tasks.discard(tid)
                            rs['current_task_id'] = None
                            rs['current_goal'] = None
                            rs['state'] = 'IDLE'
                            rs['replan_count'] = 0
                            continue

                        rs['replan_count'] += 1
                        if rs['replan_count'] > 10:
                            tid = rs['current_task_id']
                            if tid is not None:
                                task_fail_count[tid] += 1
                                if task_fail_count[tid] >= MAX_TASK_FAIL_COUNT:
                                    print(f"  Robot {rs['id']}: Too many replans, marking Task{tid + 1} as UNREACHABLE")
                                    unreachable_tasks.add(tid)
                                    unassigned_tasks.discard(tid)
                                else:
                                    print(f"  Robot {rs['id']}: Too many replans, skipping Task{tid + 1} (fail {task_fail_count[tid]}/{MAX_TASK_FAIL_COUNT})")
                                    unassigned_tasks.add(tid)
                            rs['current_task_id'] = None
                            rs['current_goal'] = None
                            rs['state'] = 'IDLE'
                            rs['replan_count'] = 0
                            continue

                        # 再計画
                        goal = rs['move_target'][1]
                        known_grid = np.where(global_discovered, true_grid, 0).astype(int)
                        try:
                            dstar = DStarLite(known_grid, rs['position'], goal)
                            dstar.compute_shortest_path()
                            new_path = dstar.extract_path()
                        except:
                            new_path = None

                        if new_path:
                            rs['path'] = new_path
                            rs['path_idx'] = 0
                        else:
                            tid = rs['current_task_id']
                            if tid is not None:
                                task_fail_count[tid] += 1
                                if task_fail_count[tid] >= MAX_TASK_FAIL_COUNT:
                                    print(f"  Robot {rs['id']}: Cannot find path, marking Task{tid + 1} as UNREACHABLE")
                                    unreachable_tasks.add(tid)
                                    unassigned_tasks.discard(tid)
                                else:
                                    print(f"  Robot {rs['id']}: Cannot find path, skipping Task{tid + 1} (fail {task_fail_count[tid]}/{MAX_TASK_FAIL_COUNT})")
                                    unassigned_tasks.add(tid)
                            rs['current_task_id'] = None
                            rs['current_goal'] = None
                            rs['state'] = 'IDLE'
                            rs['replan_count'] = 0
                    else:
                        # 移動可能
                        old_pos = rs['position']
                        rs['position'] = next_pos
                        rs['path_idx'] += 1
                        # 移動距離を加算（1セル移動 = 1.0）
                        rs['total_distance'] += np.sqrt((next_pos[0] - old_pos[0])**2 + (next_pos[1] - old_pos[1])**2)
                        reveal_with_lidar(rs['position'], global_discovered, true_grid,
                                          obstacle_rects, sensor_range)

                        # ゴール到達チェック
                        if rs['position'] == rs['move_target'][1]:
                            rs['replan_count'] = 0
                            if rs['move_target'][0] == 'obs':
                                rs['state'] = 'GPR_EXPLORING'
                                rs['gpr_center'] = rs['position']
                                rs['gpr_iter'] = 0
                                rs['gpr_cumulative'] = {}
                                rs['gpr_targets'] = []
                                rs['gpr_target_idx'] = 0
                                rs['gpr_measuring'] = False
                                rs['gpr_steps'] = 0  # GPR探査ステップ数をリセット
                                print(f"  Robot {rs['id']}: Reached Task{rs['current_task_id'] + 1}, starting GPR")
                            elif rs['move_target'][0] == 'center':
                                rs['gpr_center'] = rs['position']
                else:
                    # パスが空になった場合（通常は発生しない）
                    rs['current_task_id'] = None
                    rs['current_goal'] = None
                    rs['state'] = 'IDLE'
                    rs['replan_count'] = 0

            # GPR_EXPLORING: GPR探査
            elif rs['state'] == 'GPR_EXPLORING':
                cx, cy = rs['gpr_center']

                # 測定対象セルリストを作成（初回のみ）
                if not rs['gpr_measuring']:
                    targets = []
                    for dx in range(-search_radius, search_radius + 1):
                        for dy in range(-search_radius, search_radius + 1):
                            tx, ty = cx + dx, cy + dy
                            if 0 <= tx < W and 0 <= ty < H:
                                if true_grid[ty, tx] == 1:
                                    continue
                                if (tx, ty) not in rs['gpr_cumulative']:
                                    targets.append((tx, ty))

                    rs['gpr_targets'] = targets
                    rs['gpr_target_idx'] = 0
                    rs['gpr_measuring'] = True

                # 測定対象セルへ1セルずつ移動して測定
                if rs['gpr_target_idx'] < len(rs['gpr_targets']):
                    target = rs['gpr_targets'][rs['gpr_target_idx']]
                    tx, ty = target

                    if rs['position'] != target:
                        cx_now, cy_now = rs['position']
                        dx = tx - cx_now
                        dy = ty - cy_now

                        if dx != 0:
                            next_pos = (cx_now + (1 if dx > 0 else -1), cy_now)
                        elif dy != 0:
                            next_pos = (cx_now, cy_now + (1 if dy > 0 else -1))
                        else:
                            next_pos = rs['position']

                        nx, ny = next_pos
                        if 0 <= nx < W and 0 <= ny < H and true_grid[ny, nx] == 0:
                            old_pos = rs['position']
                            rs['position'] = next_pos
                            # 移動距離を加算（1セル移動 = 1.0）
                            rs['total_distance'] += np.sqrt((next_pos[0] - old_pos[0])**2 + (next_pos[1] - old_pos[1])**2)
                            rs['gpr_steps'] += 1  # GPR探査ステップ数をインクリメント
                            reveal_with_lidar(rs['position'], global_discovered, true_grid,
                                              obstacle_rects, sensor_range)
                        else:
                            # 障害物にブロックされた場合、このターゲットをスキップ
                            rs['gpr_target_idx'] += 1
                            rs['gpr_steps'] += 1  # スタック防止のためステップをカウント
                            continue

                        # GPRステップ上限チェック（ゴースト判定）
                        if use_gpr_step_limit and rs['gpr_steps'] >= max_gpr_steps:
                                print(f"  Robot {rs['id']}: GPR step limit reached ({rs['gpr_steps']} steps) -> GHOST")
                                # 現在のデータでGPRをフィットして終了（ゴーストとしてマーク）
                                rs['gpr_pred'] = fit_gpr_for_robot(rs)
                                obs_id = rs['current_task_id']
                                is_ghost = True  # ステップ上限到達 = ゴースト
                                # 元のタスク位置を取得
                                orig_x, orig_y = task_positions_grid[obs_id]
                                orig_x_disp, orig_y_disp = orig_x + 0.5, orig_y + 0.5
                                if rs['gpr_pred']:
                                    # GPR推定強度と実際の強度を取得
                                    estimated_intensity = rs['gpr_pred']['peak_val']
                                    peak_gx = int(np.clip(np.round(rs['gpr_pred']['peak_x']), 0, W-1))
                                    peak_gy = int(np.clip(np.round(rs['gpr_pred']['peak_y']), 0, H-1))
                                    actual_intensity = float(sim_observed_counts[peak_gy, peak_gx]) if sim_observed_counts is not None else 0.0
                                    completed_gpr_peaks.append((
                                        rs['gpr_pred']['peak_x'],
                                        rs['gpr_pred']['peak_y'],
                                        rs['id'],
                                        obs_id,
                                        is_ghost,
                                        orig_x_disp,  # 元のタスク位置X
                                        orig_y_disp,  # 元のタスク位置Y
                                        estimated_intensity,  # GPR推定強度
                                        actual_intensity      # 実際の強度
                                    ))
                                    print(f"  Robot {rs['id']}: Completed GPR at Task{obs_id + 1} (GHOST - step limit), "
                                          f"GPR peak at ({rs['gpr_pred']['peak_x']:.2f}, {rs['gpr_pred']['peak_y']:.2f}), "
                                          f"original task at ({orig_x_disp:.2f}, {orig_y_disp:.2f}), "
                                          f"estimated={estimated_intensity:.2e}, actual={actual_intensity:.2e}")
                                # タスク完了を記録
                                task_assignment_log[rs['id']].append(obs_id)
                                rs['current_task_id'] = None
                                rs['current_goal'] = None
                                rs['state'] = 'IDLE'
                                rs['gpr_pred'] = None
                                rs['gpr_targets'] = []
                                rs['gpr_target_idx'] = 0
                                rs['gpr_measuring'] = False
                                continue
                    else:
                        val = float(sim_observed_counts[ty, tx]) if sim_observed_counts is not None else 0.0
                        rs['gpr_cumulative'][(tx, ty)] = val
                        rs['gpr_target_idx'] += 1
                else:
                    # 全セル測定完了
                    rs['gpr_measuring'] = False
                    rs['gpr_pred'] = fit_gpr_for_robot(rs)
                    rs['gpr_iter'] += 1

                    if rs['gpr_iter'] >= max_gpr_iters or len(rs['gpr_targets']) == 0:
                        obs_id = rs['current_task_id']
                        is_ghost = False  # 正常完了 = 真の線源
                        if rs['gpr_pred']:
                            # GPR推定強度と実際の強度を取得
                            estimated_intensity = rs['gpr_pred']['peak_val']
                            peak_gx = int(np.clip(np.round(rs['gpr_pred']['peak_x']), 0, W-1))
                            peak_gy = int(np.clip(np.round(rs['gpr_pred']['peak_y']), 0, H-1))
                            actual_intensity = float(sim_observed_counts[peak_gy, peak_gx]) if sim_observed_counts is not None else 0.0
                            completed_gpr_peaks.append((
                                rs['gpr_pred']['peak_x'],
                                rs['gpr_pred']['peak_y'],
                                rs['id'],
                                obs_id,
                                is_ghost,
                                estimated_intensity,  # GPR推定強度
                                actual_intensity      # 実際の強度
                            ))
                            print(f"  Robot {rs['id']}: Completed GPR at Task{obs_id + 1} (TRUE SOURCE), "
                                  f"peak at ({rs['gpr_pred']['peak_x']:.2f}, {rs['gpr_pred']['peak_y']:.2f}), "
                                  f"estimated={estimated_intensity:.2e}, actual={actual_intensity:.2e}")

                        # タスク完了を記録
                        task_assignment_log[rs['id']].append(obs_id)
                        rs['current_task_id'] = None
                        rs['current_goal'] = None
                        rs['state'] = 'IDLE'
                        rs['gpr_pred'] = None
                        rs['gpr_targets'] = []
                        rs['gpr_target_idx'] = 0
                    else:
                        if rs['gpr_pred']:
                            new_cx = int(np.round(rs['gpr_pred']['peak_x']))
                            new_cy = int(np.round(rs['gpr_pred']['peak_y']))
                            if (new_cx, new_cy) != rs['gpr_center']:
                                if 0 <= new_cx < W and 0 <= new_cy < H and true_grid[new_cy, new_cx] == 0:
                                    rs['gpr_center'] = (new_cx, new_cy)
                                    rs['gpr_targets'] = []
                                    rs['gpr_target_idx'] = 0
                                    rs['gpr_measuring'] = False

        # フレーム生成
        if step % 1 == 0 or not any_active:
            status_text = f"Step {step}"
            overall_frames.append(make_overall_frame(status_text))

        if not any_active:
            print(f"All robots completed at step {step}")
            break

    # 最終フレーム
    for _ in range(5):
        overall_frames.append(make_overall_frame(f"Step {step} - Completed"))

    # 各ロボットの移動距離を集計
    robot_distances = {rs['id']: rs['total_distance'] for rs in robot_states}

    # タスク割り当て結果を表示（1始まりで表示）
    print("\n=== Dynamic task allocation result ===")
    for r in range(num_robots):
        task_names = [f"Task{t+1}" for t in task_assignment_log[r]]
        print(f"  Robot {r}: {len(task_assignment_log[r])} tasks => {task_names}")

    # 障害物内タスクを表示
    if obstacle_tasks:
        obstacle_names = [f"Task{t+1}" for t in sorted(obstacle_tasks)]
        print(f"\n=== Tasks on obstacles (discovered and removed) ===")
        print(f"  {len(obstacle_tasks)} tasks: {obstacle_names}")

    # 到達不能タスクを表示
    if unreachable_tasks:
        unreachable_names = [f"Task{t+1}" for t in sorted(unreachable_tasks)]
        print(f"\n=== Unreachable tasks (removed) ===")
        print(f"  {len(unreachable_tasks)} tasks: {unreachable_names}")

    # ゴースト統計を表示
    # タプル構造: TRUE SOURCE=5要素, GHOST=7要素（両方とも5番目がis_ghost）
    num_ghosts = sum(1 for peak_data in completed_gpr_peaks if peak_data[4])
    num_true_sources = len(completed_gpr_peaks) - num_ghosts
    print(f"\n=== Ghost detection result ===")
    print(f"  True sources detected: {num_true_sources}")
    print(f"  Ghosts detected: {num_ghosts}")

    return overall_frames, completed_gpr_peaks, robot_distances, task_assignment_log
