import csv
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter

class RectangleObstacle:
    def __init__(self, x0, x1, y0, y1, mu=1.0, color='gray', alpha=1.0):
        self.x0, self.x1, self.y0, self.y1, self.mu = x0, x1, y0, y1, mu
        self.color, self.alpha = color, alpha

    def draw(self, ax, zorder=5):
        ax.fill(
            [self.x0, self.x1, self.x1, self.x0],
            [self.y0, self.y0, self.y1, self.y1],
            color=self.color, alpha=self.alpha, zorder=zorder
        )

# --- 任意に障害物を定義 ---
obstacles = [
    RectangleObstacle(5, 44, 5, 6, mu=1.0),
    RectangleObstacle(5, 6, 6, 44, mu=1.0),
    RectangleObstacle(43, 44, 6, 44, mu=1.0),
    RectangleObstacle(5, 44, 44, 45, mu=1.0),
    RectangleObstacle(11, 14, 6, 7, mu=1.0),
    RectangleObstacle(16, 19, 6, 7, mu=1.0),
    RectangleObstacle(11, 15, 10, 12, mu=1.0),
    RectangleObstacle(16, 19, 10, 12, mu=1.0),
    RectangleObstacle(21, 28, 10, 11, mu=1.0),
    RectangleObstacle(21, 24, 12, 13, mu=1.0),
    RectangleObstacle(25, 28, 12, 13, mu=1.0),
    RectangleObstacle(29, 30, 9, 12, mu=1.0),
    RectangleObstacle(15, 16, 23, 27, mu=1.0),
    RectangleObstacle(16, 17, 27, 29, mu=1.0),
    RectangleObstacle(17, 18, 29, 31, mu=1.0),
    RectangleObstacle(18, 19, 31, 32, mu=1.0),
    RectangleObstacle(19, 21, 32, 33, mu=1.0),
    RectangleObstacle(21, 23, 33, 34, mu=1.0),
    RectangleObstacle(23, 27, 34, 35, mu=1.0),
    RectangleObstacle(27, 29, 34, 33, mu=1.0),
    RectangleObstacle(29, 31, 32, 33, mu=1.0),
    RectangleObstacle(31, 32, 31, 32, mu=1.0),
    RectangleObstacle(32, 33, 29, 31, mu=1.0),
    RectangleObstacle(33, 34, 27, 29, mu=1.0),
    RectangleObstacle(34, 35, 23, 27, mu=1.0),
    RectangleObstacle(33, 34, 21, 23, mu=1.0),
    RectangleObstacle(32, 33, 19, 21, mu=1.0),
    RectangleObstacle(31, 32, 18, 19, mu=1.0),
    RectangleObstacle(29, 31, 17, 18, mu=1.0),
    RectangleObstacle(27, 29, 16, 17, mu=1.0),
    RectangleObstacle(23, 27, 15, 16, mu=1.0),
    RectangleObstacle(21, 23, 16, 17, mu=1.0),
    RectangleObstacle(19, 21, 17, 18, mu=1.0),
    RectangleObstacle(18, 19, 18, 19, mu=1.0),
    RectangleObstacle(17, 18, 19, 21, mu=1.0),
    RectangleObstacle(16, 17, 21, 23, mu=1.0),
    RectangleObstacle(11, 17, 19, 20, mu=1.0),
    RectangleObstacle(11, 12, 20, 24, mu=1.0),
    RectangleObstacle(9, 11, 23, 24, mu=1.0),
    RectangleObstacle(9, 10, 24, 28, mu=1.0),
    RectangleObstacle(31, 32, 16, 17, mu=1.0),
    RectangleObstacle(32, 41, 15, 16, mu=1.0),
    RectangleObstacle(40, 41, 16, 18, mu=1.0),
    RectangleObstacle(33, 43, 20, 21, mu=1.0),
    RectangleObstacle(33, 40, 29, 30, mu=1.0),
    RectangleObstacle(38, 43, 31, 32, mu=1.0),
    RectangleObstacle(29, 30, 33, 34, mu=1.0),
    RectangleObstacle(30, 31, 34, 35, mu=1.0),
    RectangleObstacle(31, 32, 35, 36, mu=1.0),
    RectangleObstacle(32, 33, 36, 37, mu=1.0),
    RectangleObstacle(33, 42, 37, 38, mu=1.0),
    RectangleObstacle(10, 14, 27, 28, mu=1.0),
    RectangleObstacle(16, 17, 31, 32, mu=1.0),
    RectangleObstacle(15, 16, 32, 33, mu=1.0),
    RectangleObstacle(14, 15, 33, 34, mu=1.0),
    RectangleObstacle(13, 14, 34, 35, mu=1.0),
    RectangleObstacle(12, 13, 35, 36, mu=1.0),
    RectangleObstacle(11, 12, 36, 37, mu=1.0),
    RectangleObstacle(18, 19, 33, 34, mu=1.0),
    RectangleObstacle(17, 18, 34, 35, mu=1.0),
    RectangleObstacle(16, 17, 35, 36, mu=1.0),
    RectangleObstacle(15, 16, 36, 37, mu=1.0),
    RectangleObstacle(14, 15, 37, 38, mu=1.0),
    RectangleObstacle(13, 14, 38, 39, mu=1.0),
    RectangleObstacle(21, 24, 42, 44, mu=1.0),
    RectangleObstacle(21, 27, 39, 40, mu=1.0),
    RectangleObstacle(21, 24, 37, 38, mu=1.0),
    RectangleObstacle(25, 27, 12, 13, mu=1.0),
    RectangleObstacle(29, 32, 39, 40, mu=1.0),
    RectangleObstacle(29, 31, 38, 39, mu=1.0),
    RectangleObstacle(6, 8, 23, 24, mu=1.0),
    ]

# --- 表示 ---
name = "Unit1_1F"
grid_size = 50
label_interval = 5   # 数字の表示間隔
grid_interval = 1    # グリッド線の間隔

fig, ax = plt.subplots(figsize=(6,6))
for rect in obstacles:
    rect.draw(ax, zorder=5)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)

# --- グリッド線は1間隔（マイナー目盛） ---
ax.xaxis.set_minor_locator(MultipleLocator(grid_interval))
ax.yaxis.set_minor_locator(MultipleLocator(grid_interval))
ax.grid(which="minor", color="lightgray", linewidth=0.5, zorder=0)

# --- 数字付きの目盛は5間隔（メジャー目盛） ---
ax.xaxis.set_major_locator(MultipleLocator(label_interval))
ax.yaxis.set_major_locator(MultipleLocator(label_interval))
ax.grid(which="major", color="lightgray", linewidth=1.0, zorder=0)

plt.title(f"{name}")
plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
plt.show()

# --- CSVに保存 ---
save_dir = "."  # 任意のディレクトリ
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, f"{name}.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x0", "x1", "y0", "y1", "mu"])
    for rect in obstacles:
        writer.writerow([rect.x0, rect.x1, rect.y0, rect.y1, rect.mu])