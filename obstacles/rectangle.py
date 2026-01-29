# obstacles/rectangle.py
# 矩形障害物クラス
import csv


class RectangleObstacle:
    """矩形障害物を表すクラス"""

    def __init__(self, x0, x1, y0, y1, mu=1.0, color='gray', alpha=0.6):
        """
        Parameters
        ----------
        x0, x1 : float
            x方向の範囲
        y0, y1 : float
            y方向の範囲
        mu : float
            減衰係数
        color : str
            描画色
        alpha : float
            透明度
        """
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.mu = mu
        self.color = color
        self.alpha = alpha

    def draw(self, ax, zorder=5, edgecolor='none'):
        """
        matplotlib の Axes に障害物を描画する

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            描画対象の Axes
        zorder : int
            描画順序
        edgecolor : str
            境界線の色（'none'で境界線なし）
        """
        ax.fill(
            [self.x0, self.x1, self.x1, self.x0],
            [self.y0, self.y0, self.y1, self.y1],
            color=self.color, alpha=self.alpha, zorder=zorder,
            edgecolor=edgecolor
        )

    def get_intersections(self, p1, p2):
        """
        線分 p1-p2 と矩形の各辺との交点を求める

        Parameters
        ----------
        p1, p2 : tuple
            線分の端点

        Returns
        -------
        list or None
            交点のリスト（2点の場合のみ）
        """
        edges = [
            ((self.x0, self.y0), (self.x0, self.y1)),
            ((self.x1, self.y0), (self.x1, self.y1)),
            ((self.x0, self.y0), (self.x1, self.y0)),
            ((self.x0, self.y1), (self.x1, self.y1))
        ]

        def segment_intersection(p1, p2, q1, q2):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = q1
            x4, y4 = q2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
            if 0 <= t <= 1 and 0 <= u <= 1:
                return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            return None

        intersections = []
        for edge in edges:
            pt = segment_intersection(p1, p2, *edge)
            if pt and pt not in intersections:
                intersections.append(pt)
        return intersections if len(intersections) == 2 else None

    @staticmethod
    def load_obstacles(csv_path):
        """
        CSVファイルから障害物を読み込む

        Parameters
        ----------
        csv_path : str
            CSVファイルのパス

        Returns
        -------
        list
            RectangleObstacle のリスト
        """
        obstacles = []
        try:
            with open(csv_path, "r", newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    x0, x1 = float(row["x0"]), float(row["x1"])
                    y0, y1 = float(row["y0"]), float(row["y1"])
                    mu = float(row.get("mu", 1.0))
                    obstacles.append(RectangleObstacle(x0, x1, y0, y1, mu=mu))
            print(f"Loaded {len(obstacles)} obstacles from {csv_path}")
        except FileNotFoundError:
            print(f"CSV not found: {csv_path}")
        return obstacles
