import pygame as pg
from .OrderManager import OrderManager
import numpy as np
from pathlib import Path
from typing import Optional

ORDERS_DIR = Path(__file__).resolve().parents[1] / "orders"
ORDERS_DIR.mkdir(exist_ok=True)

class Subdivide:
    def __init__(self, track_name: str, spacing=30.0, lateral_divs=10):
        self.manager = OrderManager(track_name)
        self.manager.run()
        self.track_order = self.manager.get_order()
        self.track_name = track_name
        self.inner, self.outer = self.track_order
        

        self.triangles = []
        #self.spacing = 2 #spacing between points
        self.spacing = spacing
        self.lateral_divs = lateral_divs
        self.mesh = None

    def resample(self, path):
        path = np.array(path, dtype=float)
        x, y = path[:, 0], path[:, 1]

        # Compute distances between consecutive points
        distances = np.hypot(np.diff(x), np.diff(y))
        cumulative = np.concatenate(([0], np.cumsum(distances)))

        total_length = cumulative[-1] #last element of cumulative
        self.num_points = int(total_length / self.spacing)
        target_distances = np.linspace(0, total_length, self.num_points)

        # Interpolate x and y at new positions
        new_x = np.interp(target_distances, cumulative, x)
        new_y = np.interp(target_distances, cumulative, y)

        return np.stack((new_x, new_y), axis=1)
    def sync_resample(self):
        ################
        ## old resample method causes slipage between inner and outer boundaries as they turn at differet rates
        ## this method normalizes both of them to get rid of slipage
        ## nope this still doesnt work - can add to NEA :)
        ################

        def normalized_cumulative(pts):
            distances = np.hypot(np.diff(pts[:,0]), np.diff(pts[:,1]))
            sum = np.concatenate([[0.0], np.cumsum(distances)])   # length == len(pts)
            if sum[-1] == 0:
                return sum, sum #when end and start are same point
            return sum, sum / sum[-1] #sum and normalized sum

        inner = np.array(self.inner, dtype=float)
        outer = np.array(self.outer, dtype=float)
        

        s_in, s_in_norm = normalized_cumulative(inner)
        s_out, s_out_norm = normalized_cumulative(outer)
        n_samples = int(min(s_in[-1], s_out[-1]) / self.spacing)
        s_ref = np.linspace(0.0, 1.0, n_samples)

        inner_sync = np.column_stack([
            np.interp(s_ref, s_in_norm, inner[:,0]),
            np.interp(s_ref, s_in_norm, inner[:,1])
        ])
        outer_sync = np.column_stack([
            np.interp(s_ref, s_out_norm, outer[:,0]), #x
            np.interp(s_ref, s_out_norm, outer[:,1])  #y
        ])
        self.inner_sample = inner_sync
        self.outer_sample = outer_sync
        return inner_sync, outer_sync
    

    
    def constructTriangles(self):
        self.triangles = []
        num_points = min(len(self.inner_sample), len(self.outer_sample))
        for i in range(num_points -1):
            l1 = (self.inner_sample[i][0], self.inner_sample[i][1])
            l2 = (self.inner_sample[i + 1][0], self.inner_sample[i + 1][1])
            r1 = (self.outer_sample[i][0], self.outer_sample[i][1])
            r2 = (self.outer_sample[i + 1][0], self.outer_sample[i + 1][1])

            # Two triangles per quad
            self.triangles.append([l1, r1, r2])
            self.triangles.append([l1, r2, l2])
        self.triangles = np.array(self.triangles, dtype=np.int32)
        return self.triangles

    def contstructMesh(self):
        """
        Build a structured grid between inner and outer boundaries.
        spacing: distance (pixels/meters) between cross-sections along the track
        lateral_divs: number of evenly spaced points between inner and outer per section
        """
        
        n_sections = min(len(self.inner_sample), len(self.outer_sample))

        mesh = np.zeros((n_sections, self.lateral_divs, 2), dtype=float)

        for i in range(n_sections):
            inner_pt = self.inner_sample[i]
            outer_pt = self.outer_sample[i]
            for j, t in enumerate(np.linspace(0, 1, self.lateral_divs)):
                # Linear interpolation between inner and outer
                mesh[i, j] = inner_pt * (1 - t) + outer_pt * t

        self.mesh = mesh
        return mesh

    
    def drawMeshgrid(self, surface):
        for i in range(self.mesh.shape[0]):
            for j in range(self.mesh.shape[1]):
                x, y = self.mesh[i, j][0], self.mesh[i, j][1]
                pg.draw.circle(surface, (0, 255, 0), (int(x), int(y)), 2)

            # Draw lateral lines (cross-sections)
            for j in range(self.mesh.shape[1] - 1):
                p1 = (self.mesh[i, j][0], self.mesh[i, j][1])
                p2 = (self.mesh[i, j + 1][0], self.mesh[i, j + 1][1])
                #pg.draw.line(surface, (100, 100, 255), p1, p2, 1)

            # Draw longitudinal lines (grid columns)
            if i < self.mesh.shape[0] - 1:
                for j in range(self.mesh.shape[1]):
                    p1 = (self.mesh[i, j][0], self.mesh[i, j][1])
                    p2 = (self.mesh[i + 1, j][0], self.mesh[i + 1, j][1])
                    #pg.draw.line(surface, (255, 100, 100), p1, p2, 1)

    def drawTriangles(self, surface):
        self.get_triangles()
        for i in range(1, len(self.triangles) - 1):
            pg.draw.polygon(surface, (255,0,0), self.triangles[i], 2)

    def get_cross_section(self, i):
        """Return all lateral points at a given longitudinal section index."""
        return self.mesh[i]

    def get_mesh(self):
        """Return the full NxMx2 mesh array."""
        if self.mesh is None:
            self.contstructMesh()
        return self.mesh

    def get_triangles(self):
        return self.constructTriangles()
    
    def get_resampled(self):
        #self.inner_sample = self.resample(self.inner)
        #self.outer_sample = self.resample(self.outer)
        self.inner_sample, self.outer_sample = self.sync_resample()
        return self.inner_sample, self.outer_sample
        
    # ==============================================================
    # === Resampled SAVE/LOAD
    # ==============================================================

    @staticmethod
    def save_resampled(resampled: dict[str, np.ndarray],
                       track_name: str,
                       spacing: float) -> None:
        """
        Save resampled track boundaries (inner/outer) as a compressed NPZ file.
        """
        filename = ORDERS_DIR / f"{track_name}_resampled_{int(spacing)}.npz"
        np.savez(filename, inner=resampled["inner"], outer=resampled["outer"])
        print(f"Resampled boundaries saved to {filename}")

    @staticmethod
    def load_resampled(track_name: str,
                       spacing: float) -> Optional[dict[str, np.ndarray]]:
        """Load resampled track boundaries for given spacing."""
        filename = ORDERS_DIR / f"{track_name}_resampled_{int(spacing)}.npz"
        try:
            data = np.load(filename, allow_pickle=True)
            return {"inner": data["inner"], "outer": data["outer"]}
        except FileNotFoundError:
            print(f"No resampled data found: {filename}")
            return None
        except Exception as e:
            print(f"Error loading resampled data: {e}")
            return None

    # ==============================================================
    # === Mesh SAVE/LOAD
    # ==============================================================
        
    @staticmethod
    def save_mesh(mesh: np.ndarray,
                  track_name: str,
                  spacing: float,
                  lateral_divs: int) -> None:
        filename = ORDERS_DIR / f"{track_name}_mesh_{int(spacing)}x{lateral_divs}.npy"
        np.save(filename, mesh, allow_pickle=True)
        print(f"Mesh saved to {filename}")

    @staticmethod
    def load_mesh(track_name: str,
                  spacing: float,
                  lateral_divs: int) -> Optional[np.ndarray]:
        filename = ORDERS_DIR / f"{track_name}_mesh_{int(spacing)}x{lateral_divs}.npy"
        try:
            return np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            print(f"No mesh found: {filename}")
            return None

    # ==============================================================
    # === TRIANGLES SAVE/LOAD
    # ==============================================================
    @staticmethod
    def save_triangles(triangles: np.ndarray,
                       track_name: str,
                       spacing: float) -> None:
        filename = ORDERS_DIR / f"{track_name}_triangles_{int(spacing)}.npy"
        np.save(filename, triangles, allow_pickle=True)
        print(f"Triangles saved to {filename}")

    @staticmethod
    def load_triangles(track_name: str,
                       spacing: float) -> Optional[np.ndarray]:
        filename = ORDERS_DIR / f"{track_name}_triangles_{int(spacing)}.npy"
        try:
            return np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            print(f"No triangles found: {filename}")
            return None


    def main(self):
        self.resampled = self.load_resampled(self.track_name, self.spacing)
        #self.resampled = None
        if self.resampled is None:
            print("No resampled data found, computing resampled boundaries.")
            self.get_resampled()
            self.save_resampled({'inner': self.inner_sample, 'outer': self.outer_sample},
                           self.track_name, self.spacing)
        else:
            self.inner_sample = self.resampled['inner']
            self.outer_sample = self.resampled['outer']
        

        self.triangles = self.load_triangles(self.track_name, self.spacing)
        if self.triangles is None:
            print("No triangles data found, computing triangles.")
            self.get_triangles()
            self.save_triangles(self.triangles, self.track_name, self.spacing)

            
        self.mesh = self.load_mesh(self.track_name, self.spacing, self.lateral_divs)
        if self.mesh is None:
            print("No mesh data found, computing mesh.")
            self.get_mesh()
            self.save_mesh(self.mesh, self.track_name, self.spacing, self.lateral_divs)
        
        print("resampled Order lengths:", [len(self.inner_sample), len(self.outer_sample)])
        #print("First 5 points of track order:", self.track_order[0][:5])

        print(f"Track '{self.track_name}' fully processed with spacing={self.spacing}, lateral_divs={self.lateral_divs}")



        



