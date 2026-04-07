"""
Core classes for robot dynamic identification.

Provides RobotIdentifier (model loading, base parameter discovery, regressor
computation, torque simulation, least-squares estimation) and BaseParams
(the mapping from 10n standard parameters to r identifiable base parameters).
"""

import numpy as np
import pinocchio as pin
from scipy.linalg import qr
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BaseParams:
    """
    Encodes the mapping from 10n standard parameters to r base parameters.

    Given the full regressor Y and any configuration (q,v,a):

        Y_b = Y[:, P[:r]] + Y[:, P[r:]] @ K.T       (r columns, full rank)

    Any pi_standard satisfies:

        Y * pi_standard = Y_b * pi_b

    where

        pi_b = pi_standard[P[:r]] + K @ pi_standard[P[r:]]

    Fields
    ------
    P    : column permutation index array  (length 10n)
    K    : regrouping matrix               (r x (10n - r))
    r    : number of base parameters  (= rank Y)
    n_std: total standard parameters  (= 10 * n_bodies)
    """
    P:     np.ndarray
    K:     np.ndarray
    r:     int
    n_std: int


class RobotIdentifier:
    """
    Plug-and-play robot dynamic identification.

    Parameters
    ----------
    urdf_path : str | Path
    verbose   : bool
    """

    _PARAM_NAMES = ["m",
                    "mc_x", "mc_y", "mc_z",
                    "Sigma_xx", "Sigma_xy", "Sigma_xz",
                    "Sigma_yy", "Sigma_yz", "Sigma_zz"]

    def __init__(self, urdf_path: str | Path, verbose: bool = True):
        self.urdf_path = str(urdf_path)
        suffix = Path(self.urdf_path).suffix.lower()
        if suffix == '.xml':
            self.model = pin.buildModelFromMJCF(self.urdf_path)
        else:
            self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        for i in range(self.model.njoints):
            idx_q = self.model.joints[i].idx_q
            nq_j  = self.model.joints[i].nq
            for k in range(nq_j):
                if not np.isfinite(self.model.lowerPositionLimit[idx_q + k]):
                    self.model.lowerPositionLimit[idx_q + k] = -np.pi
                if not np.isfinite(self.model.upperPositionLimit[idx_q + k]):
                    self.model.upperPositionLimit[idx_q + k] = np.pi

        self.nq       = self.model.nq
        self.nv       = self.model.nv
        self.n_bodies = self.model.nbodies - 1
        self.n_std    = 10 * self.n_bodies

        if verbose:
            print(f"\nRobot  : {self.model.name}")
            print(f"  nv              : {self.nv}  (DOF)")
            print(f"  bodies          : {self.n_bodies}")
            print(f"  standard params : {self.n_std}  ({self.n_bodies} x 10)")

    def find_base_params(
        self,
        n_samples: int   = 1000,
        tol:       float = None,
    ) -> BaseParams:
        """
        Identify the base (identifiable) parameter structure via rank-revealing QR.
        """
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(n_samples):
            q   = pin.randomConfiguration(self.model)
            dq  = rng.standard_normal(self.nv)
            ddq = rng.standard_normal(self.nv)
            Y_i = pin.computeJointTorqueRegressor(
                self.model, self.data, q, dq, ddq
            )
            rows.append(Y_i)

        Y_rand = np.vstack(rows)

        _, R, P = qr(Y_rand, pivoting=True)

        diag_R = np.abs(np.diag(R))
        if tol is None:
            tol = diag_R[0] * max(Y_rand.shape) * np.finfo(float).eps * 100

        r = int(np.sum(diag_R > tol))

        R11 = R[:r, :r]
        R12 = R[:r, r:]
        K   = np.linalg.solve(R11, R12)

        bp = BaseParams(P=P, K=K, r=r, n_std=self.n_std)

        print(f"  standard params : {self.n_std}")
        print(f"  base params     : {r}  (identifiable)")
        print(f"  singular values (log10): {np.log10(diag_R[:r+2].clip(1e-15))}")
        self._print_base_structure(bp)

        return bp

    def fourier_trajectory(
        self,
        T:           float = 30.0,
        dt:          float = 1e-2,
        omega_f:     float = 0.5,
        n_harmonics: int   = 2,
        amplitude:   float = 1,
        q_center            = None,
        seed:        int   = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Band-limited Fourier-series excitation trajectory with analytic derivatives.
        """
        rng = np.random.default_rng(seed)
        n   = self.nv

        a = rng.uniform(-amplitude, amplitude, (n, n_harmonics))
        b = rng.uniform(-amplitude, amplitude, (n, n_harmonics))

        if q_center is None:
            q_center = np.zeros(n)
        else:
            q_center = np.asarray(q_center, float)[:n]

        t_arr    = np.arange(0.0, T, dt)
        N        = len(t_arr)

        q_traj   = np.tile(q_center, (N, 1)).copy()
        dq_traj  = np.zeros((N, n))
        ddq_traj = np.zeros((N, n))

        for j in range(n):
            for k in range(1, n_harmonics + 1):
                kw  = k * omega_f
                sk  = np.sin(kw * t_arr)
                ck  = np.cos(kw * t_arr)
                q_traj[:, j]   +=  a[j, k-1] / kw * sk - b[j, k-1] / kw * ck
                dq_traj[:, j]  +=  a[j, k-1]      * ck + b[j, k-1]      * sk
                ddq_traj[:, j] += -a[j, k-1] * kw  * sk + b[j, k-1] * kw  * ck

        return t_arr, q_traj, dq_traj, ddq_traj

    def compute_base_regressor(
        self,
        q_traj:   np.ndarray,
        dq_traj:  np.ndarray,
        ddq_traj: np.ndarray,
        bp:       BaseParams,
    ) -> np.ndarray:
        """
        Build stacked base regressor  Y_b in R^{N*nv x r}.
        """
        N = len(q_traj)
        Y_b = np.zeros((N * self.nv, bp.r))

        for i in range(N):
            Y_full_i = pin.computeJointTorqueRegressor(
                self.model, self.data,
                q_traj[i], dq_traj[i], ddq_traj[i],
            )
            Y_b[i * self.nv : (i + 1) * self.nv, :] = Y_full_i[:, bp.P[:bp.r]]

        return Y_b

    def simulate_torques(
        self,
        q_traj:   np.ndarray,
        dq_traj:  np.ndarray,
        ddq_traj: np.ndarray,
        noise_std: float = 0.0,
    ) -> np.ndarray:
        """
        Compute inverse-dynamics torques via RNEA.
        Returns tau in R^{N*nv}.
        """
        N   = len(q_traj)
        tau = np.zeros(N * self.nv)

        for i in range(N):
            tau_i = pin.rnea(
                self.model, self.data,
                q_traj[i], dq_traj[i], ddq_traj[i],
            )
            tau[i * self.nv : (i + 1) * self.nv] = tau_i

        if noise_std > 0.0:
            tau += np.random.default_rng(1).normal(0.0, noise_std, tau.shape)

        return tau

    def estimate(
        self,
        Y_b:  np.ndarray,
        tau:  np.ndarray,
    ) -> tuple[np.ndarray, float, int]:
        """
        pi_b_est = argmin ||Y_b * pi_b - tau||^2  (ordinary least squares)

        Returns (pi_b_est, condition_number, rank)
        """
        cond = np.linalg.cond(Y_b)
        pi_b_est, _, rank, _ = np.linalg.lstsq(Y_b, tau, rcond=None)
        return pi_b_est, cond, rank

    def true_standard_params(self) -> np.ndarray:
        """
        Extract the full 10n standard parameter vector from the Pinocchio model.
        """
        pi = []
        for i in range(1, self.model.nbodies):
            inertia = self.model.inertias[i]
            m   = inertia.mass
            c   = inertia.lever
            I_c = inertia.inertia

            mc    = m * c
            Sigma = I_c + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))

            pi.extend([
                m,
                mc[0], mc[1], mc[2],
                Sigma[0, 0], Sigma[0, 1], Sigma[0, 2],
                Sigma[1, 1], Sigma[1, 2], Sigma[2, 2],
            ])
        return np.array(pi)

    def true_base_params(self, bp: BaseParams) -> np.ndarray:
        """
        Project the true standard parameter vector onto the base parameter space.

        pi_b_true = pi_std[P[:r]]  +  K @ pi_std[P[r:]]
        """
        pi_std = self.true_standard_params()
        pi_p   = pi_std[bp.P]
        return pi_p[:bp.r] + bp.K @ pi_p[bp.r:]

    def _param_label(self, standard_idx: int) -> str:
        body  = standard_idx // 10 + 1
        param = self._PARAM_NAMES[standard_idx % 10]
        return f"b{body}:{param}"

    def _print_base_structure(self, bp: BaseParams):
        print(f"\n  Base parameter breakdown  (r = {bp.r}):")
        print(f"  {'pi_b idx':>7}  {'expression (top contributions)':}")
        print("  " + "-" * 60)
        for bi in range(bp.r):
            primary_std_idx = bp.P[bi]
            label = f"{self._param_label(primary_std_idx)}"
            absorbed = []
            for di in range(bp.n_std - bp.r):
                coeff = bp.K[bi, di]
                if abs(coeff) > 1e-6:
                    dep_std_idx = bp.P[bp.r + di]
                    absorbed.append(f"{coeff:+.4f}*{self._param_label(dep_std_idx)}")
            if absorbed:
                label += "  +  " + "  +  ".join(absorbed[:3])
                if len(absorbed) > 3:
                    label += f"  (+{len(absorbed)-3} more)"
            print(f"  {bi:>7}  {label}")
