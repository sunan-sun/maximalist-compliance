"""
Sub-regressor helpers and ground-truth dynamic term computation.

Sub-regressors isolate individual dynamic terms (M, G, C*dq, p) by zeroing
specific inputs to the joint torque regressor. Ground-truth functions use
Pinocchio's dedicated algorithms for comparison.
"""

import numpy as np
import pinocchio as pin

from maxcomp.core import BaseParams


def _base_cols(Y_full: np.ndarray, bp: BaseParams) -> np.ndarray:
    """Select base columns from full regressor:  Y_b = Y_full[:, P[:r]]"""
    return Y_full[:, bp.P[:bp.r]]


def sub_regressor_inertia(
    model, data, q: np.ndarray, ddq: np.ndarray, bp: BaseParams
) -> np.ndarray:
    """
    Y_M_b(q, 0, ddq)  with gravity zeroed.
    Y_M_b * pi_b  =  M(q)*ddq
    """
    g_saved = model.gravity.linear.copy()
    model.gravity.linear[:] = 0.0

    Y_M = pin.computeJointTorqueRegressor(
        model, data, q, np.zeros(model.nv), ddq
    )
    Y_M_b = _base_cols(Y_M, bp)

    model.gravity.linear[:] = g_saved
    return Y_M_b


def sub_regressor_gravity(
    model, data, q: np.ndarray, bp: BaseParams
) -> np.ndarray:
    """
    Y_G_b(q, 0, 0)  with gravity active.
    Y_G_b * pi_b  =  G(q)
    """
    Y_G = pin.computeJointTorqueRegressor(
        model, data, q, np.zeros(model.nv), np.zeros(model.nv)
    )
    return _base_cols(Y_G, bp)


def sub_regressor_mass_matrix(
    model, data, q: np.ndarray, bp: BaseParams, pi_b: np.ndarray
) -> np.ndarray:
    """
    Reconstruct M(q) from the inertia sub-regressor column by column.

    M(q)[:, j]  =  Y_M_b(q, 0, e_j)|_{g=0}  * pi_b

    Returns the full (nv, nv) mass matrix estimate.
    """
    nv = model.nv
    M_est = np.zeros((nv, nv))

    g_saved = model.gravity.linear.copy()
    model.gravity.linear[:] = 0.0

    for j in range(nv):
        e_j = np.zeros(nv)
        e_j[j] = 1.0
        Y_M = pin.computeJointTorqueRegressor(
            model, data, q, np.zeros(nv), e_j
        )
        Y_M_b = _base_cols(Y_M, bp)
        M_est[:, j] = Y_M_b @ pi_b

    model.gravity.linear[:] = g_saved
    return M_est


def sub_regressor_coriolis(
    model, data, q: np.ndarray, dq: np.ndarray, bp: BaseParams
) -> np.ndarray:
    """
    Y_C_b(q, dq, 0)  with gravity zeroed.
    Y_C_b * pi_b  =  C(q,dq)*dq

    Note: this gives C*dq, NOT C^T*dq.
    """
    g_saved = model.gravity.linear.copy()
    model.gravity.linear[:] = 0.0

    Y_C = pin.computeJointTorqueRegressor(
        model, data, q, dq, np.zeros(model.nv)
    )
    Y_C_b = _base_cols(Y_C, bp)

    model.gravity.linear[:] = g_saved
    return Y_C_b


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth via Pinocchio
# ─────────────────────────────────────────────────────────────────────────────

def ground_truth_mass_matrix(model, data, q):
    """M(q) via Composite Rigid Body Algorithm."""
    M = pin.crba(model, data, q)
    return np.triu(M) + np.triu(M, 1).T


def ground_truth_inertia_term(model, data, q, ddq):
    """M(q)*ddq via Composite Rigid Body Algorithm."""
    M = pin.crba(model, data, q)
    return M @ ddq


def ground_truth_gravity(model, data, q):
    """G(q) via Pinocchio's generalised gravity."""
    return pin.computeGeneralizedGravity(model, data, q)


def ground_truth_coriolis_term(model, data, q, dq):
    """C(q,dq)*dq  =  RNEA(q, dq, 0) - G(q)"""
    tau_noa = pin.rnea(model, data, q, dq, np.zeros(model.nv))
    G       = pin.computeGeneralizedGravity(model, data, q)
    return tau_noa - G


def ground_truth_coriolis_transpose_term(model, data, q, dq):
    """
    C^T(q,dq)*dq via pin.computeCoriolisMatrix.

    This is the term that appears in the momentum observer beta:
        beta = C^T*dq - G(q)
    """
    C = pin.computeCoriolisMatrix(model, data, q, dq)
    return C.T @ dq


def ground_truth_momentum(model, data, q, dq):
    """p = M(q)*dq"""
    M = pin.crba(model, data, q)
    return M @ dq
