import os
import numpy as np
import torch
import sys

from MathHelpers.none_check import all_set

#FIX HARDCODED PATH

NN_dir = os.path.abspath('/home/dstan/PolySpace/MLOrbits/AIAA_Student_Conference')
sys.path.append(NN_dir)
from LambertAIAA import Lambert
from lambertNN_dataset_aiaa import gooding_T, gooding_q
from lambertNN_net_aiaa import ResidualMLP, CKPT_FILE, gooding_update

# orekit
import orekit
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir(from_pip_library=True)
from org.hipparchus.geometry.euclidean.threed import Vector3D


class NNLambert:
    def __init__(self, MU=1.0, ckpt_file=CKPT_FILE, device="cpu"):
        '''
        Lambert wrapper: root solver + NN residual fast path.

        Canonical conventions:
            - MU is canonical mu (usually 1.0)
            - r is in DU
            - tof is in TU
            - v is in DU/TU

        Inputs:
            MU : float
                canonical mu (default 1.0)
            ckpt_file : str
                torch checkpoint path for residual NN
            device    : str
        '''
        if not all_set(MU, ckpt_file):
            raise TypeError("define MU and ckpt_file")

        self.MU = float(MU)
        self.device = str(device)
        self.ckpt_file = str(ckpt_file)

        self.lambert = Lambert(self.MU)

        torch.set_default_dtype(torch.float64)
        self._load_nn()

    # ------------------------------------------------------------------
    # NN loading + inference
    # ------------------------------------------------------------------
    def _load_nn(self):
        '''
        Load checkpoint and build residual network.

        Expects checkpoint keys:
            - model_state_dict
            - input_mean (len 2)
            - input_std  (len 2)
            - output_mean (scalar)
            - output_std  (scalar)
        '''
        if not os.path.exists(self.ckpt_file):
            raise FileNotFoundError(f"NN checkpoint not found: {self.ckpt_file}")

        ckpt = torch.load(self.ckpt_file, map_location=self.device)

        self.model = ResidualMLP(in_dim=2).double()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.in_mean = torch.tensor(ckpt["input_mean"], dtype=torch.float64, device=self.device)
        self.in_std  = torch.tensor(ckpt["input_std"],  dtype=torch.float64, device=self.device)
        self.out_mean = float(ckpt["output_mean"])
        self.out_std  = float(ckpt["output_std"])

    def x_resid_from_T_Lambda(self, T, Lambda):
        '''
        NN predicts x_resid given (T, Lambda).

        Inputs:
            T      : float
                Gooding nondimensional time variable (canonical definition)
            Lambda : float
                Battin lambda parameter (can be signed)

        Output:
            x_resid : float
                residual in x-domain used by PolySpace/Battin formulation
        '''
        x_in_raw = torch.tensor([float(T), float(Lambda)], dtype=torch.float64, device=self.device)
        x_in = (x_in_raw - self.in_mean) / self.in_std

        with torch.no_grad():
            resid_norm = float(self.model(x_in.unsqueeze(0)).item())

        return resid_norm * self.out_std + self.out_mean

    # ------------------------------------------------------------------
    # root solver path
    # ------------------------------------------------------------------
    def solve_traditional_lambert_canonical(self, r1_du, r2_du, tof_tu, nrev=0, verbose=False):
        '''
        Root-find Lambert using LambertAIAA (canonical units).

        Inputs:
            r1_du   : (3,) array
            r2_du   : (3,) array
            tof_tu  : float (TU)   (sign selects branch in LambertAIAA)
            nrev    : int
            verbose : bool

        Outputs:
            v1_duptu : (3,) ndarray
            v2_duptu : (3,) ndarray
            exitflag : int (1 success)
        '''
        from org.hipparchus.geometry.euclidean.threed import Vector3D

        r1_du = np.asarray(r1_du, dtype=float).reshape(3)
        r2_du = np.asarray(r2_du, dtype=float).reshape(3)

        pos1 = Vector3D(float(r1_du[0]), float(r1_du[1]), float(r1_du[2]))
        pos2 = Vector3D(float(r2_du[0]), float(r2_du[1]), float(r2_du[2]))

        V1, V2, exitflag = self.lambert.solve(
            pos1, pos2, float(tof_tu),
            nrev=int(nrev),
            verbose=bool(verbose)
        )

        # normalize return type to numpy arrays
        if hasattr(V1, "getX"):
            v1 = np.array([V1.getX(), V1.getY(), V1.getZ()], dtype=float)
            v2 = np.array([V2.getX(), V2.getY(), V2.getZ()], dtype=float)
        else:
            v1 = np.asarray(V1, dtype=float).reshape(3)
            v2 = np.asarray(V2, dtype=float).reshape(3)

        return v1, v2, int(exitflag)

    # NN solver path 
    def solve_nn_lambert_canonical(self, r1_du, r2_du, tof_tu, GOODING=True, nrev=0):
        '''
        NN-accelerated Lambert in canonical units (0-rev only).

        This uses the same formulation you were using in your cassini harness:
            - compute an analytic first guess x_first
            - compute Gooding T and Battin Lambda
            - NN predicts x_resid(T, Lambda)
            - x = x_first + x_resid
            - reconstruct v1, v2

        Inputs:
            r1_du  : (3,) array, DU
            r2_du  : (3,) array, DU
            tof_tu : float, TU (sign controls long/short branch like root solver)
            nrev   : int (must be 0 here)

        Outputs:
            v1_duptu : (3,) ndarray
            v2_duptu : (3,) ndarray
            exitflag : int (1 success, 0 fail)
        '''
        if int(nrev) != 0:
            raise ValueError("NNLambert.solve_nn_lambert_canonical supports nrev=0 only")

        r1_du = np.asarray(r1_du, dtype=float).reshape(3)
        r2_du = np.asarray(r2_du, dtype=float).reshape(3)

        tof_tu = float(tof_tu)
        if tof_tu == 0.0:
            raise ValueError("tof_tu must be nonzero")

        # sign convention: match Lambert.py usage
        longway = float(np.sign(tof_tu))   # +1 short, -1 long
        tof_abs = abs(tof_tu)

        # normalized geom
        mu = float(self.MU)

        r1 = float(np.linalg.norm(r1_du))
        r2 = float(np.linalg.norm(r2_du))
        if r1 <= 0.0 or r2 <= 0.0:
            return np.zeros(3), np.zeros(3), 0

        r1vec = r1_du / r1
        r2vec = r2_du / r1  # scaled by r1 (so |r1| = 1 internally)

        V = np.sqrt(mu / r1)    # DU/TU
        Tscale = r1 / V         # TU
        tf = tof_abs / Tscale   # dimensionless time in PolySpace formulation
        if tf <= 0.0:
            return np.zeros(3), np.zeros(3), 0

        mr2vec = float(np.linalg.norm(r2vec))

        # transfer angle from geometry (short angle first)
        cos_dth = float(np.dot(r1vec, r2vec) / (mr2vec + 1e-30))
        cos_dth = max(-1.0, min(1.0, cos_dth))
        dth = float(np.arccos(cos_dth))

        # long-way flips angle
        if longway < 0.0:
            dth = float(2.0 * np.pi - dth)

        # chord + semiperimeter in scaled-by-r1 space
        c = float(np.sqrt(1.0 + mr2vec**2 - 2.0 * mr2vec * np.cos(dth)))
        s = float((1.0 + mr2vec + c) / 2.0)

        aMin = float(s / 2.0)

        # Battin lambda (signed naturally via cos(dth/2))
        Lambda = float(np.sqrt(mr2vec) * np.cos(dth / 2.0) / s)

        # normal vector (defines plane)
        crossprd = np.cross(r1vec, r2vec)
        mcr = float(np.linalg.norm(crossprd))
        if mcr == 0.0:
            return np.zeros(3), np.zeros(3), 0
        nrmunit = crossprd / mcr
        ih = longway * nrmunit

        # analytic x_first 
        logt = float(np.log(tf))

        inn1 = -0.5233
        inn2 = +0.5233
        x1 = float(np.log(1.0 + inn1))
        x2 = float(np.log(1.0 + inn2))

        xx = np.array([inn1, inn2], dtype=float)
        aa = aMin / (1.0 - np.square(xx))

        # bbeta carries longway sign
        arg_beta = (s - c) / (2.0 * aa)
        arg_beta = np.maximum(0.0, np.minimum(1.0, arg_beta))
        bbeta = longway * 2.0 * np.arcsin(np.sqrt(arg_beta))

        aalfa = 2.0 * np.arccos(np.maximum(-1.0, np.minimum(1.0, xx)))

        # y12 is time-of-flight expression (m=0 here)
        y12 = aa * np.sqrt(aa) * ((aalfa - np.sin(aalfa)) - (bbeta - np.sin(bbeta)))

        # guard
        if np.any(y12 <= 0.0):
            return np.zeros(3), np.zeros(3), 0

        y1 = float(np.log(y12[0]) - logt)
        y2 = float(np.log(y12[1]) - logt)

        denom = (y2 - y1)
        if denom == 0.0:
            return np.zeros(3), np.zeros(3), 0

        x1log = float((x1 * y2 - y1 * x2) / denom)
        x_first = float(np.exp(x1log) - 1.0)

        # Gooding T (canonical) uses *canonical* semiperimeter in DU
        s_canon = float(s * r1)         # DU
        T_gooding = float(gooding_T(s_canon, tof_tu, mu=mu))
        q_gooding = float(gooding_q(r1, r2, s_canon, cos_dth))

        # NN residual -> x
        x_resid = float(self.x_resid_from_T_Lambda(T_gooding, Lambda))
        x = float(x_first + x_resid)

        if GOODING:
            x = gooding_update(x, q_gooding, T_gooding)

        # -reconstruct velos
        a = float(aMin / (1.0 - x**2))

        if x < 1.0:
            arg = (s - c) / (2.0 * a)
            arg = max(0.0, min(1.0, float(arg)))
            beta = longway * 2.0 * np.arcsin(np.sqrt(arg))
            alfa = 2.0 * np.arccos(max(-1.0, min(1.0, x)))
            psi = float((alfa - beta) / 2.0)
            eta2 = float(2.0 * a * (np.sin(psi) ** 2) / s)
            if eta2 <= 0.0:
                return np.zeros(3), np.zeros(3), 0
            eta = float(np.sqrt(eta2))
        else:
            arg = (c - s) / (2.0 * a)
            arg = max(0.0, float(arg))
            beta = longway * 2.0 * np.asinh(np.sqrt(arg))
            alfa = 2.0 * np.arccosh(x)
            psi = float((alfa - beta) / 2.0)
            eta2 = float(-2.0 * a * (np.sinh(psi) ** 2) / s)
            if eta2 <= 0.0:
                return np.zeros(3), np.zeros(3), 0
            eta = float(np.sqrt(eta2))

        r2n = r2vec / mr2vec

        crsprd1 = np.cross(ih, r1vec)
        crsprd2 = np.cross(ih, r2n)

        Vr1 = (1.0 / eta / np.sqrt(aMin)) * (2.0 * Lambda * aMin - Lambda - x * eta)

        denom_vt = float(aMin * eta2)
        if denom_vt <= 0.0:
            return np.zeros(3), np.zeros(3), 0

        Vt1 = float(np.sqrt(mr2vec / denom_vt * (np.sin(dth / 2.0) ** 2)))

        Vt2 = float(Vt1 / mr2vec)

        tan_half = float(np.tan(dth / 2.0))
        if tan_half == 0.0:
            return np.zeros(3), np.zeros(3), 0

        Vr2 = float((Vt1 - Vt2) / tan_half - Vr1)

        v1_duptu = (Vr1 * r1vec + Vt1 * crsprd1) * V
        v2_duptu = (Vr2 * r2n   + Vt2 * crsprd2) * V

        return np.asarray(v1_duptu, dtype=float).reshape(3), np.asarray(v2_duptu, dtype=float).reshape(3), 1
