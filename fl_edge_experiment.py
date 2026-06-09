"""
fl_edge_experiment.py
=====================================================================
On-device + collaborative learning for a swarm of embodied (Physical AI)
endpoints.  We compare *what to communicate* between co-located devices:

  - Local        : on-device learning only, no collaboration (baseline)
  - Central       : oracle, all raw data pooled (upper bound, unrealistic)
  - FedAvg        : share MODEL WEIGHTS via a coordinator   (McMahan+ 2017)
  - Gossip        : share MODEL WEIGHTS peer-to-peer on a ring (Lian+ 2017)
  - FedDistill    : share PREDICTIONS (soft labels) on a public probe set
                    (Jeong+ 2018 ; Li & Wang 2019 / FedMD)

Each "device" is a robot operating in a distinct physical environment,
modelled by (i) a non-IID label distribution (Dirichlet skew) and
(ii) a device-specific sensor transform (gain / offset / noise) ==
domain shift.  Devices have few on-device labels.

We measure, per communication round:
  * mean LOCAL test accuracy  (personalized objective: each robot in its
    own environment)  <-- primary metric for Physical AI
  * GLOBAL test accuracy      (generalization to the canonical task)
  * cumulative COMMUNICATION cost (floats transmitted over the network)

Three extra studies:
  (A) heterogeneous model architectures across devices  -> FedAvg/Gossip
      become INAPPLICABLE; FedDistill still works.
  (B) message quantization for FedDistill (float32 / int8 / int4).
  (C) analytical scaling of communication vs model size.

Pure NumPy MLP implemented from scratch (no deep-learning framework) so
every byte of communication is accounted for explicitly.

Author: (student) | Coding assistant: Claude Opus 4.8 (Anthropic), via Claude Code.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CFG = dict(
    n_agents     = 10,     # number of embodied endpoints
    alpha        = 0.3,    # Dirichlet concentration (smaller = more non-IID)
    probe_size   = 120,    # |public probe set| for federated distillation
    n_classes    = 10,
    n_features   = 64,     # 8x8 digit "camera"
    hidden       = 32,     # hidden units (homogeneous experiments)
    rounds       = 50,     # communication rounds
    local_epochs = 1,      # local supervised epochs per round
    distill_epochs = 1,    # distillation epochs per round (FedDistill)
    warmup       = 1,      # rounds of local-only before distillation starts
    pretrain_epochs = 8,   # FedMD transfer-learning warm-up on the public set
    lr           = 0.20,
    distill_lr   = 0.10,   # smaller LR for the distillation step
    momentum     = 0.9,
    batch        = 32,
    seeds        = [0, 1, 2],
)

METHOD_COLORS = {
    "Local":      "#9e9e9e",
    "Central":    "#000000",
    "FedAvg":     "#1f77b4",
    "Gossip":     "#2ca02c",
    "FedDistill": "#d62728",
}

# --------------------------------------------------------------------------
# Numpy MLP (1 hidden layer) implemented from scratch
# --------------------------------------------------------------------------
def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_out)
        self._zero_velocity()

    def _zero_velocity(self):
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2

    def logits(self, X):
        return self.forward(X)[2]

    def proba(self, X, T=1.0):
        return softmax(self.logits(X), T)

    def predict(self, X):
        return np.argmax(self.logits(X), axis=1)

    def count_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def get_flat(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_flat(self, v):
        i = 0
        for attr in ("W1", "b1", "W2", "b2"):
            arr = getattr(self, attr)
            n = arr.size
            setattr(self, attr, v[i:i + n].reshape(arr.shape).copy())
            i += n
        self._zero_velocity()


def sgd_supervised(model, X, y, cfg, rng, epochs=1, lr=None):
    """In-place mini-batch SGD with momentum on softmax cross-entropy."""
    n = len(X)
    if n == 0:
        return
    lr = cfg["lr"] if lr is None else lr
    mom, batch = cfg["momentum"], cfg["batch"]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            Xb, yb, B = X[b], y[b], len(b)
            z1, a1, z2 = model.forward(Xb)
            p = softmax(z2)
            Y = np.zeros_like(p); Y[np.arange(B), yb] = 1.0
            dz2 = (p - Y) / B
            dW2 = a1.T @ dz2; db2 = dz2.sum(0)
            da1 = dz2 @ model.W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = Xb.T @ dz1; db1 = dz1.sum(0)
            model.vW1 = mom * model.vW1 - lr * dW1; model.W1 += model.vW1
            model.vb1 = mom * model.vb1 - lr * db1; model.b1 += model.vb1
            model.vW2 = mom * model.vW2 - lr * dW2; model.W2 += model.vW2
            model.vb2 = mom * model.vb2 - lr * db2; model.b2 += model.vb2


def sgd_distill(model, Xp, Q, cfg, rng, epochs=1, lr=None):
    """In-place SGD matching the model's softmax to consensus soft targets Q
    (knowledge distillation, temperature 1)."""
    n = len(Xp)
    lr = cfg["distill_lr"] if lr is None else lr
    mom, batch = cfg["momentum"], cfg["batch"]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            Xb, Qb, B = Xp[b], Q[b], len(b)
            z1, a1, z2 = model.forward(Xb)
            p = softmax(z2)
            dz2 = (p - Qb) / B
            dW2 = a1.T @ dz2; db2 = dz2.sum(0)
            da1 = dz2 @ model.W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = Xb.T @ dz1; db1 = dz1.sum(0)
            model.vW1 = mom * model.vW1 - lr * dW1; model.W1 += model.vW1
            model.vb1 = mom * model.vb1 - lr * db1; model.b1 += model.vb1
            model.vW2 = mom * model.vW2 - lr * dW2; model.W2 += model.vW2
            model.vb2 = mom * model.vb2 - lr * db2; model.b2 += model.vb2


def quantize_rows(P, bits, normalize=False):
    """Per-row min-max quantization to `bits` (no-op if bits>=32)."""
    if bits >= 32:
        return P
    levels = (1 << bits) - 1
    lo = P.min(axis=1, keepdims=True)
    hi = P.max(axis=1, keepdims=True)
    scale = np.where(hi > lo, (hi - lo) / levels, 1.0)
    q = np.round((P - lo) / scale)
    deq = lo + q * scale
    if normalize:                       # probability vectors
        deq = np.clip(deq, 0, None)
        deq = deq / deq.sum(axis=1, keepdims=True)
    return deq                          # logits: keep sign/scale


def pretrain_public(models, Xp, yp, cfg, rng):
    """FedMD transfer-learning warm-up: every device trains briefly on the
    small public labeled set so even tiny models gain all-class competence
    (shared one-time setup; not counted as collaboration traffic)."""
    for m in models:
        sgd_supervised(m, Xp, yp, cfg, rng, cfg["pretrain_epochs"])


def fd_round(models, agents, Xprobe, cfg, rng, do_distill=True, quant_bits=32):
    """One federated-distillation round.

    Consensus = softmax(mean of the devices' LOGITS on the public probe set);
    each device distills toward it, THEN does a local supervised step (so
    personalization is the last thing written).  Returns floats transmitted.
    """
    N = len(models)
    S, C = len(Xprobe), cfg["n_classes"]
    comm = 0.0
    if do_distill:
        logits = np.stack([m.logits(Xprobe) for m in models])     # (N,S,C)
        if quant_bits < 32:
            logits = np.stack([quantize_rows(logits[a], quant_bits) for a in range(N)])
        Q = softmax(logits.mean(axis=0))                          # (S,C) consensus
        for a in range(N):
            sgd_distill(models[a], Xprobe, Q, cfg, rng, cfg["distill_epochs"])
        comm = 2.0 * N * S * C        # upload logits + download consensus
    for a in range(N):                # local supervised step LAST
        sgd_supervised(models[a], agents[a]["Xtr"], agents[a]["ytr"],
                       cfg, rng, cfg["local_epochs"])
    return comm


# --------------------------------------------------------------------------
# Data: build a swarm of embodied agents with non-IID labels + sensor shift
# --------------------------------------------------------------------------
def make_swarm(cfg, seed):
    rng = np.random.default_rng(1000 + seed)
    digits = load_digits()
    X = digits.data / 16.0           # [0,1], shape (1797, 64)
    y = digits.target.astype(int)

    # canonical (untransformed) global test set + public probe come from a
    # held-out pool so no agent ever trains on them.
    X_pool, X_gtest, y_pool, y_gtest = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)

    perm = rng.permutation(len(X_pool))
    probe_idx = perm[:cfg["probe_size"]]
    rest_idx = perm[cfg["probe_size"]:]
    X_probe = X_pool[probe_idx].copy()          # public shared set (canonical sensor)
    y_probe = y_pool[probe_idx].copy()          # labels available (FedMD transfer step)
    Xr, yr = X_pool[rest_idx], y_pool[rest_idx]

    # ---- Dirichlet label partition across agents (label heterogeneity) ----
    N = cfg["n_agents"]
    agent_items = [[] for _ in range(N)]
    for c in range(cfg["n_classes"]):
        c_idx = np.where(yr == c)[0]
        rng.shuffle(c_idx)
        prop = rng.dirichlet(cfg["alpha"] * np.ones(N))
        cuts = (np.cumsum(prop)[:-1] * len(c_idx)).astype(int)
        for a, part in enumerate(np.split(c_idx, cuts)):
            agent_items[a].extend(part.tolist())

    # ---- per-agent sensor transform (domain shift) ----
    agents = []
    class_counts = np.zeros((N, cfg["n_classes"]), dtype=int)
    for a in range(N):
        items = np.array(agent_items[a], dtype=int)
        rng.shuffle(items)
        Xi, yi = Xr[items], yr[items]
        # sensor parameters for this physical environment (mild domain shift)
        gain = rng.uniform(0.8, 1.2)
        offset = rng.uniform(-0.05, 0.05)
        sigma = rng.uniform(0.0, 0.05)

        def sensor(M, _g=gain, _o=offset, _s=sigma, _rng=rng):
            out = _g * M + _o + _rng.normal(0, _s, size=M.shape)
            return np.clip(out, 0.0, 1.0)

        Xi_t = sensor(Xi)
        # local train/test split (few on-device labels)
        if len(yi) >= 6:
            Xtr, Xte, ytr, yte = train_test_split(
                Xi_t, yi, test_size=0.3, random_state=seed,
                stratify=yi if np.min(np.bincount(yi, minlength=10)[np.unique(yi)]) >= 2 else None)
        else:
            Xtr, Xte, ytr, yte = Xi_t, Xi_t, yi, yi
        for c in np.unique(yi):
            class_counts[a, c] = int(np.sum(yi == c))
        agents.append(dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                           gain=gain, offset=offset, sigma=sigma,
                           n=len(ytr)))

    swarm = dict(agents=agents, X_probe=X_probe, y_probe=y_probe,
                 X_gtest=X_gtest, y_gtest=y_gtest,
                 class_counts=class_counts,
                 X_all=np.vstack([ag["Xtr"] for ag in agents] + [X_probe]),
                 y_all=np.concatenate([ag["ytr"] for ag in agents] + [y_probe]))
    return swarm


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def mean_local_acc(models, agents):
    accs = []
    for a in range(len(agents)):
        if len(agents[a]["yte"]) == 0:
            continue
        accs.append(float((models[a].predict(agents[a]["Xte"]) == agents[a]["yte"]).mean()))
    return (float(np.mean(accs)) if accs else 0.0), accs


def mean_global_acc(models, Xg, yg):
    accs = [float((models[a].predict(Xg) == yg).mean()) for a in range(len(models))]
    return float(np.mean(accs))


def ring_mixing_matrix(N):
    """Metropolis weights on a ring (doubly stochastic)."""
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 0.5
        A[i, (i - 1) % N] = 0.25
        A[i, (i + 1) % N] = 0.25
    return A


# --------------------------------------------------------------------------
# One training run for a given method (homogeneous models)
# --------------------------------------------------------------------------
def run_method(method, swarm, cfg, seed, hidden=None):
    rng = np.random.default_rng(7000 + seed)
    agents = swarm["agents"]
    N = len(agents)
    H = hidden if hidden is not None else cfg["hidden"]
    nin, nout = cfg["n_features"], cfg["n_classes"]
    Xg, yg = swarm["X_gtest"], swarm["y_gtest"]
    Xprobe = swarm["X_probe"]
    S, C = len(Xprobe), cfg["n_classes"]

    # init one model per agent from the SAME seed (common starting point)
    models = [MLP(nin, H, nout, seed=123) for _ in range(N)]
    P = models[0].count_params()
    A = ring_mixing_matrix(N)
    if method != "Central":
        pretrain_public(models, swarm["X_probe"], swarm["y_probe"], cfg, rng)

    hist = dict(round=[], local=[], glob=[], comm=[])
    comm = 0.0  # floats transmitted over the network (cumulative)

    if method == "Central":
        central = MLP(nin, H, nout, seed=123)
        # one-time cost: every device uploads its raw labeled data
        comm = float(sum(ag["Xtr"].shape[0] * nin for ag in agents))

    for r in range(1, cfg["rounds"] + 1):
        if method == "Local":
            for a in range(N):
                sgd_supervised(models[a], agents[a]["Xtr"], agents[a]["ytr"],
                               cfg, rng, cfg["local_epochs"])

        elif method == "Central":
            sgd_supervised(central, swarm["X_all"], swarm["y_all"],
                           cfg, rng, cfg["local_epochs"])
            models = [central] * N  # broadcast (eval convenience)

        elif method == "FedAvg":
            for a in range(N):
                sgd_supervised(models[a], agents[a]["Xtr"], agents[a]["ytr"],
                               cfg, rng, cfg["local_epochs"])
            ns = np.array([ag["n"] for ag in agents], dtype=float)
            w = ns / ns.sum()
            avg = sum(w[a] * models[a].get_flat() for a in range(N))
            for a in range(N):
                models[a].set_flat(avg)
            comm += 2.0 * N * P            # upload + download full model

        elif method == "Gossip":
            for a in range(N):
                sgd_supervised(models[a], agents[a]["Xtr"], agents[a]["ytr"],
                               cfg, rng, cfg["local_epochs"])
            flats = np.stack([m.get_flat() for m in models])   # (N,P)
            mixed = A @ flats
            for a in range(N):
                models[a].set_flat(mixed[a])
            comm += 2.0 * N * P            # each device sends to 2 neighbors

        elif method == "FedDistill":
            comm += fd_round(models, agents, Xprobe, cfg, rng,
                             do_distill=(r > cfg["warmup"]))
        else:
            raise ValueError(method)

        la, _ = mean_local_acc(models, agents)
        ga = mean_global_acc(models, Xg, yg)
        hist["round"].append(r); hist["local"].append(la)
        hist["glob"].append(ga); hist["comm"].append(comm)

    hist["P"] = P
    hist["final_local"] = hist["local"][-1]
    hist["final_glob"] = hist["glob"][-1]
    hist["final_comm"] = hist["comm"][-1]
    return hist


# --------------------------------------------------------------------------
# Heterogeneous-architecture study: devices with different compute budgets.
# Weight-sharing (FedAvg/Gossip) cannot average mismatched tensors.
# FedDistill only shares predictions -> still works.
# --------------------------------------------------------------------------
def run_hetero(swarm, cfg, seed):
    rng = np.random.default_rng(9000 + seed)
    agents = swarm["agents"]
    N = len(agents)
    nin, nout = cfg["n_features"], cfg["n_classes"]
    Xg, yg = swarm["X_gtest"], swarm["y_gtest"]
    Xprobe = swarm["X_probe"]
    S, C = len(Xprobe), cfg["n_classes"]
    H_choices = [8, 16, 32, 64]
    Hs = [H_choices[a % len(H_choices)] for a in range(N)]

    def fresh():
        return [MLP(nin, Hs[a], nout, seed=123) for a in range(N)]

    # Local (heterogeneous, no sharing)
    models = fresh()
    pretrain_public(models, Xprobe, swarm["y_probe"], cfg, rng)
    for r in range(cfg["rounds"]):
        for a in range(N):
            sgd_supervised(models[a], agents[a]["Xtr"], agents[a]["ytr"],
                           cfg, rng, cfg["local_epochs"])
    local_la, _ = mean_local_acc(models, agents)
    local_ga = mean_global_acc(models, Xg, yg)

    # FedDistill (heterogeneous) -- shares predictions, architecture-agnostic
    models = fresh()
    pretrain_public(models, Xprobe, swarm["y_probe"], cfg, rng)
    comm = 0.0
    for r in range(cfg["rounds"]):
        comm += fd_round(models, agents, Xprobe, cfg, rng,
                         do_distill=(r >= cfg["warmup"]))
    fd_la, _ = mean_local_acc(models, agents)
    fd_ga = mean_global_acc(models, Xg, yg)

    return dict(Hs=Hs, local_local=local_la, local_glob=local_ga,
                fd_local=fd_la, fd_glob=fd_ga, fd_comm=comm)


# --------------------------------------------------------------------------
# Message-quantization study for FedDistill
# --------------------------------------------------------------------------
def run_fd_quant(swarm, cfg, seed, bits):
    rng = np.random.default_rng(11000 + seed)
    agents = swarm["agents"]
    N = len(agents)
    nin, nout = cfg["n_features"], cfg["n_classes"]
    Xprobe = swarm["X_probe"]
    S, C = len(Xprobe), cfg["n_classes"]
    models = [MLP(nin, cfg["hidden"], nout, seed=123) for _ in range(N)]
    pretrain_public(models, Xprobe, swarm["y_probe"], cfg, rng)
    for r in range(cfg["rounds"]):
        fd_round(models, agents, Xprobe, cfg, rng,
                 do_distill=(r >= cfg["warmup"]), quant_bits=bits)
    la, _ = mean_local_acc(models, agents)
    bytes_per_round = 2.0 * N * S * C * (bits / 8.0)
    return dict(bits=bits, local=la, bytes_per_round=bytes_per_round)


# ==========================================================================
# FIGURES
# ==========================================================================
def fig_noniid(swarm, cfg):
    cc = swarm["class_counts"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(cc, aspect="auto", cmap="viridis")
    ax.set_xlabel("Class (digit)"); ax.set_ylabel("Device (robot)")
    ax.set_xticks(range(cfg["n_classes"]))
    ax.set_yticks(range(cfg["n_agents"]))
    ax.set_title("Non-IID experience across devices (samples per class)")
    fig.colorbar(im, ax=ax, label="# samples")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_noniid.png"), dpi=150)
    plt.close(fig)


def fig_domainshift(swarm, cfg):
    digits = load_digits()
    base = digits.data[0] / 16.0          # a canonical "5"... actually digit 0
    agents = swarm["agents"]
    n_show = min(6, len(agents))
    fig, axes = plt.subplots(1, n_show + 1, figsize=(1.5 * (n_show + 1), 2.0))
    axes[0].imshow(base.reshape(8, 8), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("canonical", fontsize=8); axes[0].axis("off")
    rng = np.random.default_rng(0)
    for i in range(n_show):
        ag = agents[i]
        t = np.clip(ag["gain"] * base + ag["offset"]
                    + rng.normal(0, ag["sigma"], size=base.shape), 0, 1)
        axes[i + 1].imshow(t.reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[i + 1].set_title(f"dev {i}\ng={ag['gain']:.2f}", fontsize=8)
        axes[i + 1].axis("off")
    fig.suptitle("Same object seen through each device's sensor (domain shift)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_domainshift.png"), dpi=150)
    plt.close(fig)


def fig_convergence(agg):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, title, ax in [("local", "Mean LOCAL test accuracy", axes[0]),
                           ("glob", "GLOBAL test accuracy", axes[1])]:
        for m in ["Local", "Central", "FedAvg", "Gossip", "FedDistill"]:
            mean = agg[m][key + "_mean"]; std = agg[m][key + "_std"]
            rounds = np.arange(1, len(mean) + 1)
            ax.plot(rounds, mean, label=m, color=METHOD_COLORS[m],
                    lw=2, ls="--" if m == "Central" else "-")
            ax.fill_between(rounds, mean - std, mean + std, color=METHOD_COLORS[m], alpha=0.15)
        ax.set_xlabel("Communication round"); ax.set_ylabel("Accuracy")
        ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_convergence.png"), dpi=150)
    plt.close(fig)


def fig_pareto(agg, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = [("local", "Final LOCAL accuracy (personalization)", axes[0]),
              ("glob", "Final GLOBAL accuracy (generalization)", axes[1])]
    for key, title, ax in panels:
        for m in ["Local", "Central", "FedAvg", "Gossip", "FedDistill"]:
            x = agg[m]["comm_mean"][-1] * 4 / 1e6      # float32 -> MB
            y = agg[m][key + "_mean"][-1]
            yerr = agg[m][key + "_std"][-1]
            x_plot = max(x, 5e-4)                       # Local(0) visible on log axis
            ax.errorbar(x_plot, y, yerr=yerr, fmt="o", ms=10,
                        color=METHOD_COLORS[m], capsize=3, label=m)
            ax.annotate(m, (x_plot, y), textcoords="offset points",
                        xytext=(7, 5), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Total communication, %d rounds [MB, float32] (log)" % cfg["rounds"])
        ax.set_ylabel("Accuracy"); ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Accuracy vs communication cost (up-and-left is better)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_pareto.png"), dpi=150)
    plt.close(fig)


def fig_payload(agg, cfg):
    """Per-round payload by *what* is communicated."""
    P = agg["FedAvg"]["P"]
    N, S, C = cfg["n_agents"], cfg["probe_size"], cfg["n_classes"]
    raw_once = float(sum_raw)  # set globally below
    items = [
        ("Raw data\n(Central, once)", raw_once * 4 / 1e3, "#000000"),
        ("Weights\n(FedAvg)", 2 * N * P * 4 / 1e3, METHOD_COLORS["FedAvg"]),
        ("Weights\n(Gossip)", 2 * N * P * 4 / 1e3, METHOD_COLORS["Gossip"]),
        ("Predictions\n(FedDistill)", 2 * N * S * C * 4 / 1e3, METHOD_COLORS["FedDistill"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = range(len(items))
    ax.bar(xs, [v for _, v, _ in items], color=[c for _, _, c in items])
    ax.set_xticks(list(xs)); ax.set_xticklabels([k for k, _, _ in items], fontsize=8)
    ax.set_ylabel("Payload per round  [kB]")
    ax.set_title("What is communicated, per round (our 2.4k-param model)")
    for i, (_, v, _) in enumerate(items):
        ax.annotate(f"{v:.0f} kB", (i, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_payload.png"), dpi=150)
    plt.close(fig)


def fig_hetero(hagg):
    groups = ["Local\n(hetero)", "FedAvg", "Gossip", "FedDistill\n(hetero)"]
    local_vals = [hagg["local_local_mean"], 0, 0, hagg["fd_local_mean"]]
    local_err = [hagg["local_local_std"], 0, 0, hagg["fd_local_std"]]
    glob_vals = [hagg["local_glob_mean"], 0, 0, hagg["fd_glob_mean"]]
    glob_err = [hagg["local_glob_std"], 0, 0, hagg["fd_glob_std"]]
    x = np.arange(len(groups)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(x - w / 2, local_vals, w, yerr=local_err, capsize=3,
           color="#9e9e9e", label="local acc (own environment)")
    ax.bar(x + w / 2, glob_vals, w, yerr=glob_err, capsize=3,
           color="#d62728", label="global acc (generalization)")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Final accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Heterogeneous device architectures (H in {8,16,32,64})")
    for i in (1, 2):
        ax.annotate("N/A\ncannot average\nmismatched weights", (i, 0.03),
                    ha="center", va="bottom", fontsize=8, color="white",
                    bbox=dict(boxstyle="round", fc="#1f77b4", ec="none", alpha=0.9))
    for xi, v in [(0, local_vals[0]), (3, local_vals[3])]:
        ax.annotate(f"{v:.3f}", (xi - w / 2, v), ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 3), fontsize=8)
    for xi, v in [(0, glob_vals[0]), (3, glob_vals[3])]:
        ax.annotate(f"{v:.3f}", (xi + w / 2, v), ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 3), fontsize=8)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_hetero.png"), dpi=150)
    plt.close(fig)


def fig_quant(qagg):
    bits = [q["bits"] for q in qagg]
    acc = [q["local_mean"] for q in qagg]
    err = [q["local_std"] for q in qagg]
    kb = [q["bytes_per_round"] / 1e3 for q in qagg]
    labels = [f"{b}-bit" if b < 32 else "float32" for b in bits]
    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    xs = range(len(bits))
    ax1.bar(xs, acc, yerr=err, capsize=3, color="#d62728", alpha=0.85)
    ax1.set_ylabel("Final mean LOCAL accuracy", color="#d62728")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(list(xs)); ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.plot(xs, kb, "o-", color="#1f77b4", lw=2)
    ax2.set_ylabel("Payload per round [kB]", color="#1f77b4")
    ax1.set_title("FedDistill: quantizing the *messages* (soft labels)")
    for i, a in enumerate(acc):
        ax1.annotate(f"{a:.3f}", (i, a), ha="center", va="bottom",
                     textcoords="offset points", xytext=(0, 3), fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_quant.png"), dpi=150)
    plt.close(fig)


def fig_scaling(cfg):
    """Analytical: communication per round vs model size."""
    N, S, C = cfg["n_agents"], cfg["probe_size"], cfg["n_classes"]
    Ps = np.logspace(3, 7, 100)          # 1e3 .. 1e7 params
    fedavg = 2 * N * Ps * 4 / 1e6        # MB
    fd = np.full_like(Ps, 2 * N * S * C * 4 / 1e6)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.loglog(Ps, fedavg, color=METHOD_COLORS["FedAvg"], lw=2,
              label="FedAvg / Gossip  (weights) $\\propto$ model size")
    ax.loglog(Ps, fd, color=METHOD_COLORS["FedDistill"], lw=2,
              label="FedDistill (predictions): constant in model size")
    for P, name in [(2410, "our MLP (2.4k)"), (1e5, "TinyML (100k)"),
                    (1e7, "edge-LLM head (10M)")]:
        ax.axvline(P, color="gray", ls=":", alpha=0.5)
        ax.text(P, 250, name, rotation=90, fontsize=7.5, color="dimgray",
                ha="right", va="top")
    ax.set_xlim(8e2, 1.6e7)
    ax.set_xlabel("Model parameters  P  (log)")
    ax.set_ylabel("Communication per round  [MB, log]")
    ax.set_title("Why predictions scale and weights do not")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_scaling.png"), dpi=150)
    plt.close(fig)


def fig_peragent(swarm, cfg, per_agent_runs):
    """Per-agent local accuracy: Local vs FedAvg vs FedDistill (seed 0)."""
    N = cfg["n_agents"]
    width = 0.27
    xs = np.arange(N)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for k, m in enumerate(["Local", "FedAvg", "FedDistill"]):
        ax.bar(xs + (k - 1) * width, per_agent_runs[m], width,
               label=m, color=METHOD_COLORS[m])
    ax.set_xlabel("Device (robot)"); ax.set_ylabel("Local test accuracy")
    ax.set_xticks(xs); ax.set_ylim(0, 1)
    ax.set_title("Per-device personalization (collaboration helps the data-poor)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_peragent.png"), dpi=150)
    plt.close(fig)


# ==========================================================================
# MAIN
# ==========================================================================
sum_raw = 0.0  # set after first swarm build (for payload figure)


def aggregate(histories):
    """histories: list (over seeds) of hist dicts -> mean/std arrays."""
    keys = ["local", "glob", "comm"]
    out = {}
    for k in keys:
        M = np.array([h[k] for h in histories])
        out[k + "_mean"] = M.mean(0)
        out[k + "_std"] = M.std(0)
    out["P"] = histories[0]["P"]
    return out


def main():
    global sum_raw
    cfg = CFG
    methods = ["Local", "Central", "FedAvg", "Gossip", "FedDistill"]
    results = {"config": cfg}

    # ----- main comparison over seeds -----
    raw_hist = {m: [] for m in methods}
    swarm0 = None
    per_agent_runs = {}
    for si, seed in enumerate(cfg["seeds"]):
        swarm = make_swarm(cfg, seed)
        if swarm0 is None:
            swarm0 = swarm
            sum_raw = float(sum(ag["Xtr"].shape[0] * cfg["n_features"]
                                for ag in swarm["agents"]))
        for m in methods:
            h = run_method(m, swarm, cfg, seed)
            raw_hist[m].append(h)
        print(f"[main] seed {seed} done")

    agg = {m: aggregate(raw_hist[m]) for m in methods}

    # per-agent breakdown (seed 0) for fig_peragent
    s0 = make_swarm(cfg, cfg["seeds"][0])

    def per_agent(method, swarm):
        N = cfg["n_agents"]; nin, nout = cfg["n_features"], cfg["n_classes"]
        rng = np.random.default_rng(7000 + cfg["seeds"][0])
        models = [MLP(nin, cfg["hidden"], nout, seed=123) for _ in range(N)]
        Xprobe = swarm["X_probe"]
        pretrain_public(models, Xprobe, swarm["y_probe"], cfg, rng)
        for r in range(1, cfg["rounds"] + 1):
            if method == "FedDistill":
                fd_round(models, swarm["agents"], Xprobe, cfg, rng,
                         do_distill=(r > cfg["warmup"]))
            else:
                for a in range(N):
                    sgd_supervised(models[a], swarm["agents"][a]["Xtr"],
                                   swarm["agents"][a]["ytr"], cfg, rng, cfg["local_epochs"])
                if method == "FedAvg":
                    ns = np.array([ag["n"] for ag in swarm["agents"]], float); w = ns / ns.sum()
                    avg = sum(w[a] * models[a].get_flat() for a in range(N))
                    for a in range(N):
                        models[a].set_flat(avg)
        out = []
        for a in range(N):
            yte = swarm["agents"][a]["yte"]
            if len(yte) == 0:
                out.append(0.0); continue
            out.append(float((models[a].predict(swarm["agents"][a]["Xte"]) == yte).mean()))
        return out
    for m in ["Local", "FedAvg", "FedDistill"]:
        per_agent_runs[m] = per_agent(m, s0)

    # ----- heterogeneous architectures -----
    h_runs = [run_hetero(make_swarm(cfg, s), cfg, s) for s in cfg["seeds"]]
    hagg = dict(
        local_local_mean=float(np.mean([r["local_local"] for r in h_runs])),
        local_local_std=float(np.std([r["local_local"] for r in h_runs])),
        local_glob_mean=float(np.mean([r["local_glob"] for r in h_runs])),
        local_glob_std=float(np.std([r["local_glob"] for r in h_runs])),
        fd_local_mean=float(np.mean([r["fd_local"] for r in h_runs])),
        fd_local_std=float(np.std([r["fd_local"] for r in h_runs])),
        fd_glob_mean=float(np.mean([r["fd_glob"] for r in h_runs])),
        fd_glob_std=float(np.std([r["fd_glob"] for r in h_runs])),
        Hs=h_runs[0]["Hs"],
    )

    # ----- quantization study -----
    qagg = []
    for bits in [32, 8, 4]:
        runs = [run_fd_quant(make_swarm(cfg, s), cfg, s, bits) for s in cfg["seeds"]]
        qagg.append(dict(bits=bits,
                         local_mean=float(np.mean([r["local"] for r in runs])),
                         local_std=float(np.std([r["local"] for r in runs])),
                         bytes_per_round=runs[0]["bytes_per_round"]))

    # ----- figures -----
    fig_noniid(swarm0, cfg)
    fig_domainshift(swarm0, cfg)
    fig_convergence(agg)
    fig_pareto(agg, cfg)
    fig_payload(agg, cfg)
    fig_hetero(hagg)
    fig_quant(qagg)
    fig_scaling(cfg)
    fig_peragent(swarm0, cfg, per_agent_runs)

    # ----- numeric summary -> results.json -----
    summary = {}
    for m in methods:
        summary[m] = dict(
            final_local=float(agg[m]["local_mean"][-1]),
            final_local_std=float(agg[m]["local_std"][-1]),
            final_glob=float(agg[m]["glob_mean"][-1]),
            final_glob_std=float(agg[m]["glob_std"][-1]),
            total_comm_floats=float(agg[m]["comm_mean"][-1]),
            total_comm_MB=float(agg[m]["comm_mean"][-1] * 4 / 1e6),
            params=int(agg[m]["P"]),
        )
    results["summary"] = summary
    results["hetero"] = hagg
    results["quant"] = qagg
    results["raw_upload_MB"] = float(sum_raw * 4 / 1e6)
    # device statistics
    ns = [int(ag["n"]) for ag in swarm0["agents"]]
    results["device_train_sizes"] = ns
    results["probe_size"] = cfg["probe_size"]
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ----- console table -----
    print("\n=== FINAL RESULTS (mean over seeds) ===")
    print(f"{'method':<12}{'local':>9}{'global':>9}{'comm[MB]':>12}{'params':>9}")
    for m in methods:
        s = summary[m]
        print(f"{m:<12}{s['final_local']:>9.3f}{s['final_glob']:>9.3f}"
              f"{s['total_comm_MB']:>12.4f}{s['params']:>9}")
    print("\nHetero (final local acc):  "
          f"Local={hagg['local_local_mean']:.3f}  FedDistill={hagg['fd_local_mean']:.3f}  "
          "(FedAvg/Gossip: N/A)")
    print("Quant (final local acc):  " +
          "  ".join(f"{q['bits']}b={q['local_mean']:.3f}" for q in qagg))
    print(f"Raw-upload baseline (Central): {results['raw_upload_MB']:.4f} MB")
    print("\nFigures written to:", FIG_DIR)


if __name__ == "__main__":
    main()
