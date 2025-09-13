# np_climsim_combined.py
# ------------------------------------------------------------
# Neyman–Pearson (Target + Source) with decision rule
# Multidimensional Gaussians + switchable model (Linear/MLP)
# Robust SGDA + projection with temperatured logistic surrogate (k_sur).
#
# Single-file version: contains all helpers and a main() runner at the end.
#
# Stability & clarity updates (2025-09):
#   - Notebook-safe path resolver (no __file__ needed in notebooks).
#   - Standardization helpers for tabular features (compute_standardizer).
#   - Gentler model: optional LayerNorm and positive final bias to avoid
#     initial all-negative logits (build_model).
#   - Adam weight decay everywhere to reduce drift.
#   - Consistent inclusion of `delta` in g1 across stages.
#   - Epsilon constants printed with actual n's and c's (optimize_alpha_prime).
#   - Pretraining warmup (unconstrained logistic) before NP training.
#   - Caps on dual variables (λ) to avoid domination when α is tight.
#   - Floor on tightened α used for θ* training to avoid α_tight≈0 pathologies.
#   - Failsafe retry for θ* if it returns degenerate R1≈1.
# ------------------------------------------------------------

import os
import math
import torch
from torch import nn
from climsim_data_import import load_climsim_data, prepare_climsim_data

torch.manual_seed(0)

# =========================
# Utilities
# =========================

def eps_from_n(n, c=1.0):
    return float(c) / math.sqrt(float(n))

def get_activation(name: str):
    name = (name or "relu").lower()
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU(), "sigmoid": nn.Sigmoid()}.get(name, nn.ReLU())

def xavier_small_(lin: nn.Linear, gain=0.1):
    nn.init.xavier_uniform_(lin.weight, gain=gain)
    nn.init.zeros_(lin.bias)

# ===== Standardization & debug =====
@torch.no_grad()
def compute_standardizer(*train_tensors):
    """
    Returns (apply_fn, mean, std) where apply_fn standardizes inputs using TRAIN stats.
    """
    cat = torch.cat([t.float() for t in train_tensors], dim=0)
    mean = cat.mean(dim=0, keepdim=True)
    std  = cat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    def apply(x): return (x.float() - mean) / std
    return apply, mean, std

@torch.no_grad()
def debug_logits(model, x0, x1, tag=""):
    h0 = model(x0).flatten()
    h1 = model(x1).flatten()
    def stats(v):
        # guard quantiles if tensor is small
        q10 = float(torch.quantile(v, 0.10)) if v.numel() > 10 else float(v.min())
        q50 = float(torch.quantile(v, 0.50)) if v.numel() > 1 else float(v.mean())
        q90 = float(torch.quantile(v, 0.90)) if v.numel() > 10 else float(v.max())
        return dict(mean=float(v.mean()),
                    std=float(v.std(unbiased=False)),
                    p_pos=float((v > 0).float().mean()),
                    q10=q10, q50=q50, q90=q90)
    info = {"tag": tag, "h0": stats(h0), "h1": stats(h1)}
    print(f"[logits {tag}] h0: {info['h0']}, h1: {info['h1']}")
    return info

def build_model(d,
                model_type="linear",
                hidden_sizes=(32, 32),
                activation="relu",
                init="xavier_small",
                use_layernorm=True,
                init_final_bias=0.5):
    """
    - LayerNorm helps with scale/shift after standardization.
    - init_final_bias > 0 nudges away from 'all-negative' logits at init.
    """
    act = get_activation(activation)
    if model_type == "mlp":
        layers = []
        if use_layernorm:
            layers.append(nn.LayerNorm(d))
        in_dim = d
        for h in (hidden_sizes or []):
            lin = nn.Linear(in_dim, int(h))
            if init == "xavier_small":
                xavier_small_(lin, gain=0.1)
            layers += [lin, act]
            in_dim = int(h)
        out = nn.Linear(in_dim, 1)
        if init == "xavier_small":
            xavier_small_(out, gain=0.1)
        if init_final_bias is not None:
            with torch.no_grad():
                out.bias.fill_(float(init_final_bias))
        layers += [out]
        return nn.Sequential(*layers)
    else:
        mods = []
        if use_layernorm:
            mods.append(nn.LayerNorm(d))
        lin = nn.Linear(d, 1)
        if init == "xavier_small":
            xavier_small_(lin, gain=0.1)
        if init_final_bias is not None:
            with torch.no_grad():
                lin.bias.fill_(float(init_final_bias))
        mods.append(lin)
        return nn.Sequential(*mods)

def make_model_factory(d,
                       model_type="linear",
                       hidden_sizes=(32, 32),
                       activation="relu",
                       init="xavier_small",
                       use_layernorm=True,
                       init_final_bias=0.5):
    def factory():
        return build_model(d, model_type, hidden_sizes, activation, init,
                           use_layernorm=use_layernorm, init_final_bias=init_final_bias)
    return factory

@torch.no_grad()
def polyak_update(avg_model: nn.Module, model: nn.Module, t: int):
    for p_avg, p in zip(avg_model.parameters(), model.parameters()):
        p_avg.add_(p - p_avg, alpha=1.0 / float(t))

def make_gaussian(n, d, t, c, generator=None):
    x = torch.randn(n, d, generator=generator) if generator is not None else torch.randn(n, d)
    return x * math.sqrt(c) + t * torch.ones(1, d)

def make_pair_sep(n0, n1, d, t0, c0, t1, c1, seed=None):
    g = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    x0 = make_gaussian(n0, d, t0, c0, g)
    x1 = make_gaussian(n1, d, t1, c1, g)
    return x0, x1

# temperatured logistic surrogate (still in [0,1])
def phi_pos(z, k_sur):   # for R0: φ(h)
    return torch.sigmoid(k_sur * z)

def phi_neg(z, k_sur):   # for R1: φ(-h) implemented as σ(-k*h)
    return torch.sigmoid(-k_sur * z)

def evaluate_target_test(model, k_sur, x0t, x1t, title):
    with torch.no_grad():
        h0 = model(x0t); h1 = model(x1t)
        R0T_sur = phi_pos(h0, k_sur).mean().item()
        R1T_sur = phi_neg(h1, k_sur).mean().item()
        typeI   = (h0 > 0).float().mean().item()
        typeII  = (h1 <= 0).float().mean().item()
    print(f"\n=== {title} (N={x0t.shape[0]} per class) ===")
    print(f"Surrogate: R0_T={R0T_sur:.4f}, R1_T={R1T_sur:.4f}")
    print(f"0–1 error: Type-I_T={typeI:.4f}, Type-II_T={typeII:.4f}")
    return {"R0T_sur": R0T_sur, "R1T_sur": R1T_sur, "typeI": typeI, "typeII": typeII}

# =========================
# Constraint helpers & projection
# =========================

def g_values_stage2(model, meta, alpha_prime_hat):
    x0T = meta["train"]["x0T"]; x1T = meta["train"]["x1T"]; x0S = meta["train"]["x0S"]
    alpha = meta["train"]["alpha"]; k_sur = meta["train"]["k_sur"]; delta = meta["train"]["delta"]
    eps0T_plus = meta["eps0T_plus"]; eps0S = meta["eps0S"]; eps1T = meta["eps1T"]
    R1_star = meta["R1_star_train"]
    R0_T = phi_pos(model(x0T), k_sur).mean()
    R0_S = phi_pos(model(x0S), k_sur).mean()
    R1_T = phi_neg(model(x1T), k_sur).mean()
    # Constraints use α + ε0T_plus and include delta consistently
    g1 = R0_T - alpha - eps0T_plus - delta
    g2 = R0_S - alpha_prime_hat - eps0S
    g3 = R1_T - R1_star - eps1T
    return g1, g2, g3

def project_to_feasible_max(model, compute_gmax, steps=2500, lr=0.01, tol=1e-7):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        g = compute_gmax(model)
        penalty = torch.relu(g)**2
        if penalty.item() <= tol:
            break
        penalty.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

def project_to_feasible_three(model, compute_g123, steps=2500, lr=0.01, tol=1e-7):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        g1, g2, g3 = compute_g123(model)
        penalty = torch.relu(g1)**2 + torch.relu(g2)**2 + torch.relu(g3)**2
        if penalty.item() <= tol:
            break
        penalty.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

# =========================
# Warmup & Base NP on TARGET
# =========================

def pretrain_unconstrained(model, x0, x1, steps=300, lr=1e-2):
    """
    Simple logistic pretraining to set a sensible sign before NP constraints bite.
    """
    y0 = -torch.ones(x0.size(0), 1); y1 = torch.ones(x1.size(0), 1)
    X = torch.cat([x0, x1], dim=0); Y = torch.cat([y0, y1], dim=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        h = model(X)
        # logistic loss log(1+exp(-y h))
        loss = torch.log1p(torch.exp(-Y * h)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

def train_np_sigmoid(alpha, k_sur, T, eta_theta, eta_lambda, delta, gamma,
                     x0, x1, model_factory, lam_cap=5.0):
    model = model_factory()
    # Warm-start to avoid "all negative" basin
    pretrain_unconstrained(model, x0, x1, steps=300, lr=1e-2)

    def R0_hat(): return phi_pos(model(x0), k_sur).mean()
    def R1_hat(): return phi_neg(model(x1), k_sur).mean()
    def f(): return R1_hat()
    def g(): return R0_hat() - alpha - delta
    opt = torch.optim.Adam(model.parameters(), lr=eta_theta, weight_decay=1e-5)
    lam = 0.0

    # debug at start
    debug_logits(model, x0, x1, tag="init NP")

    for _ in range(T):
        opt.zero_grad(set_to_none=True)
        L = f() + lam * g()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        lam = max((1 - gamma * eta_lambda) * lam + eta_lambda * g().detach().item(), 0.0)
        if lam_cap is not None:
            lam = min(lam, float(lam_cap))

    with torch.no_grad():
        R0_tr = R0_hat().item(); R1_tr = R1_hat().item()
        typeI_tr = (model(x0) > 0).float().mean().item()
        typeII_tr = (model(x1) <= 0).float().mean().item()

        # Escape saturated all-negative basin if it happens
        if R0_tr < 0.05 and R1_tr > 0.95 and typeI_tr < 0.01 and typeII_tr > 0.99:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    m.weight.mul_(-1); m.bias.mul_(-1)
            # short polishing pass
            for _ in range(200):
                opt.zero_grad(set_to_none=True)
                L = f() + lam * g()
                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

    debug_logits(model, x0, x1, tag="final NP")
    return model, R0_tr, R1_tr, typeI_tr, typeII_tr

def solve_theta_star_T_tight(x0T, x1T, alpha, eps0T_minus, k_sur, T_inner,
                             eta_theta, eta_lambda, delta, gamma, model_factory,
                             alpha_floor=0.05):
    """
    Train θ* with α tightened, but floor α_tight to avoid extremely small values.
    Retry once with gentler settings if degenerate.
    """
    alpha_tight = max(alpha - eps0T_minus, alpha_floor)

    model, _, R1_star, *_ = train_np_sigmoid(
        alpha=alpha_tight, k_sur=k_sur, T=T_inner,
        eta_theta=eta_theta, eta_lambda=eta_lambda, delta=delta, gamma=gamma,
        x0=x0T, x1=x1T, model_factory=model_factory, lam_cap=5.0
    )

    # Failsafe: if degenerate, retry once with gentler dual and lower temp
    if R1_star > 0.98:
        model, _, R1_star, *_ = train_np_sigmoid(
            alpha=alpha_tight, k_sur=max(k_sur * 0.7, 0.5), T=max(800, T_inner // 2),
            eta_theta=eta_theta, eta_lambda=min(1.0, eta_lambda), delta=delta, gamma=gamma,
            x0=x0T, x1=x1T, model_factory=model_factory, lam_cap=3.0
        )

    return model, float(R1_star)

# ============================================
# Minimize α′ with constraints (averaged α′)
# ============================================

def optimize_alpha_prime(
    alpha,
    # separate variances for TARGET and SOURCE (class 0 vs 1)
    x0T, x1T, x0S,
    model_type="linear", hidden_sizes=(32, 32), activation="relu", init="xavier_small",
    use_layernorm=True, init_final_bias=0.5,
    k_sur=3.0,
    # separate epsilon constants (now with +/- for 0T)
    c_eps0T_minus=1.0, c_eps0T_plus=1.0, c_eps0S=1.0, c_eps1T=1.0,
    T_outer=2000,
    eta_theta=0.01, eta_alpha=0.02,
    eta_lambda1=5.0, eta_lambda2=5.0, eta_lambda3=5.0,
    gamma1=0.0, gamma2=0.0, gamma3=0.0,
    delta=0.0, T_inner_np=1500
):
    # infer sample counts and feature dimension from provided tensors
    n0T, d = x0T.shape
    n1T, _ = x1T.shape
    n0S, _ = x0S.shape

    # epsilons (each has its own constant)
    eps0T_minus = eps_from_n(n0T, c_eps0T_minus)  # for α - ε0T_minus
    eps0T_plus  = eps_from_n(n0T, c_eps0T_plus)   # for α + ε0T_plus (constraints)
    eps0S       = eps_from_n(n0S, c_eps0S)
    eps1T       = eps_from_n(n1T, c_eps1T)

    # debug: show the exact eps in use
    print(f"[eps] n0T={n0T}, n1T={n1T}, n0S={n0S}")
    print(f"[eps] c_minus={c_eps0T_minus}, c_plus={c_eps0T_plus}, c0S={c_eps0S}, c1T={c_eps1T}")
    print(f"[eps] eps0T_minus={eps0T_minus:.6f}, eps0T_plus={eps0T_plus:.6f}, eps0S={eps0S:.6f}, eps1T={eps1T:.6f}")

    model_factory = make_model_factory(
        d, model_type, hidden_sizes, activation, init,
        use_layernorm=use_layernorm, init_final_bias=init_final_bias
    )

    # θ* with tightened α - ε0T_minus (floored)
    theta_star_model, R1_star = solve_theta_star_T_tight(
        x0T, x1T, alpha, eps0T_minus, k_sur, T_inner_np,
        eta_theta, eta_lambda1, delta, gamma1, model_factory
    )

    model = model_factory()
    def R0_T_hat(): return phi_pos(model(x0T), k_sur).mean()
    def R1_T_hat(): return phi_neg(model(x1T), k_sur).mean()
    def R0_S_hat(): return phi_pos(model(x0S), k_sur).mean()

    alpha_prime = torch.tensor(alpha, requires_grad=True)
    lam1 = lam2 = lam3 = 0.0
    opt_theta = torch.optim.Adam(model.parameters(), lr=eta_theta, weight_decay=1e-5)
    alpha_prime_avg = 0.0

    for t in range(1, T_outer + 1):
        opt_theta.zero_grad(set_to_none=True)
        if alpha_prime.grad is not None: alpha_prime.grad.zero_()
        # g1 uses α + ε0T_plus  (+ delta consistently)
        g1 = R0_T_hat() - alpha - eps0T_plus - delta
        g2 = R0_S_hat() - alpha_prime - eps0S
        g3 = R1_T_hat() - R1_star - eps1T
        (alpha_prime + lam1*g1 + lam2*g2 + lam3*g3).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt_theta.step()
        with torch.no_grad():
            alpha_prime -= eta_alpha * (1.0 - lam2)
            alpha_prime.clamp_(min=alpha)
            alpha_prime_avg += (alpha_prime.item() - alpha_prime_avg) / t
            lam1 = max((1 - gamma1*eta_lambda1)*lam1 + eta_lambda1*g1.detach().item(), 0.0)
            lam2 = max((1 - gamma2*eta_lambda2)*lam2 + eta_lambda2*g2.detach().item(), 0.0)
            lam3 = max((1 - gamma3*eta_lambda3)*lam3 + eta_lambda3*g3.detach().item(), 0.0)
            # cap duals to avoid domination
            lam1 = min(lam1, 5.0); lam2 = min(lam2, 5.0); lam3 = min(lam3, 5.0)

    return {
        "alpha_prime": alpha_prime_avg,
        "eps0T_minus": eps0T_minus, "eps0T_plus": eps0T_plus,
        "eps0S": eps0S, "eps1T": eps1T,
        "R1_star_train": R1_star,
        "theta_star_model": theta_star_model,
        "train": {
            "x0T": x0T, "x1T": x1T, "x0S": x0S,
            "alpha": alpha, "k_sur": k_sur, "delta": delta,
            "d": d,
            "model_cfg": dict(model_type=model_type, hidden_sizes=hidden_sizes,
                              activation=activation, init=init,
                              use_layernorm=use_layernorm, init_final_bias=init_final_bias)
        }
    }

# ============================================================
# Stage 2: R*_{1,T}(H(α′)) — min R1_T s.t. g1,g2,g3<=0  + projection
# ============================================================

def solve_min_R1_given_alpha_prime(alpha_prime_hat, meta, T=3000,
                                   eta_theta=0.01,
                                   eta_lambda1=5.0, eta_lambda2=5.0, eta_lambda3=5.0,
                                   gamma1=0.0, gamma2=0.0, gamma3=0.0,
                                   proj_steps=2500, proj_lr=0.01):
    x0T = meta["train"]["x0T"]; x1T = meta["train"]["x1T"]; x0S = meta["train"]["x0S"]
    alpha = meta["train"]["alpha"]; k_sur = meta["train"]["k_sur"]; delta = meta["train"]["delta"]
    eps0T_plus = meta["eps0T_plus"]; eps0S = meta["eps0S"]; eps1T = meta["eps1T"]
    R1_star = meta["R1_star_train"]
    cfg = meta["train"]["model_cfg"]; d = meta["train"]["d"]
    model_factory = make_model_factory(d, **cfg)

    model = model_factory()
    avg_model = model_factory()
    opt_theta = torch.optim.Adam(model.parameters(), lr=eta_theta, weight_decay=1e-5)

    def R0_T(m): return phi_pos(m(x0T), k_sur).mean()
    def R0_S(m): return phi_pos(m(x0S), k_sur).mean()
    def R1_T(m): return phi_neg(m(x1T), k_sur).mean()

    lam1 = lam2 = lam3 = 0.0
    for t in range(1, T + 1):
        opt_theta.zero_grad(set_to_none=True)
        # g1 uses α + ε0T_plus + delta consistently
        g1 = R0_T(model) - alpha - eps0T_plus - delta
        g2 = R0_S(model) - alpha_prime_hat - eps0S
        g3 = R1_T(model) - R1_star - eps1T
        (R1_T(model) + lam1*g1 + lam2*g2 + lam3*g3).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt_theta.step()
        with torch.no_grad():
            if t == 1: avg_model.load_state_dict(model.state_dict())
            else:      polyak_update(avg_model, model, t)
            lam1 = max((1 - gamma1*eta_lambda1)*lam1 + eta_lambda1*g1.item(), 0.0)
            lam2 = max((1 - gamma2*eta_lambda2)*lam2 + eta_lambda2*g2.item(), 0.0)
            lam3 = max((1 - gamma3*eta_lambda3)*lam3 + eta_lambda3*g3.item(), 0.0)
            # cap duals
            lam1 = min(lam1, 5.0); lam2 = min(lam2, 5.0); lam3 = min(lam3, 5.0)

    def g123(m): return g_values_stage2(m, meta, alpha_prime_hat)
    project_to_feasible_three(avg_model, g123, steps=proj_steps, lr=proj_lr)

    with torch.no_grad():
        R1_train_val = R1_T(avg_model).item()
        g1p, g2p, g3p = g_values_stage2(avg_model, meta, alpha_prime_hat)
        print(f"[Stage2 projection] g1={g1p.item():.4f}, g2={g2p.item():.4f}, g3={g3p.item():.4f}")
    return avg_model, R1_train_val

# ============================================================
# R1,S* over H(α′): value only (used for decision threshold)
# ============================================================

def compute_R1S_star_over_Halpha(meta_alpha, alpha_prime_hat, x1S,
                                 T=1500, eta_theta=0.01,
                                 eta_lambda1=5.0, eta_lambda2=5.0, eta_lambda3=5.0):
    k_sur = meta_alpha["train"]["k_sur"]
    x0T = meta_alpha["train"]["x0T"]; x1T = meta_alpha["train"]["x1T"]; x0S = meta_alpha["train"]["x0S"]
    alpha = meta_alpha["train"]["alpha"]; delta = meta_alpha["train"]["delta"]
    eps0T_plus = meta_alpha["eps0T_plus"]; eps0S = meta_alpha["eps0S"]; eps1T = meta_alpha["eps1T"]
    R1_star = meta_alpha["R1_star_train"]
    cfg = meta_alpha["train"]["model_cfg"]; d = meta_alpha["train"]["d"]
    model_factory = make_model_factory(d, **cfg)

    model = model_factory()
    opt = torch.optim.Adam(model.parameters(), lr=eta_theta, weight_decay=1e-5)

    def R0_T(m): return phi_pos(m(x0T), k_sur).mean()
    def R0_S(m): return phi_pos(m(x0S), k_sur).mean()
    def R1_T(m): return phi_neg(m(x1T), k_sur).mean()
    def R1_S(m): return phi_neg(m(x1S), k_sur).mean()

    # Note: we keep lam1..3 at 0 (value-only solve), but keep code ready if needed.
    for _ in range(T):
        opt.zero_grad(set_to_none=True)
        g1 = R0_T(model) - alpha - eps0T_plus - delta
        g2 = R0_S(model) - alpha_prime_hat - eps0S
        g3 = R1_T(model) - R1_star - eps1T
        (R1_S(model) + 0.0*g1 + 0.0*g2 + 0.0*g3).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        return R1_S(model).item()

# ============================================================
# Stage 3: FINAL — min R1_S s.t. max{g1,g2,g3,g4}<=0  + projection
# ============================================================

def solve_min_R1S_final(meta_alpha, R1T_star_Halph,
                        x1S, T=3000,
                        eta_theta=0.01, eta_lambda=5.0, gamma=0.0,
                        lam_cap=None,
                        init_model=None, d=1,
                        proj_steps=3000, proj_lr=0.01):
    k_sur = meta_alpha["train"]["k_sur"]
    x0T = meta_alpha["train"]["x0T"]; x1T = meta_alpha["train"]["x1T"]; x0S = meta_alpha["train"]["x0S"]
    alpha = meta_alpha["train"]["alpha"]; delta = meta_alpha["train"]["delta"]
    eps0T_plus = meta_alpha["eps0T_plus"]; eps0S = meta_alpha["eps0S"]; eps1T = meta_alpha["eps1T"]
    alpha_prime_hat = float(meta_alpha["alpha_prime"])
    R1T_baseline_tight = float(meta_alpha["R1_star_train"])
    cfg = meta_alpha["train"]["model_cfg"]; d = meta_alpha["train"]["d"]
    model_factory = make_model_factory(d, **cfg)

    def R0_T(m): return phi_pos(m(x0T), k_sur).mean()
    def R0_S(m): return phi_pos(m(x0S), k_sur).mean()
    def R1_T(m): return phi_neg(m(x1T), k_sur).mean()
    def R1_S(m): return phi_neg(m(x1S), k_sur).mean()

    model = model_factory()
    if init_model is not None:
        model.load_state_dict(init_model.state_dict())
    avg_model = model_factory()
    opt = torch.optim.Adam(model.parameters(), lr=eta_theta, weight_decay=1e-5)
    lam = 0.0

    def g_max(m):
        # g1 uses α + ε0T_plus + delta consistently
        g1 = R0_T(m) - (alpha + eps0T_plus + delta)
        g2 = R0_S(m) - (alpha_prime_hat + eps0S)
        g3 = R1_T(m) - R1T_baseline_tight - eps1T
        g4 = R1_T(m) - R1T_star_Halph   - eps1T
        return torch.stack([g1, g2, g3, g4]).max()

    for t in range(1, T + 1):
        opt.zero_grad(set_to_none=True)
        obj = R1_S(model)
        g_val = g_max(model)
        (obj + lam * g_val).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            if t == 1: avg_model.load_state_dict(model.state_dict())
            else:      polyak_update(avg_model, model, t)
            lam = (1 - gamma * eta_lambda) * lam + eta_lambda * g_val.item()
            lam = max(lam, 0.0)
            if lam_cap is not None:
                lam = min(lam, float(lam_cap))

    project_to_feasible_max(avg_model, g_max, steps=proj_steps, lr=proj_lr)

    with torch.no_grad():
        R1S_train = R1_S(avg_model).item()
        g_after = g_max(avg_model).item()
        print(f"[Final projection] max(g1,g2,g3,g4)={g_after:.4f}")

    return avg_model, R1S_train, x1S

# ============================================================
# Runner (main)
# ============================================================

def resolve_climsim_root():
    """Resolve a sensible default for the climsim data root, notebook-safe."""
    env = os.getenv("CLIMSIM_ROOT", None)
    if env and os.path.isdir(env):
        return env
    # fallback to CWD when __file__ is not defined (e.g., notebooks)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    for cand in [
        os.path.join(here, "data", "climsim_data"),
        os.path.join(os.path.dirname(here), "data", "climsim_data"),
        os.path.join(os.getcwd(), "data", "climsim_data"),
    ]:
        if os.path.isdir(cand):
            return cand
    return None

def main():
    torch.manual_seed(0)

    # ------ Data / model controls ------
    d = 124

    # Intended sample sizes (loader may cap/override based on availability)
    n0T, n1T = 40, 40
    n0S, n1S = 1000, 1000

    # Model choice
    model_type   = "mlp"     # "linear" or "mlp"
    hidden_sizes = (64, 64)
    activation   = "relu"
    init         = "xavier_small"

    # NP + optimization knobs
    alpha = 0.10             # try 0.4, 0.3, 0.2; small α is now more stable
    k_sur = 1.0              # gentler temperature to avoid saturation

    # ---- Separate epsilon constants (with +/- for 0T) ----
    c_eps0T_minus = 0.1     # for ε0T_minus used in α - ε0T_minus
    c_eps0T_plus  = 0.1     # for ε0T_plus  used in α + ε0T_plus constraints
    c_eps0S       = 0.50     # for ε0S
    c_eps1T       = 1.00     # for ε1T
    c_eps1S       = 1.00     # for ε1S (decision rule only)

    # Training lengths
    T_inner_np = 1500
    T_alpha    = 2000
    T_stage2   = 2000
    T_final    = 3000

    # Shared TARGET test set size
    Ntest = 2000

    # ----------------------------
    # Load ClimSim data
    # ----------------------------
    root_path = r'H:\My Drive\LEAP_NP_Project\tlnp_supplementary_code\data\climsim_data'
    if root_path:
        print(f"[info] Using climsim data root: {root_path}")
    else:
        print("[info] Using climsim data root: (loader default)")

    climsim_config = {
        "data_frequency": "daily",
        "data_mode": "cluster_4",      # filter column
        "targets": [26],               # TARGET split ids
        "sources": [27],               # SOURCE split ids

        "num_target_normal_training":  n0T,
        "num_target_abnormal_training": n1T,
        "num_source_normal":           n0S,
        "num_source_abnormal":         n1S,

        "num_target_normal_test":  Ntest,
        "num_target_abnormal_test": Ntest,

        "input_dim": d
    }

    climsim_data = load_climsim_data(root_path=root_path)
    x0T, x1T, x0S, x1S, x0T_test, x1T_test = prepare_climsim_data(
        climsim_data=climsim_data,
        config=climsim_config,
        seed=0
    )

    # ----------------------------
    # Standardize using TRAIN stats
    # ----------------------------
    standardize, mean_, std_ = compute_standardizer(x0T, x1T, x0S, x1S)
    x0T, x1T, x0S, x1S = map(standardize, (x0T, x1T, x0S, x1S))
    x0T_test = (x0T_test.float() - mean_) / std_
    x1T_test = (x1T_test.float() - mean_) / std_

    # Basic sanity checks
    assert x0T.shape[1] == d == x1T.shape[1] == x0S.shape[1] == x1S.shape[1], \
        f"Feature dims mismatch with d={d}: " \
        f"{x0T.shape}, {x1T.shape}, {x0S.shape}, {x1S.shape}"

    # ----------------------------
    # 1) Learn α′ and θ* at α − ε0T_minus
    # ----------------------------
    meta = optimize_alpha_prime(
        alpha=alpha,
        x0T=x0T, x1T=x1T, x0S=x0S,
        model_type=model_type, hidden_sizes=hidden_sizes,
        activation=activation, init=init,
        use_layernorm=True, init_final_bias=0.5,
        k_sur=k_sur,
        c_eps0T_minus=c_eps0T_minus, c_eps0T_plus=c_eps0T_plus,
        c_eps0S=c_eps0S, c_eps1T=c_eps1T,
        T_outer=T_alpha,
        eta_theta=0.01, eta_alpha=0.02,
        eta_lambda1=5.0, eta_lambda2=5.0, eta_lambda3=5.0,
        T_inner_np=T_inner_np
    )
    hat_aprime = meta["alpha_prime"]

    # ε1S used only for decision rule
    n1S_actual = x1S.shape[0]
    eps1S = eps_from_n(n1S_actual, c_eps1S)

    print(f">>> learned hat(alpha') = {hat_aprime:.4f}")
    print(f"    eps0T_minus={meta['eps0T_minus']:.6f}, eps0T_plus={meta['eps0T_plus']:.6f}, "
          f"eps0S={meta['eps0S']:.6f}, eps1T={meta['eps1T']:.6f}, eps1S(decision)={eps1S:.6f}")

    # Baseline θ* (tightened) test perf
    _ = evaluate_target_test(meta["theta_star_model"], meta["train"]["k_sur"],
                             x0T_test, x1T_test, "TARGET TEST for θ* (α - ε0T_minus floored)")

    # ----------------------------
    # 2) θ_{α′,T} with feasibility projection; get R*_{1,T}(H(α′)) on TRAIN
    # ----------------------------
    theta_alphaT_model, R1T_star_Halph = solve_min_R1_given_alpha_prime(
        alpha_prime_hat=hat_aprime, meta=meta,
        T=T_stage2, eta_theta=0.01,
        eta_lambda1=5.0, eta_lambda2=5.0, eta_lambda3=5.0,
        proj_steps=2000, proj_lr=0.01
    )
    metrics_alphaT = evaluate_target_test(theta_alphaT_model, meta["train"]["k_sur"],
                                          x0T_test, x1T_test, "TARGET TEST for θ_{α′,T}")

    # ----------------------------
    # 3) Compute R1,S* over H(α′) (value only) on TRAIN splits
    # ----------------------------
    R1S_star_val = compute_R1S_star_over_Halpha(
        meta_alpha=meta, alpha_prime_hat=hat_aprime, x1S=x1S,
        T=1500, eta_theta=0.01,
        eta_lambda1=5.0, eta_lambda2=5.0, eta_lambda3=5.0
    )

    # ----------------------------
    # 4) Final stage → θ̂ with feasibility projection
    # ----------------------------
    theta_hat, R1S_hat_theta, _ = solve_min_R1S_final(
        meta_alpha=meta, R1T_star_Halph=R1T_star_Halph,
        x1S=x1S,
        T=T_final, eta_theta=0.01, eta_lambda=5.0,
        lam_cap=5.0, init_model=theta_alphaT_model, d=d,
        proj_steps=3000, proj_lr=0.01
    )
    metrics_theta_hat = evaluate_target_test(theta_hat, meta["train"]["k_sur"],
                                             x0T_test, x1T_test, "TARGET TEST for θ̂ (final stage)")

    # ----------------------------
    # 5) Decision rule between θ_{α′,T} and θ̂ using ε1,S
    # ----------------------------
    gap = R1S_hat_theta - R1S_star_val
    use_alphaT = (gap > eps1S)
    chosen_model = theta_alphaT_model if use_alphaT else theta_hat
    chosen_label = "θ_{α′,T}" if use_alphaT else "θ̂ (final)"

    print(f"\n[Decision] R1S(θ̂) - R1S*(H(α′)) = {gap:.6f}  vs  ε1,S = {eps1S:.6f}")
    print(f"[Decision] Selected model: {chosen_label}")

    # ----------------------------
    # 6) TARGET test for SELECTED MODEL (same test set)
    # ----------------------------
    selected_metrics = metrics_alphaT if use_alphaT else metrics_theta_hat
    print(f"\n=== TARGET TEST for SELECTED MODEL (same test set) ===")
    print(f"Surrogate: R0_T={selected_metrics['R0T_sur']:.4f}, R1_T={selected_metrics['R1T_sur']:.4f}")
    print(f"0–1 error: Type-I_T={selected_metrics['typeI']:.4f}, Type-II_T={selected_metrics['typeII']:.4f}")

if __name__ == "__main__":
    main()

