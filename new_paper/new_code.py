
# ------------------------------------------------------------
# Neyman–Pearson (Target + Source) with decision rule
# Multidimensional Gaussians + switchable model (Linear/MLP)
# Robust SGDA + projection with temperatured logistic surrogate (k_sur).
#
# NEW: Separate variances for each class (target & source) and
#      SEPARATE EPSILON CONSTANTS (now with two epsilons for target class-0):
#         ε0T_minus = c_eps0T_minus / sqrt(n0T)   # used in α - ε0T_minus (tighten)
#         ε0T_plus  = c_eps0T_plus  / sqrt(n0T)   # used in α + ε0T_plus   (constraints)
#         ε0S       = c_eps0S       / sqrt(n0S)
#         ε1T       = c_eps1T       / sqrt(n1T)
#         ε1S       = c_eps1S       / sqrt(n1S)   (decision rule only)
# ------------------------------------------------------------

import math
import torch
from torch import nn
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

def build_model(d, model_type="linear", hidden_sizes=(32, 32), activation="relu", init="xavier_small"):
    act = get_activation(activation)
    if model_type == "mlp":
        layers, in_dim = [], d
        for h in (hidden_sizes or []):
            lin = nn.Linear(in_dim, int(h))
            if init == "xavier_small": xavier_small_(lin, gain=0.1)
            layers += [lin, act]
            in_dim = int(h)
        out = nn.Linear(in_dim, 1)
        if init == "xavier_small": xavier_small_(out, gain=0.1)
        layers += [out]
        return nn.Sequential(*layers)
    else:
        lin = nn.Linear(d, 1)
        if init == "xavier_small": xavier_small_(lin, gain=0.1)
        return nn.Sequential(lin)

def make_model_factory(d, model_type="linear", hidden_sizes=(32, 32), activation="relu", init="xavier_small"):
    def factory():
        return build_model(d, model_type, hidden_sizes, activation, init)
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
    # Constraints use α + ε0T_plus
    g1 = R0_T - alpha - eps0T_plus - delta
    g2 = R0_S - alpha_prime_hat - eps0S
    g3 = R1_T - R1_star - eps1T
    return g1, g2, g3

def project_to_feasible_max(model, compute_gmax, steps=2500, lr=0.01, tol=1e-7):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
# Base NP on TARGET
# =========================

def train_np_sigmoid(alpha, k_sur, T, eta_theta, eta_lambda, delta, gamma, x0, x1, model_factory):
    model = model_factory()
    def R0_hat(): return phi_pos(model(x0), k_sur).mean()
    def R1_hat(): return phi_neg(model(x1), k_sur).mean()
    def f(): return R1_hat()
    def g(): return R0_hat() - alpha - delta
    opt = torch.optim.Adam(model.parameters(), lr=eta_theta)
    lam = 0.0
    for _ in range(T):
        opt.zero_grad(set_to_none=True)
        L = f() + lam * g()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        lam = max((1 - gamma * eta_lambda) * lam + eta_lambda * g().detach().item(), 0.0)
    with torch.no_grad():
        R0_tr = R0_hat().item(); R1_tr = R1_hat().item()
        typeI_tr = (model(x0) > 0).float().mean().item()
        typeII_tr = (model(x1) <= 0).float().mean().item()
    return model, R0_tr, R1_tr, typeI_tr, typeII_tr

def solve_theta_star_T_tight(x0T, x1T, alpha, eps0T_minus, k_sur, T_inner,
                             eta_theta, eta_lambda, delta, gamma, model_factory):
    # Tighten with α - ε0T_minus
    alpha_tight = max(alpha - eps0T_minus, 0.0)
    model, _, R1_star, *_ = train_np_sigmoid(
        alpha=alpha_tight, k_sur=k_sur, T=T_inner,
        eta_theta=eta_theta, eta_lambda=eta_lambda, delta=delta, gamma=gamma,
        x0=x0T, x1=x1T, model_factory=model_factory
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

    model_factory = make_model_factory(d, model_type, hidden_sizes, activation, init)

    # θ* with tightened α - ε0T_minus
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
    opt_theta = torch.optim.Adam(model.parameters(), lr=eta_theta)
    alpha_prime_avg = 0.0

    for t in range(1, T_outer + 1):
        opt_theta.zero_grad(set_to_none=True)
        if alpha_prime.grad is not None: alpha_prime.grad.zero_()
        # g1 uses α + ε0T_plus
        g1 = R0_T_hat() - alpha - eps0T_plus - delta
        g2 = R0_S_hat() - alpha_prime - eps0S
        g3 = R1_T_hat() - R1_star - eps1T
        (alpha_prime + lam1*g1 + lam2*g2 + lam3*g3).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt_theta.step()
        with torch.no_grad():
            alpha_prime -= eta_alpha * (1.0 - lam2); alpha_prime.clamp_(min=alpha)
            alpha_prime_avg += (alpha_prime.item() - alpha_prime_avg) / t
            lam1 = max((1 - gamma1*eta_lambda1)*lam1 + eta_lambda1*g1.detach().item(), 0.0)
            lam2 = max((1 - gamma2*eta_lambda2)*lam2 + eta_lambda2*g2.detach().item(), 0.0)
            lam3 = max((1 - gamma3*eta_lambda3)*lam3 + eta_lambda3*g3.detach().item(), 0.0)

    return {
        "alpha_prime": alpha_prime_avg,
        "eps0T_minus": eps0T_minus, "eps0T_plus": eps0T_plus,
        "eps0S": eps0S, "eps1T": eps1T,
        "R1_star_train": R1_star,
        "theta_star_model": theta_star_model,
        "train": {
            "x0T": x0T, "x1T": x1T, "x0S": x0S,
            "alpha": alpha, "k_sur": k_sur, "delta": delta,
            "d": d, "model_cfg": dict(model_type=model_type, hidden_sizes=hidden_sizes,
                                      activation=activation, init=init)
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
    opt_theta = torch.optim.Adam(model.parameters(), lr=eta_theta)

    def R0_T(m): return phi_pos(m(x0T), k_sur).mean()
    def R0_S(m): return phi_pos(m(x0S), k_sur).mean()
    def R1_T(m): return phi_neg(m(x1T), k_sur).mean()

    lam1 = lam2 = lam3 = 0.0
    for t in range(1, T + 1):
        opt_theta.zero_grad(set_to_none=True)
        # g1 uses α + ε0T_plus
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
    alpha = meta_alpha["train"]["alpha"]
    eps0T_plus = meta_alpha["eps0T_plus"]; eps0S = meta_alpha["eps0S"]; eps1T = meta_alpha["eps1T"]
    R1_star = meta_alpha["R1_star_train"]
    cfg = meta_alpha["train"]["model_cfg"]; d = meta_alpha["train"]["d"]
    model_factory = make_model_factory(d, **cfg)

    model = model_factory()
    opt = torch.optim.Adam(model.parameters(), lr=eta_theta)

    def R0_T(m): return phi_pos(m(x0T), k_sur).mean()
    def R0_S(m): return phi_pos(m(x0S), k_sur).mean()
    def R1_T(m): return phi_neg(m(x1T), k_sur).mean()
    def R1_S(m): return phi_neg(m(x1S), k_sur).mean()

    lam1 = lam2 = lam3 = 0.0
    for _ in range(T):
        opt.zero_grad(set_to_none=True)
        # g1 uses α + ε0T_plus
        g1 = R0_T(model) - alpha - eps0T_plus
        g2 = R0_S(model) - alpha_prime_hat - eps0S
        g3 = R1_T(model) - R1_star - eps1T
        (R1_S(model) + lam1*g1 + lam2*g2 + lam3*g3).backward()
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
    opt = torch.optim.Adam(model.parameters(), lr=eta_theta)
    lam = 0.0

    def g_max(m):
        # g1 uses α + ε0T_plus
        g1 = R0_T(m) - (alpha + eps0T_plus)
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
            if lam_cap is not None: lam = min(max(lam, 0.0), lam_cap)
            else:                    lam = max(lam, 0.0)

    project_to_feasible_max(avg_model, g_max, steps=proj_steps, lr=proj_lr)

    with torch.no_grad():
        R1S_train = R1_S(avg_model).item()
        g_after = g_max(avg_model).item()
        print(f"[Final projection] max(g1,g2,g3,g4)={g_after:.4f}")

    return avg_model, R1S_train, x1S

