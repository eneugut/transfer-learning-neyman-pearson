from new_code import *

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # ------ Data controls (multidimensional Gaussians) ------
    d = 10
    # TARGET (separate variances):
    t0_T, c0_T = 0,   1.0   # class-0 mean/variance
    t1_T, c1_T = 1,   2.0   # class-1 mean/variance
    # SOURCE (separate variances):
    t0_S, c0_S = -0.2, 1.0  # class-0
    t1_S, c1_S =  1.0, 2.0  # class-1

    # Sample sizes
    n0T, n1T = 30, 30
    n0S, n1S = 1000, 1000

    # Model choice
    model_type   = "mlp"   # "linear" or "mlp"
    hidden_sizes = (16, 16)
    activation   = "relu"
    init         = "xavier_small"

    # NP + optimization knobs
    alpha   = 0.10
    k_sur   = 3.0          # surrogate temperature

    # ---- Separate epsilon constants (NOW with +/- for 0T) ----
    c_eps0T_minus = 0.5   # for ε0T_minus used in α - ε0T_minus
    c_eps0T_plus  = 0.03   # for ε0T_plus  used in α + ε0T_plus constraints
    c_eps0S       = 5.0   # for ε0S
    c_eps1T       = 1.0   # for ε1T
    c_eps1S       = 1   # for ε1S (decision rule)

    T_inner_np = 1500
    T_alpha    = 2000
    T_stage2   = 2000
    T_final    = 3000

    # One shared TARGET test set (uses separate variances)
    Ntest = 2000
    x0T_test, x1T_test = make_pair_sep(Ntest, Ntest, d, t0_T, c0_T, t1_T, c1_T, seed=123)

    x0T, x1T = make_pair_sep(n0T, n1T, d, t0_T, c0_T, t1_T, c1_T)
    x0S = make_gaussian(n0S, d, t0_S, c0_S)
    x1S = make_gaussian(n1S, d, t1_S, c1_S)

    # 1) Learn α′ and θ* at α − ε0T_minus
    meta = optimize_alpha_prime(
        alpha=alpha,
        x0T=x0T, x1T=x1T, x0S=x0S,
        model_type=model_type, hidden_sizes=hidden_sizes,
        activation=activation, init=init,
        k_sur=k_sur,
        c_eps0T_minus=c_eps0T_minus, c_eps0T_plus=c_eps0T_plus,
        c_eps0S=c_eps0S, c_eps1T=c_eps1T,
        T_outer=T_alpha,
        eta_theta=0.01, eta_alpha=0.02,
        eta_lambda1=5.0, eta_lambda2=5.0, eta_lambda3=5.0,
        T_inner_np=T_inner_np
    )
    hat_aprime = meta["alpha_prime"]
    eps1S = eps_from_n(n1S, c_eps1S)  # separate control for ε1S

    print(f">>> learned hat(alpha') = {hat_aprime:.4f}")
    print(f"    eps0T_minus={meta['eps0T_minus']:.4f}, eps0T_plus={meta['eps0T_plus']:.4f}")

    # Baseline θ* (tightened) test perf
    _ = evaluate_target_test(meta["theta_star_model"], meta["train"]["k_sur"],
                             x0T_test, x1T_test, "TARGET TEST for θ* (α - ε0T_minus)")

    # 2) θ_{α′,T} with feasibility projection; get R*_{1,T}(H(α′)) on TRAIN
    theta_alphaT_model, R1T_star_Halph = solve_min_R1_given_alpha_prime(
        alpha_prime_hat=hat_aprime, meta=meta,
        T=T_stage2, eta_theta=0.01,
        eta_lambda1=5.0, eta_lambda2=5.0, eta_lambda3=5.0,
        proj_steps=2000, proj_lr=0.01
    )
    metrics_alphaT = evaluate_target_test(theta_alphaT_model, meta["train"]["k_sur"],
                                          x0T_test, x1T_test, "TARGET TEST for θ_{α′,T}")

    # 3) Compute R1,S* over H(α′) (value only) on same TRAIN splits
    R1S_star_val = compute_R1S_star_over_Halpha(
        meta_alpha=meta, alpha_prime_hat=hat_aprime, x1S=x1S,
        T=1500, eta_theta=0.01,
        eta_lambda1=5.0, eta_lambda2=5.0, eta_lambda3=5.0
    )

    # 4) Final stage → θ̂ with feasibility projection
    theta_hat, R1S_hat_theta, _ = solve_min_R1S_final(
        meta_alpha=meta, R1T_star_Halph=R1T_star_Halph,
        x1S=x1S,
        T=T_final, eta_theta=0.01, eta_lambda=5.0,
        lam_cap=None, init_model=theta_alphaT_model,
        d=d, proj_steps=3000, proj_lr=0.01
    )
    metrics_theta_hat = evaluate_target_test(theta_hat, meta["train"]["k_sur"],
                                             x0T_test, x1T_test, "TARGET TEST for θ̂ (final stage)")

    # 5) Decision rule (between θ_{α′,T} and θ̂), with separate ε1,S
    gap = R1S_hat_theta - R1S_star_val
    use_alphaT = (gap > eps1S)
    chosen_model = theta_alphaT_model if use_alphaT else theta_hat
    chosen_label = "θ_{α′,T}" if use_alphaT else "θ̂ (final)"

    print(f"\n[Decision] R1S(θ̂) - R1S*(H(α′)) = {gap:.6f}  vs  ε1,S = {eps1S:.6f}")
    print(f"[Decision] Selected model: {chosen_label}")

    # 6) TARGET test for SELECTED MODEL (same test set)
    selected_metrics = metrics_alphaT if use_alphaT else metrics_theta_hat
    print(f"\n=== TARGET TEST for SELECTED MODEL (same test set) ===")
    print(f"Surrogate: R0_T={selected_metrics['R0T_sur']:.4f}, "
          f"R1_T={selected_metrics['R1T_sur']:.4f}")
    print(f"0–1 error: Type-I_T={selected_metrics['typeI']:.4f}, "
          f"Type-II_T={selected_metrics['typeII']:.4f}")

