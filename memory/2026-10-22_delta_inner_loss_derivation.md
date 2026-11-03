# Δ inner-loss meta-learner: derivation

Implementation: `src/gd_optimic/meta_learner/bilevel_optimizer.py` (`_delta_inner_step`, `delta_chain`).

Notation: the forward model is **G(ψ)**, and the rollout has **M** steps, m = 0…M−1.
In the code, M is `delta_inner_k`, and the code comments call it K.

Everything is indexed from the point of view of **outer iteration k**, as in the code.
At iteration k, the inner loop trains on the **previous** move (k−1 → k), and then the
iteration makes the **new** move (k → k+1).

---

## 0. Objects and notation

| symbol | meaning | in code |
|---|---|---|
| ψ ∈ ℝᴺ | initial condition, flattened from [B,T,C,H,W] into one vector of N numbers | `x` |
| G(ψ) | forward model (GloNet), run for `num_forecast_steps` | `forward_model.forward` |
| J(ψ) = ℓ(G(ψ), y) | cost of an IC against the target sequence | `loss_fn` |
| g_k = ∇J(ψ^k) ∈ ℝᴺ | outer gradient at IC k | `gradient` |
| S_θ(ψ, g, k) ∈ ℝᴺ | network output: one proposed step | `one_task_update(...)[0]` |
| θ ∈ ℝᴾ | weights of S | `network_s.parameters()` |
| k | outer iteration | `iteration` |
| l = 0…L−1 | meta-step (θ update) | loop `for m in range(num_meta_steps)` |
| m = 0…M−1 | rollout step | loop inside `delta_chain` (M = `delta_inner_k`) |

**Why flatten into vectors?** Every derivative below is then an ordinary matrix: a gradient is
in ℝᴺ, a Jacobian is N×N or N×P. The tensor shape [B,T,C,H,W] is only a way of storing the
N numbers, so flattening loses nothing.

---

## 1. Outer loop: what we are optimizing

The goal is

```
ψ* = argmin_ψ J(ψ),        J(ψ) = ℓ( G(ψ), y )
```

At outer iteration k we pay **one forward + adjoint pass through G** to get

```
g_k = ∇J(ψ^k) = ( ∂G/∂ψ |_{ψ^k} )ᵀ · ∇ℓ( G(ψ^k), y )                (1)
```

This is the only expensive operation in the method, and everything else is designed around it.
(Equation (1) has the same transpose structure as section 3: an adjoint pass is a
vector–Jacobian product.)

The baseline is plain GD: ψ^{k+1} = ψ^k − η g_k. That is **one step per expensive gradient**.

> **Justification for adding S:** we want more than one step's worth of progress from each g_k.

---

## 2. The move: M-step rollout with a frozen gradient

Given g_k and the weights θ^k at the end of iteration k's inner loop, the code applies S M times:

```
ψ^k_0     = ψ^k
ψ^k_{m+1} = ψ^k_m − S_θ(ψ^k_m, g_k, k)        m = 0 … M−1       (2)
ψ^{k+1}   = ψ^k_M(θ^k)
```

- **g_k is frozen for all m.** Recomputing it would cost a pass through G per m, which defeats
  the purpose.
- **S sees ψ^k_m** (`hybrid_input: true`), so each step can react to where the previous steps
  landed. Without this, all M steps would be identical, and (2) would reduce to ψ^k − M·S(g_k).
- **G is not called in (2).** The code runs one forward pass of G afterwards, but only to report
  J(ψ^{k+1}).

This is the move we will need to train θ for. The question is how to compute its gradient.

---

## 3. What θ should minimize, and the exact gradient

The ideal objective for θ is the cost after the move:

```
Φ(θ) = J( ψ_M(θ) )                                              (3)
```

By the chain rule:

```
dΦ/dθ = ( ∂ψ_M/∂θ )ᵀ · ∇J( ψ_M(θ) )                             (4)
          P×N           N×1        →  P×1
```

### Why the transpose?

- ∂ψ_M/∂θ is the Jacobian. Its entry (a, b) is ∂(ψ_M)_a / ∂θ_b, so it has one row per IC value
  and one column per weight: **N×P**.
- ∇J is a column in ℝᴺ.
- We want dΦ/dθ as a column in ℝᴾ, one number per weight.

The scalar chain rule for one weight b is

```
dΦ/dθ_b = Σ_a  (∂J/∂ψ_a) · (∂ψ_a/∂θ_b)
```

This sums over the **row** index a of the Jacobian. In matrix form, "sum over rows, one result
per column" is exactly Jacobianᵀ · vector. So the transpose is simply what the index sum looks
like when you write it with matrices.

You can also write (4) as a row vector: dΦ/dθ = ∇Jᵀ · (∂ψ_M/∂θ), which is (1×N)(N×P) = 1×P.
The two forms say the same thing. The transpose appears only because we choose to store
gradients as columns.

In code you never see a transpose. Autograd computes "vector × Jacobian" products (VJPs)
directly, and the inner product ⟨g, v⟩ = gᵀv is written `(g * v).sum()`.

### The problem with (4)

(4) needs ∇J(ψ_M(θ)), the gradient at wherever the rollout lands for the **current** θ. That
costs a forward + adjoint pass through G **for every meta-step l**. This is exactly what the
old meta loop did (G is called in every meta-step). The delta mode removes it with two moves,
described in sections 4 and 5.

---

## 4. First move: delay, so the judge's gradient is free

At outer iteration k+1 we compute g_{k+1} = ∇J(ψ^{k+1}) anyway, since we need it to make the
next move. That is the true gradient at the point where move k actually landed.

So training on move k is postponed to iteration k+1. Re-indexed from the current iteration's
point of view:

> **At outer iteration k, the inner loop trains θ on the previous move (k−1 → k), and the judge
> is g_k.**

This is why the code stores `_chain_prev = (ψ^{k-1}, g_{k-1}, stats, k−1, ‖g_{k-1}‖², step)`.
With these, the previous move can be **replayed** with any θ.

**Is the delay an approximation?** Not for the math, but it does rest on one assumption. θ is
trained on move k−1 and then used for move k, so this only helps if consecutive moves are
similar (ψ^k ≈ ψ^{k-1}, g_k ≈ g_{k-1}). That is a smoothness assumption about the optimization
path. It isn't a Taylor expansion, but it's the reason why "learning from the last move" should
generalize to the next one.

---

## 5. Second move: the Taylor expansion (the main approximation)

At outer iteration k, meta-step l, replay the previous move with the current θ_l:

```
ψ^{k-1}_0       = ψ^{k-1}
ψ^{k-1}_{m+1}   = ψ^{k-1}_m − S_{θ_l}(ψ^{k-1}_m, g_{k-1}, k−1)       m = 0 … M−1     (5)
```

It lands at ψ_M(θ_l). If θ_l were still the θ that made the move, this would be exactly ψ^k.

The exact objective (3) needs J(ψ_M(θ_l)), which would require a new pass through G. Instead,
take a **first-order Taylor expansion of J around ψ^k**, the point where we already know the
gradient:

```
J( ψ_M(θ) ) ≈ J(ψ^k) + g_kᵀ ( ψ_M(θ) − ψ^k )                     (6)
```

Terms that don't depend on θ are J(ψ^k) and g_kᵀψ^k. Drop them, and the θ-dependent part is just

```
J̃(θ) = g_kᵀ ψ_M(θ)          (+ constants)                        (7)
```

### From (7) to `l_perf`

Add the constant −g_kᵀψ^{k-1} and divide by a positive constant. Neither changes the minimizer:

```
l_perf(θ) = − g_kᵀ ( ψ^{k-1} − ψ_M(θ) )  /  ( M·η·‖g_{k-1}‖² )    (8)
```

The normalization makes it unitless. If the move equals M plain GD steps
(ψ^{k-1} − ψ_M = Mη g_{k-1}) and the gradient didn't change (g_k = g_{k-1}), then l_perf = −1.

### How large is the Taylor error?

Expand the gradient one order further:

```
∇J( ψ_M(θ_l) ) = g_k + H ( ψ_M(θ_l) − ψ^k ) + O(‖·‖²)            (9)
```

H is the Hessian of J at ψ^k. Replacing ∇J(ψ_M(θ_l)) by g_k in (4) therefore drops the term

```
( ∂ψ_M/∂θ )ᵀ · H · ( ψ_M(θ_l) − ψ^k )                           (10)
```

| case | drift ψ_M(θ_l) − ψ^k | error (10) |
|---|---|---|
| first `l_perf` step, `grad_align_steps = 0` | 0, since θ_l = θ that made the move | **0 (exact)** |
| after ALIGN steps or earlier PERF steps | grows as θ moves away | ∝ ‖H‖ · drift |

That is the only place the Taylor approximation enters the training gradient. The code's comment
"exactly dJ/dθ only while θ is unchanged" is case 1 of this table.

---

## 6. The rollout derivative ∂ψ_M/∂θ (exact, no approximation)

Write D_m = ∂ψ_m/∂θ (N×P). At step m, define

```
A_m = ∂S/∂ψ  at (ψ_m, g_{k-1}, θ_l)         N×N
B_m = ∂S/∂θ  at (ψ_m, g_{k-1}, θ_l)         N×P
```

Differentiate (5). S depends on θ both directly and through ψ_m:

```
D_{m+1} = D_m − ( A_m D_m + B_m ) = (I − A_m) D_m − B_m          (11)
D_0     = 0          (ψ^{k-1} is stored data, it does not depend on θ)
```

Unrolling (11) is solving a linear recurrence:

```
D_M = (I−A_{M-1})…(I−A_0) · D_0                    ← vanishes since D_0 = 0
      − Σ_{m=0}^{M-1} (I−A_{M-1})…(I−A_{m+1}) · B_m                (12)
```

The term m = M−1 has an empty product (= I) and contributes −B_{M-1}.

Example, M = 3, without dropping anything:

```
D₃ = (I−A₂) D₂ − B₂
   = (I−A₂)(I−A₁) D₁ − (I−A₂) B₁ − B₂
   = (I−A₂)(I−A₁)(I−A₀) D₀ − (I−A₂)(I−A₁) B₀ − (I−A₂) B₁ − B₂
```

The intermediate D₂, D₁ are not zero, but each one becomes B terms as you unroll. Only the D₀
term vanishes.

**Why this is exact.** g_{k-1} is detached data, A_m and B_m are evaluated at the actual
replayed ψ_m(θ_l), and gradient checkpointing only recomputes S during backward, so the values
are identical. No Taylor expansion appears here.

### Reading the propagator (I − A_m)

(I − A_m) is the Jacobian of one rollout step F(ψ) = ψ − S(ψ). It carries a perturbation from
before step m to after it. In one eigen-direction of A with eigenvalue a, each step multiplies
the perturbation by (1 − a):

| a | factor (1−a) | effect over the remaining steps | GD analogue |
|---|---|---|---|
| a ≈ 0 | ≈ 1 | carried through unchanged | flat direction |
| 0 < a < 1 | between 0 and 1 | shrinks: later steps correct earlier ones | stable step |
| 1 < a < 2 | between −1 and 0 | sign flips, oscillates but decays | over-relaxed, stable |
| a > 2 | < −1 | oscillates and grows | divergent, η > 2/λ |
| a < 0 | > 1 | grows, the move reinforces itself | negative curvature or momentum |

- **Contractive** chain: early steps get little gradient (vanishing-gradient regime).
- **Expanding** chain: early steps dominate and the gradient can blow up (`clip_grad_norm_`
  limits this).
- **A = 0** (S ignores ψ): D_M = −M·B, so the rollout is just one step scaled by M.

For comparison, the Jacobian of one real GD step ψ − η∇J(ψ) is I − ηH (the derivative of the
gradient is the Hessian). So A plays the role of ηH.

---

## 7. Putting (4), (7) and (12) together: the `l_perf` gradient, and why backward goes in reverse

Let c = 1/(M·η·‖g_{k-1}‖²). From (7), (8) and (12):

```
∇_θ l_perf = c · D_Mᵀ g_k
           = −c · Σ_m  B_mᵀ (I−A_{m+1})ᵀ … (I−A_{M-1})ᵀ g_k        (13)
```

This is the second place transposes show up. Transposing a product **reverses its order**:
(XY)ᵀ = YᵀXᵀ. In (13), g_k is multiplied first by the propagator of the **last** step,
(I−A_{M-1})ᵀ, and the earliest step comes last. So the cheap way to compute it is to start at
the end and walk backwards, carrying a single N-vector λ:

```
λ_M = c · g_k                                                     (14)
for m = M−1 down to 0:
    ∇_θ += −B_mᵀ λ_{m+1}          (weights used at step m get their share)
    λ_m  = (I − A_m)ᵀ λ_{m+1}     (= λ_{m+1} − A_mᵀ λ_{m+1}, pass to the earlier step)
```

Each line is one VJP: `l_meta.backward()` through the checkpointed `_chain_update`. No N×N or
N×P matrix is ever built, only N-vectors, P-vectors and recomputed activations of S. This is
reverse-mode autodiff (backprop through time). It's efficient because the output is a scalar
(l_perf), so propagating one vector from the end is enough. The adjoint pass through G in (1)
works the same way.

---

## 8. The full inner loop at outer iteration k

For l = 0 … L−1:

```
l_align(θ_l) = 1 − cos( S_{θ_l}(ψ^k, g_k, k), g_k )   over ocean points        (15)
l_perf(θ_l)  = eq. (8), via replay (5)
l_reg(θ_l)   = λ_reg² · mean(θ_l²)

phase(l):  ALIGN  (l < L_a)              L = w_align·l_align                    + l_reg
           TRANS  (L_a ≤ l < L_a+L_t)    L = w_align·l_align + w_perf·l_perf    + l_reg
           PERF   (otherwise)            L =                   w_perf·l_perf    + l_reg

θ_{l+1} = Adam( θ_l, clip( ∇_θ L ) )
```

Justification for each piece:

- **`l_align` (15)** uses the current ψ^k and g_k with one application of S, so it involves no
  delay, no replay and no Taylor expansion. It only constrains the **direction** of S's step.
  It's cheap and stable, which is why it's used to bootstrap.
- **`l_perf`** sets the step **size**, which `l_align` cannot see because cosine ignores length.
- **Phases:** ALIGN first gives θ a sensible direction before `l_perf`'s step-size pressure is
  added. The cost is that ALIGN steps move θ away from the θ that made the move, so by the first
  `l_perf` step the drift in (10) is already nonzero. If `grad_align_steps > 0`, the exact case
  in section 5 never happens.
- **At k = 0** there is no previous move, so every step is ALIGN (`prev is None`).

Note: `l_reg` in the delta loop is `lambda_reg * (lambda_reg * Σp²/n)`, i.e. **λ²**. The original
loop uses λ once. If this isn't intentional, the regularization in delta mode is weaker than the
config value suggests by a factor of λ.

---

## 9. End of outer iteration k

```
ψ^{k+1}  = rollout (2) with θ^k = θ_L and g_k, no grad
J(ψ^{k+1}) = ℓ( G(ψ^{k+1}), y ) via one forward pass of G (reported only)
store (ψ^k, g_k, stats, k, ‖g_k‖², step = ψ^k − ψ^{k+1})   → judged at k+1
```

---

## 10. Full timeline

```
outer k−1:  g_{k-1} ── train θ on move k−2 ── make move k−1: ψ^{k-1} → ψ^k  (store)
                                                                │
outer k:    g_k = ∇J(ψ^k)   ◄── 1 forward+adjoint through G ───┘
              │
              ├─ l = 0…L−1:   replay move k−1 with θ_l, m = 0…M−1  (5)
              │               l_perf = Taylor around ψ^k with judge g_k  (6–8)
              │               backprop through the M rollout steps  (12–14)
              │               + l_align at (ψ^k, g_k)  (15)
              │               θ_{l+1} = Adam(...)
              │
              └─ make move k: ψ^k → ψ^{k+1} with θ_L and frozen g_k, m = 0…M−1  (2)   (store)
                                                                │
outer k+1:  g_{k+1} = ∇J(ψ^{k+1})  ◄────────────────────────────┘  judges move k ...
```

---

## 11. Where the approximations are

| # | where | what is assumed | exact when |
|---|---|---|---|
| **T1** | `l_perf`, eq. (6) | **Taylor expansion of J** around ψ^k, so the true gradient at the replay point is replaced by g_k | first `l_perf` step with no prior θ change; otherwise the error is ∝ H · drift, eq. (10) |
| **T2** | rollout design, eq. (2) | **Taylor expansion of ∇J** along the path: ∇J(ψ_m) ≈ g + H(ψ_m − ψ_0). Frozen g ignores the H term, so S can only make up for it through A_m (ideally A ≈ ηH) | this is **interpretation**: nothing in the loss requires it, but it explains why the ψ input matters |
| **D** | delay, section 4 | θ trained on move k−1 generalizes to move k | consecutive ψ and g are similar |
| — | rollout Jacobians, eqs. (11–14) | none | always exact (autograd + checkpointing) |
| — | `l_align` | none | always exact |

T2 is also what the ρ diagnostic measures. With the step s = ψ^{k-1} − ψ^k (the sum of the M
outputs of S), T2 gives g_k ≈ g_{k-1} − H s, so

```
ρ = g_kᵀ s / g_{k-1}ᵀ s ≈ 1 − sᵀ H s / g_{k-1}ᵀ s
```

- ρ ≈ 1: the curvature along s is negligible compared with the step, so the move was too short.
- ρ ≈ 0: the move consumed exactly the descent available along s. This is the line-search
  optimum for a quadratic.
- ρ < 0: the move overshot.

So ρ is the curvature along the actual move, as seen through the T2 expansion. That's why it's
a good companion to `l_perf`, whose own value is a T1 quantity.
