🌫️ Diffusion & Flow Models — Minimal, Clean Implementations

This repository contains a set of mathematically transparent mini-projects exploring modern generative models, including diffusion processes, ODE-based solvers, and flow matching. The goal is to understand how generation-by-noising works from first principles — by building each model step-by-step with clear visualizations and simple 2D datasets.

🧩 Project Topics

Each notebook focuses on a core idea in diffusion or flow-based generative modeling:

🌪️ DDPM — Denoising Diffusion Probabilistic Models
Forward diffusion, reverse denoising, noise-matching loss, sampling.

🔁 DDIM — Deterministic Sampling (η = 0)
ODE interpretation, exact 
𝑥
0
x
0
	​

 reconstruction, fast non-stochastic trajectories.

⚡ DPM-Solver (1st & 2nd Order)
ODE sampling via exponential integrators, λ-space stepping, midpoint correction.

🧷 Flow Matching (FM)
Neural ODE viewpoint, velocity-field learning, Gaussian flow, continuous-time generation.

🌀 Toy Dataset Transformations
Spiral → Gaussian, checkerboard → noise, and reverse sampling demonstrations.

📂 Repository Structure

Each notebook is self-contained, with runnable code, equations, and visualizations.

Notebook List
Notebook	Description
DDPM_DDIM.ipynb	Implements DDPM training + DDIM deterministic sampling on 2D datasets
DPM_Solver.ipynb	Full DPM-Solver implementation (α, σ, λ schedules, 1st/2nd order solvers)
flow_matching.ipynb	Gaussian flow matching, velocity networks, ODE-based generative mapping
🔍 Key Concepts Covered

Forward Process

𝑥
𝑡
=
𝛼
𝑡
𝑥
0
+
𝜎
𝑡
𝜀
x
t
	​

=α
t
	​

x
0
	​

+σ
t
	​

ε
Visualization of 
𝑞
(
𝑥
𝑡
)
q(x
t
	​

) across timesteps.

Noise Prediction Objective

𝐿
=
𝐸
∥
𝜀
𝜃
(
𝑥
𝑡
,
𝑡
)
−
𝜀
∥
2
L=E∥ε
θ
	​

(x
t
	​

,t)−ε∥
2
.

Sampling ODE

𝑥
˙
𝑡
=
𝑓
(
𝑡
)
𝑥
𝑡
+
𝑔
(
𝑡
)
2
∇
𝑥
log
⁡
𝑝
𝑡
(
𝑥
)
x
˙
t
	​

=f(t)x
t
	​

+g(t)
2
∇
x
	​

logp
t
	​

(x).

DPM-Solver Midpoint Rule
Predictor–corrector update using 
𝜆
=
log
⁡
(
𝛼
/
𝜎
)
λ=log(α/σ).

Flow Matching
Learn 
𝑣
(
𝑥
𝑡
,
𝑡
)
v(x
t
	​

,t) instead of noise; generate via continuous ODE integration.

📊 Visual Demos Included

Evolution of swiss-roll → Gaussian under diffusion

Reverse sampling via DDIM & DPM-Solver

Score fields and velocity fields

𝑥
0
x
0
	​

-prediction convergence

Trajectory plots in 2D

🧪 Requirements

Install dependencies:

pip install torch numpy matplotlib tqdm scikit-learn


Run notebooks:

pip install notebook
jupyter notebook

🛠️ Roadmap

 Add DPM-Solver-3

 Add MNIST / CIFAR-10 implementations

 Add Consistency Models

 Create unified /src modules

 Compare Flow Matching vs Diffusion on same datasets

📚 References

Ho et al. (2020) — Denoising Diffusion Probabilistic Models

Song et al. (2020) — Denoising Diffusion Implicit Models

Lu et al. (2022) — DPM-Solver: Fast ODE Solvers for Diffusion Models

Lipman et al. (2023) — Flow Matching for Generative Modeling
