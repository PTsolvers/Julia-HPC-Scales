# Julia-HPC-Scales
Julia for HPC workshop @ [**SCALES conference 2023**](https://model.uni-mainz.de/scales-conference-2023/)

> :warning: Make sure to `git pull` this repo right before starting the workshop on Monday morning in order to ensure you have access to the latest updates

The computational resources for this workshop are provided by the “Paderborn Center for Parallel Computing (PC2)” [https://pc2.uni-paderborn.de/](https://pc2.uni-paderborn.de/) as part of the "NHR Alliance” [https://www.nhr-verein.de/](https://www.nhr-verein.de/).

## Program

### Morning session (9h - 12h30)
- [Noctua 2: VSCode on compute node (and troubleshooting)](#vscode-on-noctua-2)
- [Brief **intro to Julia for HPC** :book:](#julia-for-hpc)
  - Performance, CPUs, GPUs, array and kernel programming
- [Presentation of **the challenge of today** :book:](#the-challenge-of-today)
  - Optimising injection/extraction from a heterogeneous reservoir
- [**Hands-on I** - solving the forward problem :computer:](#hands-on-i)
  - Steady-state diffusion problem
  - The accelerated pseudo-transient method
  - From CPU to GPU array programming
  - Kernel programming (performance)
    - CPU "kernel" programming -> multi-threading
    - GPU kernel programming

### Afternoon session (13h30 - 17h00)
- [Presentation of **the optimisation problem** :book:](#the-optimisation-problem)
  - Tha adjoint method
  - Julia and the automatic differentiation (AD) tooling
- [**Hands-on II** - HPC GPU-based inversions :computer:](#hands-on-ii)
  - The adjoint problem using AD
  - GPU-based adjoint solver using [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)
  - Sensitivity analysis
  - Gradient-based inversion (Gradient descent - GD)
    - Vanilla GD by hand
    - Using [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
- [**Exercises** (optionnal) :computer:](#exercises-optionnal)
  - Go for 3D
  - Make combined loss (pressure + flux)
- **Wrapping up** & outlook :beer:

## The `SMALL` print
The goal of today's workshop is to develop a fast iterative GPU-based solver for elliptic equations and use it to:
1. Solve a steady state subsurface flow problem (geothermal operations, injection and extraction of fluids)
2. Invert for the subsurface permeability having a sparse array of fluid pressure observations

We will not use any "black-box" tooling but rather try to develop concise and performant codes (300 loc max) that execute on GPUs. We will also use automatic differentiation (AD) capabilities and the differentiable Julia stack to automatise the calculation of the adjoint solutions in the gradient-based inversion procedure.

The main Julia packages we will rely on are:
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU computing on Nvidia GPUs
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for AD on GPUs
- [CairoMakie.jl](https://github.com/MakieOrg/Makie.jl) for plotting
- [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to extend the vanilla gradient-descent procedure

Most of the workshop is based on "hands-on". Changes to the scripts are incremental and should allow to build up complexity throughout the day. Blanked-out scripts for most of the steps are available in the [scripts](scripts/) folder. Solutions scripts (following the `s_xxx.jl` pattern) will be shared at some point in the [scripts_solutions](scripts_solutions) folder.

#### :bulb: Useful extra resources
- The Julia language: [https://julialang.org](https://julialang.org)
- PDE on GPUs ETH Zurich course: [https://pde-on-gpu.vaw.ethz.ch](https://pde-on-gpu.vaw.ethz.ch)
- Julia Discourse (Julia Q&A): [https://discourse.julialang.org](https://discourse.julialang.org)
- Julia Slack (Julia dev chat): [https://julialang.org/slack/](https://julialang.org/slack/)

## VSCode on Noctua 2
Before we start, let's make sure that everyone can remote connect over SSH to a GPU node on the **Noctua 2** supercomputer we will use today.

Let's go back to the email titled "**SCALES workshop - login credentials**" in order to access the HackMD doc to go through the "**VSCode on the Compute Node (on Monday only)**" procedure.

If all went fine, you should be able to execute the following command in your Julia REPL:
```julia-repl
julia> include("visu_2D.jl")
```

which will produce this figure:

![out visu](docs/out_visu_2D.png)

## Julia for HPC

Some words on the Julia at scale effort, the Julia HPC packages, and the overall Julia for HPC motivation (two language barrier)

### The (yet invisible) cool stuff
Today, we will develop code that:
- Runs on graphics cards using the Julia language
- Uses a fully local and iterative approach (scalability)
- Retrieves automatically the Jacobian Vector Product (JVP) using automatic differentiation (AD)
- (All scripts feature about 300 lines of code)

Too good to be true? Hold on 🙂 ...

### Why to still bother with GPU computing in 2023
- It's around for more than a decade
- It shows massive performance gain compared to serial CPU computing
- First exascale supercomputer, Frontier, is full of GPUs
![Frontier](docs/frontier.png)

### Performance that matters
![cpu_gpu_evo](docs/cpu_gpu_evo.png)

Taking a look at a recent GPU and CPU:
- Nvidia Tesla A100 GPU
- AMD EPYC "Rome" 7282 (16 cores) CPU

| Device         | TFLOP/s (FP64) | Memory BW TB/s | Imbalance (FP64)     |
| :------------: | :------------: | :------------: | :------------------: |
| Tesla A100     | 9.7            | 1.55           | 9.7 / 1.55  × 8 = 50 |
| AMD EPYC 7282  | 0.7            | 0.085          | 0.7 / 0.085 × 8 = 66 |

**Meaning:** we can do about 50 floating point operations per number accessed from main memory.
Floating point operations are "for free" when we work in memory-bounded regimes.

👉 Requires to re-think the numerical implementation and solution strategies

Unfortunately, the cost of evaluating a first derivative $∂A / ∂x$ using finite-differences:
```julia
q[ix] = -D * (A[ix+1] - A[ix]) / dx
```
consists of:
- 1 read (`A`) + 1 write (`q`) => $2 × 8$ = **16 Bytes transferred**
- 2 (fused) addition and multiplication => **2 floating point operations**

👉 assuming $D$, $∂x$ are scalars, $q$ and $A$ are arrays of `Float64` (read from main memory)

### Performance that matters - an example
Not yet convinced? Let's have a look at an example.

Let's assess how close from memory copy (1355 GB/s) we can get solving a 2D diffusion problem on an Nvidia Tesla A100 GPU.

$$ ∇⋅(D ∇ C) = \frac{∂C}{∂t} $$

👉 Let's test the performance using a simple [scripts/perftest.jl](scripts/perftest.jl) script.

### Why to still bother with GPU computing in 2023
Because it is still challenging. Why?
- Very few software uses it efficiently.
- It requires to rethink the solving strategy as non-local operations will kill the fun.

## The challenge of today
The goal fo today is to solve a subsurface flow problem related to injection and extraction of fluid in the underground as it could occur in geothermal operations. For this purpose, we will solve an elliptic problem for fluid pressure diffusion, given impermeable boundary conditions (no flux) and two source terms, inkection and extraction wells. In addition, we will place a low permeability barrier in-between the wells to simulate a more challenging flow configuration. The model configuration is depicted hereafter:

![model setup](docs/model_setup.png)

Although on the vanilla side, this problem presents several challenges to be solved efficiently. We will need to achieve:
- an efficient steady-state solve of an elliptic equation
- handle source terms
- handle spatially variable material parameters

> :bulb: For practical purpose, we will work in 2D, however everything we will develop today is readily extensible to 3D.

The system of equation we will solve reads:

$$ q = -K~∇P_f ~, $$

$$ 0 = ∇⋅q -Q_f~, $$

where $q$ is the diffusive flux, $P_f$ the fluid pressure, $K$ is the spatially variable diffusion coefficient, and $Q_f$ the source term.

We will use a naïve iterative solving strategy combined to a finite-difference discretisation on a regular Cartesian staggered grid:

![staggrid](docs/staggrid.png)

The iterative approach relies in replacing the 0 in the mass balance equation by a pseudo-time derivative $∂/∂\tau$ and let it reahc a steady state:

$$ \frac{∂P_f}{∂\tau} = ∇⋅q -Q_f~. $$

Introducing the residual $RP_f$, one can re-write the system of equations as:

$$ q = -K~∇P_f ~, $$

$$ RP_f = ∇⋅q -Q_f~, $$

$$ \frac{∂P_f}{∂\tau} = -RP_f~. $$

We will stop the iterations when the $\mathrm{L_{inf}}$ norm of $P_f$ drops below a defined tolerance `max(abs.(RPf)) < ϵtol`.

## Hands-on I
Let's get started. In this first hands-on, we will work towards making an efficient iterative GPU solver for the forward steady state flow problem.

### Task 1: Steady-state diffusion problem
The first script we will work on is [scripts/geothermal_2D_noacc.jl](scripts/geothermal_2D_noacc.jl). This script builds upon the [scripts/visu_2D.jl](scripts/visu_2D.jl) scripts and contains the basic structure of the iterative code and the updated `# numerics` section.

As first task, let's complete the physics section in the iteration loop, replacing `# ???` by actual code.

Once done, let's run the script and briefly check how iteration count normalised by `nx` scales when changing the resolution.

### Task 2: The accelerated pseudo-transient method


### Task 3: From CPU to GPU using array programming


### Task 4: Kernel programming

#### Task 4a: CPU "kernel" programming
Actually using multi-threading

#### Task 4b: GPU kernel programming

## The optimisation problem

## Hands-on II

## Exercises (optionnal)