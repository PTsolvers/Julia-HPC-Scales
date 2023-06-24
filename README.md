# Julia-HPC-Scales
Julia for HPC workshop @ [**SCALES conference 2023**](https://model.uni-mainz.de/scales-conference-2023/)

| The computational resources for this workshop are provided by the      |                |
| :------------: | :------------: |
| [**Paderborn Center for Parallel Computing (PC2)**](https://pc2.uni-paderborn.de/) | [![PC2](docs/Logo_PC2_rgb.jpg)](https://pc2.uni-paderborn.de/) |
| **PC2** is a member of the [**NHR Alliance**](https://www.nhr-verein.de/) | [![out visu](docs/NHR_logo.png)](https://www.nhr-verein.de/) |

> :warning: Make sure to `git pull` this repo right before starting the workshop on Monday morning in order to ensure you have access to the latest updates

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
- [**Exercises** (optional) :computer:](#exercises-optionnal)
  - Go for 3D
  - Make combined loss (pressure + flux)
- **Wrapping up** & outlook :beer:

## The `SMALL` print
The goal of today's workshop is to develop a fast iterative GPU-based solver for elliptic equations and use it to:
1. Solve a steady state subsurface flow problem (geothermal operations, injection and extraction of fluids)
2. Invert for the subsurface permeability having a sparse array of fluid pressure observations

We will not use any "black-box" tooling but rather try to develop concise and performant codes (300 lines of code, max) that execute on GPUs. We will also use automatic differentiation (AD) capabilities and the differentiable Julia stack to automatise the calculation of the adjoint solutions in the gradient-based inversion procedure.

The main Julia packages we will rely on are:
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU computing on Nvidia GPUs
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for AD on GPUs
- [CairoMakie.jl](https://github.com/MakieOrg/Makie.jl) for plotting
- [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to extend the "vanilla" gradient-descent procedure

Most of the workshop is based on "hands-on". Changes to the scripts are incremental and should allow to build up complexity throughout the day. Blanked-out scripts for most of the steps are available in the [scripts](scripts/) folder. Solutions scripts (following the `s_xxx.jl` pattern) will be shared at some point in the [scripts_solutions](scripts_solutions) folder.

#### :bulb: Useful extra resources
- The Julia language: [https://julialang.org](https://julialang.org)
- PDE on GPUs ETH Zurich course: [https://pde-on-gpu.vaw.ethz.ch](https://pde-on-gpu.vaw.ethz.ch)
- Julia Discourse (Julia Q&A): [https://discourse.julialang.org](https://discourse.julialang.org)
- Julia Slack (Julia dev chat): [https://julialang.org/slack/](https://julialang.org/slack/)

## VSCode on Noctua 2
Before we start, let's make sure that everyone can remote connect over SSH from within [VSCode](https://code.visualstudio.com/docs/remote/ssh) to a GPU node on the **Noctua 2** supercomputer we will use today.

Let's go back to the email titled "**SCALES workshop - login credentials**" in order to access the HackMD doc to go through the "**VSCode on the Compute Node (on Monday only)**" procedure.

If all went fine, you should be able to execute the following command in your Julia REPL:
```julia-repl
julia> include("scripts/visu_2D.jl")
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
- 1 addition + 1 multiplication + 1 division => **3 floating point operations**

👉 assuming $D$, $∂x$ are scalars, $q$ and $A$ are arrays of `Float64` (read from main memory)

### Performance that matters - an example
Not yet convinced? Let's have a look at an example.

Let's assess how close from memory copy (1355 GB/s) we can get solving a 2D diffusion problem on an Nvidia Tesla A100 GPU.

$$ ∇⋅(D ∇ C) = \frac{∂C}{∂t} $$

👉 Let's test the performance using a simple [perftest.jl](scripts/perftest.jl) script.

### Why to still bother with GPU computing in 2023
Because it is still challenging. Why?
- Very few codes use it efficiently.
- It requires to rethink the solving strategy as non-local operations will kill the fun.

## The challenge of today
The goal fo today is to solve a subsurface flow problem related to injection and extraction of fluid in the underground as it could occur in geothermal operations. For this purpose, we will solve an elliptic problem for fluid pressure diffusion, given impermeable boundary conditions (no flux) and two source terms, injection and extraction wells. In addition, we will place a low permeability barrier in-between the wells to simulate a more challenging flow configuration. The model configuration is depicted hereafter:

![model setup](docs/model_setup.png)

Despite looking simple, this problem presents several challenges to be solved efficiently. We will need to:
- efficiently solve an elliptic equation for the pressure
- handle source terms
- handle spatially variable material parameters

> :bulb: For practical purposes, we will work in 2D, however everything we will develop today is readily extensible to 3D.

The corresponding system of equation reads:

$$ q = -K~∇P_f ~, $$

$$ 0 = -∇⋅q + Q_f~, $$

where $q$ is the diffusive flux, $P_f$ the fluid pressure, $K$ is the spatially variable diffusion coefficient, and $Q_f$ the source term.

We will use an accelerated iterative solving strategy combined to a finite-difference discretisation on a regular Cartesian staggered grid:

![staggrid](docs/staggrid.png)

The iterative approach relies in replacing the 0 in the mass balance equation by a pseudo-time derivative $∂/∂\tau$ and let it reach a steady state:

$$ \frac{∂P_f}{∂\tau} = -∇⋅q + Q_f~. $$

Introducing the residual $RP_f$, one can re-write the system of equations as:

$$ q = -K~∇P_f ~, $$

$$ RP_f = ∇⋅q -Q_f~, $$

$$ \frac{∂P_f}{∂\tau} = -RP_f~. $$

We will stop the iterations when the $\mathrm{L_{inf}}$ norm of $P_f$ drops below a defined tolerance `max(abs.(RPf)) < ϵtol`.

This rather naive iterative strategy can be accelerated using the accelerated pseudo-transient method [(Räss et al., 2022)](https://doi.org/10.5194/gmd-15-5757-2022). In a nutshell, pseudo-time derivative can also be added to the fluxes turning the system of equations into a damped wave equation. The resulting augmented system of accelerated equations reads:

$$ Rq = q +K~∇P_f ~, $$

$$ \frac{∂q}{∂\tau_q} = -Rq~, $$

$$ RP_f = ∇⋅q -Q_f~, $$

$$ \frac{∂P_f}{∂\tau_p} = -RP_f~. $$

Finding the optimal damping parameter entering the definition of $∂\tau_q$ and $∂\tau_p$ further leads to significant acceleration in the solution procedure.

## Hands-on I
Let's get started. In this first hands-on, we will work towards making an efficient iterative GPU solver for the forward steady state flow problem.

### ✏️ Task 1: Steady-state diffusion problem
The first script we will work on is [geothermal_2D_noacc.jl](scripts/geothermal_2D_noacc.jl). This script builds upon the [visu_2D.jl](scripts/visu_2D.jl) scripts and contains the basic structure of the iterative code and the updated `# numerics` section.

As first task, let's complete the physics section in the iteration loop, replacing `# ???` by actual code.

Once done, let's run the script and briefly check how iteration count normalised by `nx` scales when changing the grid resolution.

### ✏️ Task 2: The accelerated pseudo-transient method
As you can see, the iteration count scales quadratically with increasing grid resolution and the overall iteration count is really large.

To address this issue, we can implement the accelerated pseudo-transient method [(Räss et al., 2022)](https://doi.org/10.5194/gmd-15-5757-2022). Practically, we will define residuals for both x and z fluxes (`Rqx`, `Rqz`) and provide an update rule based on some optimal numerical parameters consistent with the derivations in [(Räss et al., 2022)](https://doi.org/10.5194/gmd-15-5757-2022).

Starting from the [geothermal_2D.jl](scripts/geothermal_2D.jl) script, let's implement the acceleration technique. We can now reduce the cfl from `clf = 1 / 4.1` to `cfl = 1 / 2.1`, and `dτ = cfl * min(dx, dz)^2` becomes now `vdτ = cfl * min(dx, dz)`. Other modifications are the introduction of a numerical Reynolds number (`re = 0.8π`) and the change of `maxiter = 30nx` and `ncheck = 2nx`.

Let's complete the physics section in the iteration loop, replacing `# ???` with actual code.

Run the code and check how the iteration count scales as function of grid resolution.

### ✏️ Task 3: From CPU to GPU using array programming
So far so good, we have an efficient algorithm to iteratively converge the elliptic subsurface flow problem.

The next step is to briefly showcase how to port the vectorised Julia code, using array "broadcasting", to GPU computing using "array programming". As other languages, one way to proceed in Julia is to simply initialise all arrays in GPU memory.

Julia will create GPU function during the code compilation (yes, Julia code is actually compiled "just ahead of time") and execute the vectorised operations on the GPU. **In this workshop we will use [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) to target Nvidia GPUs**, but the [Julia GPU](https://juliagpu.org/) ecosystem supports AMD, ARM and Intel GPUs through [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), [Metal.jl](https://github.com/JuliaGPU/Metal.jl) and [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl), respectively.

The following changes are needed to execute the vectorised CPU Julia [geothermal_2D.jl](scripts/geothermal_2D.jl) script on a GPU:

- Array initialisation need now to be prepended with `CUDA.`, i.e., `Pf = zeros(nx, nz)` becomes `Pf = CUDA.zeros(Float64, nx, nz)`.
- Default precision in CUDA.jl is `Float32`, i.e., single precision. To use double precision, one needs to specify `Float64` during initialisation.
- Using `CUDA.device!(0)` allows to select a specific GPU on a multi-GPU node.
- To avoid "scalar indexing", indexing of `Qf` array for well location needs to be updated from `Qf[x_iw, z_w]` to `Qf[x_iw:x_iw, z_w:z_w]` (to make it look like a range).
- GPU arrays passed for plotting need to be converted back to host storage first, using `Array()`.

Implement those changes in the [geothermal_2D_gpu_ap.jl](scripts/geothermal_2D_gpu_ap.jl) script, replacing the `#= ??? =#` comments.

These minor changes allow us to use GPU acceleration out of the box. However, one may not achieve optimal performance using array programming on GPUs. The alternative is to use kernel programming.

### Task 4: Kernel programming
In GPU computing, "kernel programming" refers to explicitly programming the compute function instead of relying on array broadcasting operations. This permits to explicitly control kernel launch-parameters and to optimise operations inside the compute function to be executed on the GPU, aka kernel.

#### ✏️ Task 4a: CPU "kernel" programming
For a smooth transition, let's go back to our vectorised CPU code, [geothermal_2D.jl](scripts/geothermal_2D.jl). We will now create a version of this code where:
1. the physics should be isolated into specific compute functions which will then be called in the iterative loop,
2. we will use nested loops (as one would do in C programming) to express the computations.

The general design of a compute function looks as following
```julia
function compute_fun!(A, A2)
    Threads.@threads for iz ∈ axes(A, 2)
        for ix ∈ axes(A, 1)
            @inbounds if (ix<=size(A, 1) && iz<=size(A, 2)) A[ix, iz] = A2[ix, iz] end
        end
    end
    return
end
```

Note that for analogy with GPU computing, we perform the bound-checking using an `if` statement inside the nested loops and not use the loop ranges.

Julia offers "native" support for multi-threading. To use it, simply decorate the outer loop with `Threads.@threads` and launch Julia with the `-t auto` or `-t 4` flag (for selecting max available or 4 threads, respectively). In VScode Julia extension settings, you can also specify the amount of threads to use.

The `@inbounds` macro deactivates bound-checking and results in better performance. Note that is recommended to only use it once you have verified the code produces correct results, else you may get a segmentation fault.

Start from the [geothermal_2D_kp.jl](scripts/geothermal_2D_kp.jl) code and finalise it replacing the `# ???` with more valid content. Note that we introduces macros to perform derivatives and averaging in a point-wise fashion:
```julia
macro d_xa(A) esc(:($A[ix+1, iz] - $A[ix, iz])) end
macro d_za(A) esc(:($A[ix, iz+1] - $A[ix, iz])) end
macro avx(A)  esc(:(0.5 * ($A[ix, iz] + $A[ix+1, iz]))) end
macro avz(A)  esc(:(0.5 * ($A[ix, iz] + $A[ix, iz+1]))) end
```
These expression can be called using e.g. `@d_xa(A)` within the code and will be replaced by the preprocessor before compilation.

Once done, run the code, change the number of threads and check out the scaling in terms of wall-time while increasing grid resolution.

#### ✏️ Task 4b: GPU kernel programming
Now that we have a multi-threaded Julia CPU code with explicitly defined compute functions, we are ready to make the final step, i.e., port the [geothermal_2D_kp.jl](scripts/geothermal_2D_kp.jl) code to GPU using kernel programming.

The main change is to replace the nested loop, which is where the operations are fully or partly serialised (depend on single or multi-threading execution, respectively). The parallel processing power of GPUs come from their ability to execute the compute functions, or kernels, asynchronously on different threads. Assigning a each grid point in our computational domain to a different GPU thread enables massive parallelisation.

The idea is to replace:
```julia
function compute_fun!(A, A2)
    Threads.@threads for iz ∈ axes(A, 2)
        for ix ∈ axes(A, 1)
            @inbounds if (ix<=size(A, 1) && iz<=size(A, 2)) A[ix, iz] = A2[ix, iz] end
        end
    end
    return
end
```
by
```julia
function compute_fun_d!(A, A2)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds if (ix<=size(A, 1) && iz<=size(A, 2)) A[ix, iz] = A2[ix, iz] end
    return
end
```

The GPU version of the compute function, `compute_fun_d`, will be executed by each thread on the GPU. The only remaining is to launch the function on as many GPU threads as there are grid points in our computational domain. This step is achieved using the "kernel launch parameters", which define the CUDA block and grid size. The picture below (from [PDE on GPUs](https://pde-on-gpu.vaw.ethz.ch/lecture6/#gpu_architecture_and_kernel_programming)) depicts this concept:

![staggrid](docs/cuda_grid.png)

> :bulb: Playing with GPUs - the rules
> 
> - Current GPUs allow typically a maximum of 1024 threads per block.
> - The maximum number of blocks allowed is huge; computing the largest possible array on the GPU will make you run out of device memory (currently 16-80 GB) before hitting the maximal number of blocks when selecting sensible kernel launch parameters (usually threads per block >= 256).
> - Threads, blocks and grid have 3D "Cartesian" topology, which is very useful for 1D, 2D and 3D Cartesian finite-difference domains.

Starting from the [geothermal_2D_gpu_kp.jl](scripts/geothermal_2D_gpu_kp.jl) script, add content to the compute functions such that they would execute on the GPU.

Use following kernel launch parameters
```julia
nthread = (16, 16)
nblock  = cld.((nx, nz), nthread)
```
and add the following parameters for a GPU kernel launch:
```julia
CUDA.@sync @cuda threads=nthread blocks=nblock compute_fun_d!(A, A2)
```
Yay :tada:, if you made it here then we are ready to use our efficient GPU-based forward model for optimisation problem.
## The optimisation problem

## Hands-on II

## Exercises (optionnal)
