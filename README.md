# Julia-HPC-Scales
Julia for HPC workshop @ SCALES conference 2023

## Program

### Morning session (9h - 12h30)
- Troubleshooting login to Noctua2
- **Brief intro to Julia**'s ecosystem :book:
  - Performance, CPUs, GPUs, array and kernel programming
- Presentation of **the challenge of the day** :book:
  - Optimising injection/extraction from a heterogeneous reservoir
- **Hands-on I** - solving the forward problem :computer:
  - Steady-state diffusion problem
  - The accelerated pseudo-transient method
  - From CPU to GPU array programming
  - Kernel programming (performance)
    - CPU "kernel" programming -> multi-threading
    - GPU kernel programming

### Afternoon session (13h30 - 17h00)
- Presentation of **the optimisation problem** :book:
  - Tha adjoint method
  - Julia and the automatic differentiation (AD) tooling
- **Hands-on II** - HPC GPU-based inversions :computer:
  - The adjoint problem with AD
  - GPU-based adjoint solver using [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)
  - Sensitivity analysis
  - Gradient-based inversion (Gradient descent - GD)
    - Vanilla GD by hand
    - using [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
- **Exercises** (optionnal) :computer:
  - go for 3D
  - make combined loss (pressure + flux)
- **Wrapping up** & outlook :beer:

