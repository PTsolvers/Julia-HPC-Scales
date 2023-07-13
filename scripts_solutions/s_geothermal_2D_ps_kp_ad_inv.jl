using Printf, LinearAlgebra
using CairoMakie
using Enzyme

using ParallelStencil

@init_parallel_stencil(CUDA, Float64, 2)
CUDA.device!(1)

macro d_xa(A) esc(:($A[ix+1, iz] - $A[ix, iz])) end
macro d_za(A) esc(:($A[ix, iz+1] - $A[ix, iz])) end
macro avx(A)  esc(:(0.5 * ($A[ix, iz] + $A[ix+1, iz]))) end
macro avz(A)  esc(:(0.5 * ($A[ix, iz] + $A[ix, iz+1]))) end
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avz(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
@views maxloc(A) = max.(A[2:end-1, 2:end-1], max.(max.(A[1:end-2, 2:end-1], A[3:end, 2:end-1]),
                                                  max.(A[2:end-1, 1:end-2], A[2:end-1, 3:end])))

@parallel_indices (ix,iz) function smooth_d!(A2, A)
    if (ix>1 && ix<size(A, 1) && iz>1 && iz<size(A, 2))
        @inbounds A2[ix, iz] = A[ix, iz] + 0.2 * (A[ix+1, iz] - 2A[ix, iz] + A[ix-1, iz] + A[ix, iz+1] - 2A[ix, iz] + A[ix, iz-1])
    end
    return
end

function smooth!(A2, A; nsm=1)
    for _ ∈ 1:nsm
        @parallel smooth_d!(A2, A)
        A, A2 = A2, A
    end
    return
end

# forward
@parallel_indices (ix,iz) function residual_fluxes!(Rqx, Rqz, qx, qz, Pf, K, dx, dz)
    @inbounds if (ix<=size(Rqx, 1) - 2 && iz<=size(Rqx, 2)    ) Rqx[ix+1, iz] = qx[ix+1, iz] + @avx(K) * @d_xa(Pf) / dx end
    @inbounds if (ix<=size(Rqz, 1)     && iz<=size(Rqz, 2) - 2) Rqz[ix, iz+1] = qz[ix, iz+1] + @avz(K) * @d_za(Pf) / dz end
    return
end

@parallel_indices (ix,iz) function residual_pressure!(RPf, qx, qz, Qf, dx, dz)
    @inbounds if (ix<=size(RPf, 1) && iz<=size(RPf, 2)) RPf[ix, iz] = @d_xa(qx) / dx + @d_za(qz) / dz - Qf[ix, iz] end
    return
end

@parallel_indices (ix,iz) function update_fluxes!(qx, qz, Rqx, Rqz, cfl, nx, nz, re)
    @inbounds if (ix<=size(qx, 1) - 2 && iz<=size(qx, 2)    ) qx[ix+1, iz] -= Rqx[ix+1, iz] / (1.0 + 2cfl * nx / re) end
    @inbounds if (ix<=size(qz, 1)     && iz<=size(qz, 2) - 2) qz[ix, iz+1] -= Rqz[ix, iz+1] / (1.0 + 2cfl * nz / re) end
    return
end

@parallel_indices (ix,iz) function update_pressure!(Pf, RPf, K_max, vdτ, lz, re)
    @inbounds if (ix<=size(Pf, 1) && iz<=size(Pf, 2)) Pf[ix, iz] -= RPf[ix, iz] * (vdτ * lz / re) / K_max[ix, iz] end
    return
end
# forward

@views function forward_solve!(logK, fields, scalars, iter_params; visu=nothing)
    (;Pf, qx, qz, Qf, RPf, Rqx, Rqz, K)               = fields
    (;nx, nz, dx, dz)                                 = scalars
    (;cfl, re, vdτ, lz, ϵtol, maxiter, ncheck, K_max) = iter_params
    isnothing(visu) || ((;qx_c, qz_c, qM, fig, plt, st) = visu)
    K .= exp.(logK)
    # approximate diagonal (Jacobi) preconditioner
    K_max .= K; K_max[2:end-1, 2:end-1] .= maxloc(K); K_max[:, [1, end]] .= K_max[:, [2, end-1]]
    # iterative loop
    iters_evo = Float64[]; errs_evo = Float64[]
    err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        @parallel residual_fluxes!(Rqx, Rqz, qx, qz, Pf, K, dx, dz)
        @parallel update_fluxes!(qx, qz, Rqx, Rqz, cfl, nx, nz, re)
        @parallel residual_pressure!(RPf, qx, qz, Qf, dx, dz)
        @parallel update_pressure!(Pf, RPf, K_max, vdτ, lz, re)
        if iter % ncheck == 0
            err = maximum(abs.(RPf))
            push!(iters_evo, iter/nx); push!(errs_evo, err)
            @printf("  #iter/nx=%.1f, max(err)=%1.3e\n", iter/nx, err)
            if !isnothing(visu)
                qx_c .= Array(avx(qx)); qz_c .= Array(avz(qz)); qM .= sqrt.(qx_c.^2 .+ qz_c.^2)
                qx_c ./= qM; qz_c ./= qM
                plt.fld.Pf[3] = Array(Pf)
                plt.fld.K[3]  = Array(log10.(K))
                plt.fld.qM[3] = qM
                plt.fld.ar[3] = qx_c[1:st:end, 1:st:end]
                plt.fld.ar[4] = qz_c[1:st:end, 1:st:end]
                plt.err[1] = Point2.(iters_evo, errs_evo)
                display(fig)
            end
        end
        iter += 1
    end
    return
end

@views function adjoint_solve!(logK, fwd_params, adj_params, loss_params)
    # unpack forward
    (;Pf, qx, qz, Qf, RPf, Rqx, Rqz, K) = fwd_params.fields
    (;nx, nz, dx, dz)  = fwd_params.scalars
    # unpack adjoint
    (;P̄f, q̄x, q̄z, R̄Pf, R̄qx, R̄qz, Ψ_qx, Ψ_qz, Ψ_Pf)      = adj_params.fields
    (;∂J_∂Pf)                                           = loss_params.fields
    (;cfl, re_a, vdτ, lz, ϵtol, maxiter, ncheck, K_max) = adj_params.iter_params
    # iterative loop
    iters_evo = Float64[]; errs_evo = Float64[]
    err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        R̄qx .= Ψ_qx
        R̄qz .= Ψ_qz
        P̄f  .= .-∂J_∂Pf
        q̄x  .= 0.0
        q̄z  .= 0.0

        @parallel ∇=(Rqx->R̄qx, Rqz->R̄qz, qx->q̄x, qz->q̄z, Pf->P̄f) residual_fluxes!(Rqx, Rqz, qx, qz, Pf, K, dx, dz)
        P̄f[[1, end], :] .= 0.0; P̄f[:, [1, end]] .= 0.0
        @parallel update_pressure!(Ψ_Pf, P̄f, K_max, vdτ, lz, re_a)
        R̄Pf .= Ψ_Pf
        @parallel ∇=(RPf->R̄Pf, qx->q̄x, qz->q̄z) residual_pressure!(RPf, qx, qz, Qf, dx, dz)
        @parallel update_fluxes!(Ψ_qx, Ψ_qz, q̄x, q̄z, cfl, nx, nz, re_a)
        if iter % ncheck == 0
            err = maximum(abs.(P̄f))
            push!(iters_evo, iter/nx); push!(errs_evo, err)
            @printf("  #iter/nx=%.1f, max(err)=%1.6e\n", iter/nx, err)
        end
        iter += 1
    end
    return
end

@views function loss(logK, fwd_params, loss_params; kwargs...)
    (;Pf_obs)       = loss_params.fields
    (;ixobs, izobs) = loss_params.scalars
    @info "Forward solve"
    forward_solve!(logK, fwd_params...; kwargs...)
    Pf = fwd_params.fields.Pf
    return 0.5*sum((Pf[ixobs, izobs] .- Pf_obs).^2)
end

function ∇loss!(logK̄, logK, fwd_params, adj_params, loss_params; reg=nothing, kwargs...)
    # unpack
    (;R̄qx, R̄qz, Ψ_qx, Ψ_qz)    = adj_params.fields
    (;Pf, qx, qz, Rqx, Rqz, K) = fwd_params.fields
    (;dx, dz)                  = fwd_params.scalars
    (;Pf_obs, ∂J_∂Pf)          = loss_params.fields
    (;ixobs, izobs)            = loss_params.scalars
    @info "Forward solve"
    forward_solve!(logK, fwd_params...; kwargs...)
    # set tangent
    ∂J_∂Pf[ixobs, izobs] .= Pf[ixobs, izobs] .- Pf_obs
    @info "Adjoint solve"
    adjoint_solve!(logK, fwd_params, adj_params, loss_params)
    # evaluate gradient dJ_dK
    R̄qx .= .-Ψ_qx
    R̄qz .= .-Ψ_qz
    logK̄ .= 0.0
    @parallel ∇=(Rqx->R̄qx, Rqz->R̄qz, logK->logK̄) residual_fluxes!(Rqx, Rqz, qx, qz, Pf, logK, dx, dz)
    # Tikhonov regularisation (smoothing)
    if !isnothing(reg)
        (;nsm, Tmp) = reg
        Tmp .= logK̄; smooth!(logK̄, Tmp; nsm)
    end
    logK̄ .*= K # convert to dJ_dlogK by chain rule
    return
end

@views function main()
    # physics
    lx, lz  = 2.0, 1.0 # domain extend
    k0_μ    = 1.0      # background permeability / fluid viscosity
    kb_μ    = 1e-6     # barrier permeability / fluid viscosity
    Q_in    = 1.0      # injection flux
    b_w     = 0.02lx   # barrier width
    b_b     = 0.3lz    # barrier bottom location
    b_t     = 0.8lz    # barrier top location
    # observations
    xobs_rng = LinRange(-lx / 6, lx / 6, 8)
    zobs_rng = LinRange(0.25lz, 0.85lz , 8)
    # numerics
    nz       = 255
    nx       = ceil(Int, (nz + 1) * lx / lz) - 1
    cfl      = 1 / 2.1
    ϵtol     = 1e-6
    maxiter  = 30nx
    ncheck   = 2nx
    re       = 0.8π # fwd re
    st       = ceil(Int, nx / 30)
    # GD params
    ngd      = 50
    Δγ       = 0.2
    # preprocessing
    re_a     = 2re  # adjoint re
    dx, dz   = lx / nx, lz / nz
    xc, zc   = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx), LinRange(dz / 2, lz - dz / 2, nz)
    vdτ      = cfl * min(dx, dz)
    ixobs    = floor.(Int, (xobs_rng .- xc[1]) ./ dx) .+ 1
    izobs    = floor.(Int, (zobs_rng .- zc[1]) ./ dz) .+ 1
    # init
    Pf       = @zeros(nx, nz)
    RPf      = @zeros(nx, nz)
    qx       = @zeros(nx + 1, nz)
    Rqx      = @zeros(nx + 1, nz)
    qz       = @zeros(nx, nz + 1)
    Rqz      = @zeros(nx, nz + 1)
    Qf       = @zeros(nx, nz)
    K        = k0_μ .* @ones(nx, nz)
    logK     = @zeros(nx, nz)
    Tmp      = @zeros(nx, nz)
    # init adjoint storage
    Ψ_qx     = @zeros(nx + 1, nz)
    q̄x       = @zeros(nx + 1, nz)
    R̄qx      = @zeros(nx + 1, nz)
    Ψ_qz     = @zeros(nx, nz + 1)
    q̄z       = @zeros(nx, nz + 1)
    R̄qz      = @zeros(nx, nz + 1)
    Ψ_Pf     = @zeros(nx, nz)
    P̄f       = @zeros(nx, nz)
    R̄Pf      = @zeros(nx, nz)
    ∂J_∂Pf   = @zeros(nx, nz)
    dJ_dlogK = @zeros(nx, nz)
    # set low permeability barrier location
    K[ceil(Int, (lx/2-b_w)/dx):ceil(Int, (lx/2+b_w)/dx), ceil(Int, b_b/dz):ceil(Int, b_t/dz)] .= kb_μ
    logK .= log.(K)
    K_max = copy(K)
    # set wells location
    x_iw, x_ew, z_w = ceil.(Int, (lx / 5 / dx, 4lx / 5 / dx, 0.45lz / dz))
    Qf[x_iw:x_iw, z_w:z_w] .=  Q_in / dx / dz # injection
    Qf[x_ew:x_ew, z_w:z_w] .= -Q_in / dx / dz # extraction
    # init visu
    iters_evo = Float64[]; errs_evo = Float64[]
    qM, qx_c, qz_c = zeros(nx, nz), zeros(nx, nz), zeros(nx, nz)
    fig = Figure(resolution=(2500, 1200), fontsize=32)
    ax = ( Pf  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="Pf"),
           K   = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="log10(K)"),
           qM  = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="|q|"),
           err = Axis(fig[2, 2]; yscale=log10, title="Convergence", xlabel="# iter/nx", ylabel="error"), )
    plt = (fld = ( Pf   = heatmap!(ax.Pf, xc, zc, Array(Pf); colormap=:turbo, colorrange=(-1,1)),
                   K    = heatmap!(ax.K , xc, zc, Array(log10.(K)); colormap=:turbo, colorrange=(-6,0)),
                   xobs = scatter!(ax.K , vec(Point2.(xobs_rng, zobs_rng')); color=:white),
                   qM   = heatmap!(ax.qM, xc, zc, qM; colormap=:turbo, colorrange=(0,30)),
                   ar   = arrows!(ax.Pf, xc[1:st:end], zc[1:st:end], qx_c[1:st:end, 1:st:end], qz_c[1:st:end, 1:st:end]; lengthscale=0.05, arrowsize=15), ),
           err = scatterlines!(ax.err, Point2.(iters_evo, errs_evo), linewidth=4), )
    Colorbar(fig[1, 1][1, 2], plt.fld.Pf)
    Colorbar(fig[1, 2][1, 2], plt.fld.K)
    Colorbar(fig[2, 1][1, 2], plt.fld.qM)
    # action
    fwd_params = (
        fields      = (;Pf, qx, qz, Qf, RPf, Rqx, Rqz, K),
        scalars     = (;nx, nz, dx, dz),
        iter_params = (;cfl, re, vdτ, lz, ϵtol, maxiter, ncheck, K_max),
    )
    fwd_visu = (;qx_c, qz_c, qM, fig, plt, st)
    @info "Synthetic solve"
    forward_solve!(logK, fwd_params...; visu=fwd_visu)
    # store true data
    Pf_obs = copy(Pf[ixobs, izobs])
    adj_params = (
        fields  = (;P̄f, q̄x, q̄z, R̄Pf, R̄qx, R̄qz, Ψ_qx, Ψ_qz, Ψ_Pf),
        iter_params = (;cfl, re_a, vdτ, lz, ϵtol, maxiter, ncheck, K_max),
    )
    loss_params = (
        fields  = (;Pf_obs, ∂J_∂Pf),
        scalars = (;ixobs, izobs),
    )
    reg = (;nsm=20, Tmp)
    # loss functions
    J(_logK) = loss(_logK, fwd_params, loss_params)
    ∇J!(_logK̄, _logK) = ∇loss!(_logK̄, _logK, fwd_params, adj_params, loss_params; reg)
    @info "Inversion for K"
    # initial guess
    K    .= k0_μ
    logK .= log.(K)
    @info "Gradient descent - inversion for K"
    cost_evo = Float64[]
    for igd in 1:ngd
        printstyled("> GD iter $igd \n"; bold=true, color=:green)
        # evaluate gradient of the cost function
        ∇J!(dJ_dlogK, logK)
        # update logK
        γ = Δγ / maximum(abs.(dJ_dlogK))
        @. logK -= γ * dJ_dlogK
        @printf "  min(K) = %1.2e \n" minimum(K)
        # loss
        push!(cost_evo, J(logK))
        @printf "  --> Loss J = %1.2e (γ = %1.2e)\n" last(cost_evo)/first(cost_evo) γ
        # visu
        qx_c .= Array(avx(qx)); qz_c .= Array(avz(qz)); qM .= sqrt.(qx_c.^2 .+ qz_c.^2)
        qx_c ./= qM; qz_c ./= qM
        plt.fld.Pf[3] = Array(Pf)
        plt.fld.K[3]  = Array(log10.(K))
        plt.fld.qM[3] = qM
        plt.fld.ar[3] = qx_c[1:st:end, 1:st:end]
        plt.fld.ar[4] = qz_c[1:st:end, 1:st:end]
        plt.err[1] = Point2.(1:igd, cost_evo ./ 0.999cost_evo[1])
        display(fig)
    end
    return
end

main()
