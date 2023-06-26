using Printf
using CairoMakie
using CUDA

macro d_xa(A) esc(:($A[ix+1, iz] - $A[ix, iz])) end
macro d_za(A) esc(:($A[ix, iz+1] - $A[ix, iz])) end
macro avx(A)  esc(:(0.5 * ($A[ix, iz] + $A[ix+1, iz]))) end
macro avz(A)  esc(:(0.5 * ($A[ix, iz] + $A[ix, iz+1]))) end
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avz(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
@views maxloc(A) = max.(A[2:end-1, 2:end-1], max.(max.(A[1:end-2, 2:end-1], A[3:end, 2:end-1]),
                                                  max.(A[2:end-1, 1:end-2], A[2:end-1, 3:end])))

function residual_fluxes!(Rqx, Rqz, qx, qz, Pf, K, dx, dz)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds if (ix<=size(Rqx, 1) - 2 && iz<=size(Rqx, 2)    ) Rqx[ix+1, iz] = qx[ix+1, iz] + @avx(K) * @d_xa(Pf) / dx end
    @inbounds if (ix<=size(Rqz, 1)     && iz<=size(Rqz, 2) - 2) Rqz[ix, iz+1] = qz[ix, iz+1] + @avz(K) * @d_za(Pf) / dz end
    return
end

function residual_pressure!(RPf, qx, qz, Qf, dx, dz)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds if (ix<=size(RPf, 1) && iz<=size(RPf, 2)) RPf[ix, iz] = @d_xa(qx) / dx + @d_za(qz) / dz - Qf[ix, iz] end
    return
end

function update_fluxes!(qx, qz, Rqx, Rqz, cfl, nx, nz, re)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds if (ix<=size(qx, 1) - 2 && iz<=size(qx, 2)    ) qx[ix+1, iz] -= Rqx[ix+1, iz] / (1.0 + 2cfl * nx / re) end
    @inbounds if (ix<=size(qz, 1)     && iz<=size(qz, 2) - 2) qz[ix, iz+1] -= Rqz[ix, iz+1] / (1.0 + 2cfl * nz / re) end
    return
end

function update_pressure!(Pf, RPf, K_max, vdτ, lz, re)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds if (ix<=size(Pf, 1) && iz<=size(Pf, 2)) Pf[ix, iz] -= RPf[ix, iz] * (vdτ * lz / re) / K_max[ix, iz] end
    return
end

@views function main()
    CUDA.device!(0) # select your GPU
    # physics
    lx, lz  = 2.0, 1.0 # domain extend
    k0_μ    = 1.0      # background permeability / fluid viscosity
    kb_μ    = 1e-6     # barrier permeability / fluid viscosity
    Q_in    = 1.0      # injection flux
    b_w     = 0.02lx   # barrier width
    b_b     = 0.3lz    # barrier bottom location
    b_t     = 0.8lz    # barrier top location
    # numerics
    nz      = 255
    nx      = ceil(Int, (nz + 1) * lx / lz) - 1
    nthread = (16, 16)
    nblock  = cld.((nx, nz), nthread)
    cfl     = 1 / 2.1
    ϵtol    = 1e-6
    maxiter = 30nx
    ncheck  = 2nx
    re      = 0.8π
    st      = ceil(Int, nx / 30)
    # preprocessing
    dx, dz  = lx / nx, lz / nz
    xc, zc  = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx), LinRange(dz / 2, lz - dz / 2, nz)
    vdτ     = cfl * min(dx, dz)
    # init
    Pf      = CUDA.zeros(Float64, nx, nz)
    RPf     = CUDA.zeros(Float64, nx, nz)
    qx      = CUDA.zeros(Float64, nx + 1, nz)
    Rqx     = CUDA.zeros(Float64, nx + 1, nz)
    qz      = CUDA.zeros(Float64, nx, nz + 1)
    Rqz     = CUDA.zeros(Float64, nx, nz + 1)
    Qf      = CUDA.zeros(Float64, nx, nz)
    K       = k0_μ .* CUDA.ones(Float64, nx, nz)
    # set low permeability barrier location
    K[ceil(Int, (lx/2-b_w)/dx):ceil(Int, (lx/2+b_w)/dx), ceil(Int, b_b/dz):ceil(Int, b_t/dz)] .= kb_μ
    # set wells location
    x_iw, x_ew, z_w = ceil.(Int, (lx / 5 / dx, 4lx / 5 / dx, 0.45lz / dz)) # well location
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
    plt = (fld = ( Pf = heatmap!(ax.Pf, xc, zc, Array(Pf); colormap=:turbo, colorrange=(-1,1)),
                   K  = heatmap!(ax.K , xc, zc, Array(log10.(K)); colormap=:turbo, colorrange=(-6,0)),
                   qM = heatmap!(ax.qM, xc, zc, qM; colormap=:turbo, colorrange=(0,30)),
                   ar = arrows!(ax.Pf, xc[1:st:end], zc[1:st:end], qx_c[1:st:end, 1:st:end], qz_c[1:st:end, 1:st:end]; lengthscale=0.05, arrowsize=15), ),
           err = scatterlines!(ax.err, Point2.(iters_evo, errs_evo), linewidth=4), )
    Colorbar(fig[1, 1][1, 2], plt.fld.Pf)
    Colorbar(fig[1, 2][1, 2], plt.fld.K)
    Colorbar(fig[2, 1][1, 2], plt.fld.qM)
    # approximate diagonal (Jacobi) preconditioner
    K_max = copy(K); K_max[2:end-1, 2:end-1] .= maxloc(K); K_max[:, [1, end]] .= K_max[:, [2, end-1]]
    # iterative loop
    err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        CUDA.@sync @cuda threads=nthread blocks=nblock residual_fluxes!(Rqx, Rqz, qx, qz, Pf, K, dx, dz)
        CUDA.@sync @cuda threads=nthread blocks=nblock update_fluxes!(qx, qz, Rqx, Rqz, cfl, nx, nz, re)
        CUDA.@sync @cuda threads=nthread blocks=nblock residual_pressure!(RPf, qx, qz, Qf, dx, dz)
        CUDA.@sync @cuda threads=nthread blocks=nblock update_pressure!(Pf, RPf, K_max, vdτ, lz, re)
        if iter % ncheck == 0
            err = maximum(abs.(RPf))
            push!(iters_evo, iter/nx); push!(errs_evo, err)
            # visu
            qx_c .= Array(avx(qx)); qz_c .= Array(avz(qz)); qM .= sqrt.(qx_c.^2 .+ qz_c.^2)
            qx_c ./= qM; qz_c ./= qM
            plt.fld.Pf[3] = Array(Pf)
            plt.fld.K[3]  = Array(log10.(K))
            plt.fld.qM[3] = qM
            plt.fld.ar[3] = qx_c[1:st:end, 1:st:end]
            plt.fld.ar[4] = qz_c[1:st:end, 1:st:end]
            plt.err[1] = Point2.(iters_evo, errs_evo)
            display(fig)
            @printf("  #iter/nx=%.1f, max(err)=%1.3e\n", iter/nx, err)
        end
        iter += 1
    end
    return
end

main()