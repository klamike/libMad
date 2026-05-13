const LIBMAD_STATUS_UNKNOWN = Clonglong(0)
const LIBMAD_STATUS_OPTIMAL = Clonglong(1)
const LIBMAD_STATUS_INFEASIBLE = Clonglong(2)
const LIBMAD_STATUS_UNBOUNDED = Clonglong(3)
const LIBMAD_STATUS_INTERRUPTED = Clonglong(4)
const LIBMAD_STATUS_ITERATION_LIMIT = Clonglong(5)
const LIBMAD_STATUS_TIME_LIMIT = Clonglong(6)
const LIBMAD_STATUS_ERROR = Clonglong(7)

push!(function_sigs, """int libmad_madipm_batch_lp_solve_csr_device(
    libmad_int nvar,
    libmad_int ncon,
    libmad_int nnz,
    libmad_int batch_size,
    const int32_t* rowptr,
    const int32_t* colidx,
    const libmad_real* values,
    const libmad_real* objective,
    libmad_real objective_scale,
    libmad_real objective_offset,
    const libmad_real* lvar,
    const libmad_real* uvar,
    const libmad_real* lcon,
    const libmad_real* ucon,
    const libmad_int* branch_var,
    const libmad_real* branch_lower,
    const libmad_real* branch_upper,
    libmad_real tol,
    libmad_real time_limit,
    libmad_int max_iter,
    LibMadTerminationCheck terminate_check,
    void* terminate_data,
    libmad_int* statuses,
    libmad_real* objectives,
    libmad_int* iterations)""")

push!(function_sigs, """int libmad_madipm_batch_lp_solve_cvxpy_device(
    libmad_int nvar,
    libmad_int ncon,
    libmad_int nnz_aug,
    libmad_int nnz_a,
    libmad_int batch_size,
    const libmad_int* a_rows,
    const libmad_int* a_cols,
    const libmad_int* b_index,
    const int8_t* row_sense,
    const libmad_real* q_values,
    const libmad_real* a_values,
    libmad_real tol,
    libmad_real time_limit,
    libmad_int max_iter,
    libmad_real* x_out,
    libmad_real* y_out,
    libmad_int* statuses,
    libmad_real* objectives,
    libmad_int* iterations)""")

push!(function_sigs, """int libmad_madipm_batch_lp_solve_cvxpy_device_with_state(
    libmad_int nvar,
    libmad_int ncon,
    libmad_int nnz_aug,
    libmad_int nnz_a,
    libmad_int batch_size,
    const libmad_int* a_rows,
    const libmad_int* a_cols,
    const libmad_int* b_index,
    const int8_t* row_sense,
    const libmad_real* q_values,
    const libmad_real* a_values,
    libmad_real tol,
    libmad_real time_limit,
    libmad_int max_iter,
    libmad_real* x_out,
    libmad_real* y_out,
    libmad_int* statuses,
    libmad_real* objectives,
    libmad_int* iterations,
    void** state_out)""")

push!(function_sigs, """int libmad_madipm_batch_lp_cvxpy_device_vjp(
    void* state,
    const libmad_real* dprimal,
    const libmad_real* ddual,
    libmad_real* dq_out,
    libmad_real* dA_out)""")

push!(function_sigs, """int libmad_madipm_batch_lp_cvxpy_device_delete_state(
    void* state)""")

function _libmad_check_ptr(name::Symbol, ptr::Ptr)
    ptr == C_NULL && throw(ArgumentError("null pointer for $name"))
    return
end

Base.@ccallable function libmad_madipm_batch_lp_solve_csr_device(
    nvar_::Clonglong,
    ncon_::Clonglong,
    nnz_::Clonglong,
    batch_size_::Clonglong,
    rowptr_ptr::Ptr{Int32},
    colidx_ptr::Ptr{Int32},
    values_ptr::Ptr{Cdouble},
    objective_ptr::Ptr{Cdouble},
    objective_scale::Cdouble,
    objective_offset::Cdouble,
    lvar_ptr::Ptr{Cdouble},
    uvar_ptr::Ptr{Cdouble},
    lcon_ptr::Ptr{Cdouble},
    ucon_ptr::Ptr{Cdouble},
    branch_var_ptr::Ptr{Clonglong},
    branch_lower_ptr::Ptr{Cdouble},
    branch_upper_ptr::Ptr{Cdouble},
    tol_::Cdouble,
    time_limit_::Cdouble,
    max_iter_::Clonglong,
    terminate_check::Ptr{Cvoid},
    terminate_data::Ptr{Cvoid},
    statuses_ptr::Ptr{Clonglong},
    objectives_ptr::Ptr{Cdouble},
    iterations_ptr::Ptr{Clonglong},
)::Cint
    try
        @static if Sys.isapple()
            return Cint(-2)
        else
            nvar = Int(nvar_)
            ncon = Int(ncon_)
            nnz = Int(nnz_)
            batch_size = Int(batch_size_)
            nvar >= 0 && ncon >= 0 && nnz >= 0 && batch_size > 0 ||
                throw(ArgumentError("invalid model dimensions"))
            rowptr0 = _libmad_device_wrap(Int32, rowptr_ptr, (ncon + 1,), :rowptr)
            colidx0 = _libmad_device_wrap(Int32, colidx_ptr, (nnz,), :colidx)
            values = _libmad_device_wrap(Float64, values_ptr, (nnz,), :values)
            objective_base = _libmad_device_wrap(Float64, objective_ptr, (nvar,), :objective)
            lvar_base = _libmad_device_wrap(Float64, lvar_ptr, (nvar,), :lvar)
            uvar_base = _libmad_device_wrap(Float64, uvar_ptr, (nvar,), :uvar)
            lcon_base = _libmad_device_wrap(Float64, lcon_ptr, (ncon,), :lcon)
            ucon_base = _libmad_device_wrap(Float64, ucon_ptr, (ncon,), :ucon)

            branch_var = Int.(_libmad_wrap(branch_var_ptr, batch_size, :branch_var))
            branch_lower = Float64.(_libmad_wrap(branch_lower_ptr, batch_size, :branch_lower))
            branch_upper = Float64.(_libmad_wrap(branch_upper_ptr, batch_size, :branch_upper))

            rowptr = rowptr0 .+ Int32(1)
            colidx = colidx0 .+ Int32(1)
            csr = CUDA.CUSPARSE.CuSparseMatrixCSR(rowptr, colidx, values, (ncon, nvar))

            scale = Float64(objective_scale)
            c0_batch = CUDA.fill(scale * Float64(objective_offset), batch_size)
            lvar, uvar, infeasible = _libmad_branch_bounds_device(
                lvar_base, uvar_base, branch_var, branch_lower, branch_upper, nvar, batch_size)
            A, c, lcon, ucon = _libmad_scale_device_batch_lp(
                csr, objective_base, scale, lcon_base, ucon_base, lvar, uvar,
                ncon, nvar, batch_size)
            x0 = _libmad_initial_point_device(lvar, uvar)
            y0 = CUDA.zeros(Float64, ncon, batch_size)

            meta = NLPModels.BatchNLPModelMeta{Float64, typeof(x0)}(
                batch_size,
                nvar;
                x0,
                lvar,
                uvar,
                ncon,
                y0,
                lcon,
                ucon,
                nnzj = nnz,
                nnzh = 0,
                minimize = true,
                islp = true,
                name = "libMadMadIPMDeviceBatchLP",
            )
            gpu_bnlp = BatchQuadraticModels.BatchLinearModel(meta, c, c0_batch, A)
            _libmad_solve_gpu_bnlp!(
                gpu_bnlp,
                infeasible,
                Float64(tol_),
                Float64(time_limit_),
                Int(max_iter_),
                terminate_check,
                terminate_data,
                statuses_ptr,
                objectives_ptr,
                iterations_ptr,
            )
            return Cint(0)
        end
    catch e
        Base.showerror(stderr, e, Base.catch_backtrace())
        println(stderr)
        return Cint(-1)
    end
end

function _libmad_wrap(ptr::Ptr{T}, n::Integer, name::Symbol) where {T}
    _libmad_check_ptr(name, ptr)
    return unsafe_wrap(Vector{T}, ptr, Int(n))
end

function _libmad_device_wrap(::Type{T}, ptr::Ptr{T}, dims::NTuple{N, Int}, name::Symbol) where {T, N}
    if prod(dims) == 0
        return CUDA.CuArray{T, N}(undef, dims)
    end
    _libmad_check_ptr(name, ptr)
    cuptr = CUDA.CuPtr{T}(UInt(ptr))
    return unsafe_wrap(CUDA.CuArray{T, N}, cuptr, dims; own = false)
end

function _libmad_cvxpy_bounds_kernel!(lcon, ucon, A_aug, b_index, row_sense, ncon::Int32, batch_size::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > ncon * batch_size && return nothing
    row = (i - Int32(1)) % ncon + Int32(1)
    batch = (i - Int32(1)) ÷ ncon + Int32(1)
    @inbounds begin
        nz = b_index[row]
        b = nz > 0 ? A_aug[nz, batch] : zero(eltype(A_aug))
        rhs = -b
        sense = row_sense[row]
        if sense == Int8(0)
            lcon[row, batch] = rhs
            ucon[row, batch] = rhs
        elseif sense == Int8(1)
            lcon[row, batch] = rhs
            ucon[row, batch] = typemax(eltype(A_aug))
        else
            lcon[row, batch] = NaN
            ucon[row, batch] = NaN
        end
    end
    return nothing
end

@static if Sys.isapple()
    function _libmad_cvxpy_batch_operator(args...)
        error("libMad MadIPM cvxpylayers device solve requires CUDA")
    end

    function _libmad_cvxpy_bounds(args...)
        error("libMad MadIPM cvxpylayers device solve requires CUDA")
    end
else
    function _libmad_cvxpy_batch_operator(ncon::Int, nvar::Int, nnz_a::Int, A_aug, a_rows, a_cols)
        rowptr, colidx = BatchQuadraticModels._coo_to_csr(a_rows, ncon)
        nz_map = CUDA.CuArray(UnitRange{Int64}(1, nnz_a))
        return BatchQuadraticModels._build_op(A_aug, a_rows, a_cols, rowptr, nz_map, a_cols, colidx)
    end

    function _libmad_cvxpy_bounds(A_aug, b_index, row_sense, ncon::Int, batch_size::Int)
        lcon = CUDA.fill(-Inf, ncon, batch_size)
        ucon = CUDA.fill( Inf, ncon, batch_size)
        n = ncon * batch_size
        if n > 0
            threads = 256
            CUDA.@cuda threads = threads blocks = cld(n, threads) _libmad_cvxpy_bounds_kernel!(
                lcon, ucon, A_aug, b_index, row_sense, Int32(ncon), Int32(batch_size))
        end
        return lcon, ucon
    end
end

mutable struct _LibMadCvxpyBatchLPState
    solver::Any
    a_rows::Any
    a_cols::Any
    b_index::Any
    row_sense::Any
    nvar::Int
    ncon::Int
    nnz_aug::Int
    nnz_a::Int
    batch_size::Int
end

function _libmad_store_ref(obj)
    ptr = pointer_from_objref(obj)
    libmad_refs[ptr] = obj
    return ptr
end

function _libmad_madipm_batch_lp_solve_cvxpy_device_impl(
    nvar_::Clonglong,
    ncon_::Clonglong,
    nnz_aug_::Clonglong,
    nnz_a_::Clonglong,
    batch_size_::Clonglong,
    a_rows_ptr::Ptr{Clonglong},
    a_cols_ptr::Ptr{Clonglong},
    b_index_ptr::Ptr{Clonglong},
    row_sense_ptr::Ptr{Int8},
    q_values_ptr::Ptr{Cdouble},
    a_values_ptr::Ptr{Cdouble},
    tol_::Cdouble,
    time_limit_::Cdouble,
    max_iter_::Clonglong,
    x_out_ptr::Ptr{Cdouble},
    y_out_ptr::Ptr{Cdouble},
    statuses_ptr::Ptr{Clonglong},
    objectives_ptr::Ptr{Cdouble},
    iterations_ptr::Ptr{Clonglong},
    state_out_ptr::Ptr{Ptr{Cvoid}},
)
    @static if Sys.isapple()
        return Cint(-2)
    else
        state_out_ptr != C_NULL && unsafe_store!(state_out_ptr, C_NULL)
        nvar = Int(nvar_)
        ncon = Int(ncon_)
        nnz_aug = Int(nnz_aug_)
        nnz_a = Int(nnz_a_)
        batch_size = Int(batch_size_)
        nvar >= 0 && ncon >= 0 && nnz_aug >= nnz_a >= 0 && batch_size > 0 ||
            throw(ArgumentError("invalid model dimensions"))
        a_rows = _libmad_device_wrap(Int64, a_rows_ptr, (nnz_a,), :a_rows)
        a_cols = _libmad_device_wrap(Int64, a_cols_ptr, (nnz_a,), :a_cols)
        b_index = _libmad_device_wrap(Int64, b_index_ptr, (ncon,), :b_index)
        row_sense = _libmad_device_wrap(Int8, row_sense_ptr, (ncon,), :row_sense)
        q_aug = _libmad_device_wrap(Float64, q_values_ptr, (nvar + 1, batch_size), :q_values)
        A_aug = _libmad_device_wrap(Float64, a_values_ptr, (nnz_aug, batch_size), :a_values)
        x_out = _libmad_device_wrap(Float64, x_out_ptr, (nvar, batch_size), :x_out)
        y_out = _libmad_device_wrap(Float64, y_out_ptr, (ncon, batch_size), :y_out)

        A = _libmad_cvxpy_batch_operator(ncon, nvar, nnz_a, A_aug, a_rows, a_cols)
        c = view(q_aug, 1:nvar, :)
        c0 = view(q_aug, nvar + 1, :)
        lcon, ucon = _libmad_cvxpy_bounds(A_aug, b_index, row_sense, ncon, batch_size)
        lvar = CUDA.fill(-Inf, nvar, batch_size)
        uvar = CUDA.fill( Inf, nvar, batch_size)
        x0 = CUDA.zeros(Float64, nvar, batch_size)
        y0 = CUDA.zeros(Float64, ncon, batch_size)

        meta = NLPModels.BatchNLPModelMeta{Float64, typeof(x0)}(
            batch_size,
            nvar;
            x0,
            lvar,
            uvar,
            ncon,
            y0,
            lcon,
            ucon,
            nnzj = nnz_a,
            nnzh = 0,
            minimize = true,
            islp = true,
            name = "libMadMadIPMCvxpyDeviceBatchLP",
        )
        gpu_bnlp = BatchQuadraticModels.BatchLinearModel(meta, c, c0, A)
        solver = MadIPM.UniformBatchMPCSolver(
            gpu_bnlp;
            uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
            tol = tol_ > 0 ? Float64(tol_) : 1e-6,
            max_wall_time = time_limit_ > 0 ? Float64(time_limit_) : 1e6,
            max_iter = max_iter_ > 0 ? Int(max_iter_) : typemax(Int),
            print_level = MadNLP.ERROR,
            rethrow_error = false,
            check_batch_structure = false,
            fixed_variable_treatment = MadNLP.RelaxBound,
            regularization = MadIPM.FixedRegularization(1e-10, 1e-10),
        )
        stats = MadIPM.solve!(solver; fetch_solution = false)
        MadNLP.unpack_x!(x_out, solver.bcb, solver.x)
        MadNLP.unpack_y!(y_out, solver.bcb, MadNLP.full(solver.y))

        objective = objectives_ptr == C_NULL ? nothing : Array(vec(solver.workspace.obj_val))
        @inbounds for j in eachindex(stats.status)
            _libmad_write_result!(
                statuses_ptr, objectives_ptr, iterations_ptr, j,
                _libmad_status(stats.status[j]),
                objective === nothing ? NaN : objective[j],
                stats.iter[j])
        end

        if state_out_ptr != C_NULL
            state = _LibMadCvxpyBatchLPState(
                solver, a_rows, a_cols, b_index, row_sense,
                nvar, ncon, nnz_aug, nnz_a, batch_size,
            )
            unsafe_store!(state_out_ptr, _libmad_store_ref(state))
        end
        return Cint(0)
    end
end

Base.@ccallable function libmad_madipm_batch_lp_solve_cvxpy_device(
    nvar_::Clonglong,
    ncon_::Clonglong,
    nnz_aug_::Clonglong,
    nnz_a_::Clonglong,
    batch_size_::Clonglong,
    a_rows_ptr::Ptr{Clonglong},
    a_cols_ptr::Ptr{Clonglong},
    b_index_ptr::Ptr{Clonglong},
    row_sense_ptr::Ptr{Int8},
    q_values_ptr::Ptr{Cdouble},
    a_values_ptr::Ptr{Cdouble},
    tol_::Cdouble,
    time_limit_::Cdouble,
    max_iter_::Clonglong,
    x_out_ptr::Ptr{Cdouble},
    y_out_ptr::Ptr{Cdouble},
    statuses_ptr::Ptr{Clonglong},
    objectives_ptr::Ptr{Cdouble},
    iterations_ptr::Ptr{Clonglong},
)::Cint
    try
        return _libmad_madipm_batch_lp_solve_cvxpy_device_impl(
            nvar_, ncon_, nnz_aug_, nnz_a_, batch_size_,
            a_rows_ptr, a_cols_ptr, b_index_ptr, row_sense_ptr,
            q_values_ptr, a_values_ptr, tol_, time_limit_, max_iter_,
            x_out_ptr, y_out_ptr, statuses_ptr, objectives_ptr, iterations_ptr,
            Ptr{Ptr{Cvoid}}(C_NULL),
        )
    catch e
        Base.showerror(stderr, e, Base.catch_backtrace())
        println(stderr)
        return Cint(-1)
    end
end

Base.@ccallable function libmad_madipm_batch_lp_solve_cvxpy_device_with_state(
    nvar_::Clonglong,
    ncon_::Clonglong,
    nnz_aug_::Clonglong,
    nnz_a_::Clonglong,
    batch_size_::Clonglong,
    a_rows_ptr::Ptr{Clonglong},
    a_cols_ptr::Ptr{Clonglong},
    b_index_ptr::Ptr{Clonglong},
    row_sense_ptr::Ptr{Int8},
    q_values_ptr::Ptr{Cdouble},
    a_values_ptr::Ptr{Cdouble},
    tol_::Cdouble,
    time_limit_::Cdouble,
    max_iter_::Clonglong,
    x_out_ptr::Ptr{Cdouble},
    y_out_ptr::Ptr{Cdouble},
    statuses_ptr::Ptr{Clonglong},
    objectives_ptr::Ptr{Cdouble},
    iterations_ptr::Ptr{Clonglong},
    state_out_ptr::Ptr{Ptr{Cvoid}},
)::Cint
    try
        _libmad_check_ptr(:state_out, state_out_ptr)
        return _libmad_madipm_batch_lp_solve_cvxpy_device_impl(
            nvar_, ncon_, nnz_aug_, nnz_a_, batch_size_,
            a_rows_ptr, a_cols_ptr, b_index_ptr, row_sense_ptr,
            q_values_ptr, a_values_ptr, tol_, time_limit_, max_iter_,
            x_out_ptr, y_out_ptr, statuses_ptr, objectives_ptr, iterations_ptr,
            state_out_ptr,
        )
    catch e
        Base.showerror(stderr, e, Base.catch_backtrace())
        println(stderr)
        return Cint(-1)
    end
end

function _libmad_adjoint_solve_kkt!(bkkt, d)
    lb_off = d.n + d.m
    MadIPM._reduce_rhs_batch!(
        d.values, d.ind_lb, lb_off, bkkt.l_diag,
                  d.ind_ub, lb_off + d.nlb, bkkt.u_diag,
    )
    rhs = bkkt.rhs_buffer
    pd = MadNLP.primal_dual(d)
    copyto!(rhs, pd)
    MadNLP.solve_linear_system!(bkkt, rhs)
    copyto!(pd, rhs)
    MadIPM._finish_aug_solve_batch!(
        d.values, d.ind_lb, lb_off, bkkt.l_lower, bkkt.l_diag,
                  d.ind_ub, lb_off + d.nlb, bkkt.u_lower, bkkt.u_diag,
    )
    return d
end

function _libmad_refactorize_solution_kkt!(solver)
    MadIPM.reset_active_view!(solver.batch_views)
    MadIPM.set_aug_diagonal_reg!(solver.kkt, solver)
    MadNLP.eval_jac_wrapper!(solver, solver.kkt)
    MadNLP.eval_lag_hess_wrapper!(solver, solver.kkt)
    MadNLP.build_kkt!(solver.kkt)
    MadNLP.factorize_kkt!(solver.kkt)
    return solver
end

function _libmad_cvxpy_vjp_q_kernel!(
    dq_out,
    adj_x,
    obj_sign,
    obj_scale,
    nvar::Int32,
    batch_size::Int32,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    nrows = nvar + Int32(1)
    i > nrows * batch_size && return nothing
    row = (i - Int32(1)) % nrows + Int32(1)
    batch = (i - Int32(1)) ÷ nrows + Int32(1)
    @inbounds begin
        if row <= nvar
            scale = obj_sign[batch] * obj_scale[batch]
            dq_out[row, batch] = -adj_x[row, batch] * scale
        else
            dq_out[row, batch] = zero(eltype(dq_out))
        end
    end
    return nothing
end

function _libmad_cvxpy_vjp_A_kernel!(
    dA_out,
    x,
    adj_y,
    con_scale,
    a_rows,
    a_cols,
    b_index,
    nnz_a::Int32,
    ncon::Int32,
    batch_size::Int32,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    nitems = nnz_a + ncon
    i > nitems * batch_size && return nothing
    item = (i - Int32(1)) % nitems + Int32(1)
    batch = (i - Int32(1)) ÷ nitems + Int32(1)
    @inbounds begin
        if item <= nnz_a
            row = a_rows[item]
            col = a_cols[item]
            dy = adj_y[row, batch] * con_scale[row, batch]
            dA_out[item, batch] = -x[col, batch] * dy
        else
            row = item - nnz_a
            nz = b_index[row]
            if nz > 0
                dy = adj_y[row, batch] * con_scale[row, batch]
                dA_out[nz, batch] = -dy
            end
        end
    end
    return nothing
end

@static if Sys.isapple()
    function _libmad_cvxpy_device_vjp!(args...)
        error("libMad MadIPM cvxpylayers device VJP requires CUDA")
    end
else
    function _libmad_cvxpy_device_vjp!(
        state::_LibMadCvxpyBatchLPState,
        dprimal,
        ddual,
        dq_out,
        dA_out,
    )
        solver = state.solver
        _libmad_refactorize_solution_kkt!(solver)

        MT = typeof(MadNLP.full(solver.d))
        VT = typeof(vec(MadNLP.full(solver.d)))
        w = MadIPM.BatchUnreducedKKTVector(
            MT, VT,
            solver.d.n,
            solver.d.m,
            solver.d.nlb,
            solver.d.nub,
            state.batch_size,
            solver.d.ind_lb,
            solver.d.ind_ub,
        )
        fill!(MadNLP.full(w), 0.0)
        view(MadNLP.primal(w), 1:state.nvar, :) .= dprimal
        MadNLP.dual(w) .= (ddual .* (solver.bcb.obj_sign ./ solver.bcb.obj_scale)) .* solver.bcb.con_scale
        _libmad_adjoint_solve_kkt!(solver.kkt, w)

        fill!(dA_out, 0.0)
        threads = 256
        nq = (state.nvar + 1) * state.batch_size
        if nq > 0
            CUDA.@cuda threads = threads blocks = cld(nq, threads) _libmad_cvxpy_vjp_q_kernel!(
                dq_out,
                MadNLP.primal(w),
                solver.bcb.obj_sign,
                solver.bcb.obj_scale,
                Int32(state.nvar),
                Int32(state.batch_size),
            )
        end
        nA = (state.nnz_a + state.ncon) * state.batch_size
        if nA > 0
            CUDA.@cuda threads = threads blocks = cld(nA, threads) _libmad_cvxpy_vjp_A_kernel!(
                dA_out,
                MadNLP.variable(solver.x),
                MadNLP.dual(w),
                solver.bcb.con_scale,
                state.a_rows,
                state.a_cols,
                state.b_index,
                Int32(state.nnz_a),
                Int32(state.ncon),
                Int32(state.batch_size),
            )
        end
        return
    end
end

Base.@ccallable function libmad_madipm_batch_lp_cvxpy_device_vjp(
    state_ptr::Ptr{Cvoid},
    dprimal_ptr::Ptr{Cdouble},
    ddual_ptr::Ptr{Cdouble},
    dq_out_ptr::Ptr{Cdouble},
    dA_out_ptr::Ptr{Cdouble},
)::Cint
    try
        _libmad_check_ptr(:state, state_ptr)
        haskey(libmad_refs, state_ptr) || throw(ArgumentError("unknown cvxpy batch LP state"))
        state = libmad_refs[state_ptr]::_LibMadCvxpyBatchLPState
        dprimal = _libmad_device_wrap(Float64, dprimal_ptr, (state.nvar, state.batch_size), :dprimal)
        ddual = _libmad_device_wrap(Float64, ddual_ptr, (state.ncon, state.batch_size), :ddual)
        dq_out = _libmad_device_wrap(Float64, dq_out_ptr, (state.nvar + 1, state.batch_size), :dq_out)
        dA_out = _libmad_device_wrap(Float64, dA_out_ptr, (state.nnz_aug, state.batch_size), :dA_out)
        _libmad_cvxpy_device_vjp!(state, dprimal, ddual, dq_out, dA_out)
        return Cint(0)
    catch e
        Base.showerror(stderr, e, Base.catch_backtrace())
        println(stderr)
        return Cint(-1)
    end
end

Base.@ccallable function libmad_madipm_batch_lp_cvxpy_device_delete_state(
    state_ptr::Ptr{Cvoid},
)::Cint
    haskey(libmad_refs, state_ptr) || return Cint(1)
    delete!(libmad_refs, state_ptr)
    return Cint(0)
end

function _libmad_status(status::MadNLP.Status)
    if MadNLP.SOLVE_SUCCEEDED <= status <= MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        return LIBMAD_STATUS_OPTIMAL
    elseif status == MadNLP.INFEASIBLE_PROBLEM_DETECTED
        return LIBMAD_STATUS_INFEASIBLE
    elseif status == MadNLP.DIVERGING_ITERATES
        return LIBMAD_STATUS_UNBOUNDED
    elseif status == MadNLP.USER_REQUESTED_STOP
        return LIBMAD_STATUS_INTERRUPTED
    elseif status == MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        return LIBMAD_STATUS_ITERATION_LIMIT
    elseif status == MadNLP.MAXIMUM_WALLTIME_EXCEEDED
        return LIBMAD_STATUS_TIME_LIMIT
    elseif status == MadNLP.INTERNAL_ERROR
        return LIBMAD_STATUS_ERROR
    else
        return LIBMAD_STATUS_UNKNOWN
    end
end

function _libmad_write_result!(
    statuses::Ptr{Clonglong},
    objectives::Ptr{Cdouble},
    iterations::Ptr{Clonglong},
    j::Integer,
    status::Clonglong,
    objective::Float64,
    iter::Integer,
)
    statuses != C_NULL && unsafe_store!(statuses, status, j)
    objectives != C_NULL && unsafe_store!(objectives, Cdouble(objective), j)
    iterations != C_NULL && unsafe_store!(iterations, Clonglong(iter), j)
    return
end

function _libmad_termination_callback(terminate_check::Ptr{Cvoid}, terminate_data::Ptr{Cvoid})
    terminate_check == C_NULL && return nothing
    return _ -> ccall(terminate_check, Cint, (Ptr{Cvoid},), terminate_data) != 0
end

function _libmad_apply_branch_bounds_kernel!(
    lvar,
    uvar,
    infeasible,
    branch_var,
    branch_lower,
    branch_upper,
    nvar::Int32,
    batch_size::Int32,
)
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j > batch_size && return nothing
    @inbounds begin
        v = branch_var[j] + Int32(1)
        if v < Int32(1) || v > nvar
            infeasible[j] = UInt8(1)
            return nothing
        end
        lo = branch_lower[j]
        hi = branch_upper[j]
        if lo > hi
            infeasible[j] = UInt8(1)
            return nothing
        end
        new_lo = max(lvar[v, j], lo)
        new_hi = min(uvar[v, j], hi)
        if new_lo > new_hi
            infeasible[j] = UInt8(1)
        else
            lvar[v, j] = new_lo
            uvar[v, j] = new_hi
        end
    end
    return nothing
end

function _libmad_initial_point_kernel!(x0, lvar, uvar, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return nothing
    @inbounds begin
        lo = lvar[i]
        hi = uvar[i]
        x0[i] =
            isfinite(lo) && isfinite(hi) ? (lo + hi) / 2 :
            isfinite(lo) ? lo + one(lo) :
            isfinite(hi) ? hi - one(hi) :
            zero(lo)
    end
    return nothing
end

@static if Sys.isapple()
    function _libmad_branch_bounds_device(args...)
        error("libMad MadIPM batch device solve requires CUDA")
    end
else
    function _libmad_branch_bounds_device(
        lvar_base,
        uvar_base,
        branch_var::Vector{Int},
        branch_lower::Vector{Float64},
        branch_upper::Vector{Float64},
        nvar::Int,
        batch_size::Int,
    )
        lvar = repeat(reshape(lvar_base, nvar, 1), 1, batch_size)
        uvar = repeat(reshape(uvar_base, nvar, 1), 1, batch_size)
        infeasible = CUDA.zeros(UInt8, batch_size)
        bvar = CUDA.CuArray(Int32.(branch_var))
        blo = CUDA.CuArray(branch_lower)
        bup = CUDA.CuArray(branch_upper)
        threads = 256
        blocks = cld(batch_size, threads)
        blocks > 0 && CUDA.@cuda threads = threads blocks = blocks _libmad_apply_branch_bounds_kernel!(
            lvar, uvar, infeasible, bvar, blo, bup, Int32(nvar), Int32(batch_size))
        return lvar, uvar, infeasible
    end
end

function _libmad_scale_device_batch_lp(
    csr,
    objective_base,
    objective_scale::Float64,
    lcon_base,
    ucon_base,
    lvar,
    uvar,
    ncon::Int,
    nvar::Int,
    batch_size::Int,
)
    coo = CUDA.CUSPARSE.CuSparseMatrixCOO(csr)
    storage = BatchQuadraticModels.Scaling.SparseCOO(
        ncon, nvar, coo.rowInd, coo.colInd, copy(coo.nzVal))
    scaled, scaling = BatchQuadraticModels.Scaling.ruiz_equilibration(storage; strict = false)
    Dr, Dc = scaling.row, scaling.col

    scaled_coo = CUDA.CUSPARSE.CuSparseMatrixCOO(
        scaled.rowval, scaled.colval, scaled.nzval, (ncon, nvar))
    A = BatchQuadraticModels.sparse_operator(scaled_coo; spmm_ncols = batch_size)

    c = objective_base .* Dc
    objective_scale == 1.0 || (c .*= objective_scale)
    lcon = repeat(reshape(lcon_base .* Dr, ncon, 1), 1, batch_size)
    ucon = repeat(reshape(ucon_base .* Dr, ncon, 1), 1, batch_size)
    col_scale = reshape(Dc, nvar, 1)
    lvar ./= col_scale
    uvar ./= col_scale
    return A, c, lcon, ucon
end

@static if Sys.isapple()
    function _libmad_initial_point_device(args...)
        error("libMad MadIPM batch device solve requires CUDA")
    end
else
    function _libmad_initial_point_device(lvar, uvar)
        x0 = similar(lvar)
        n = length(x0)
        threads = 256
        blocks = cld(n, threads)
        blocks > 0 && CUDA.@cuda threads = threads blocks = blocks _libmad_initial_point_kernel!(
            x0, lvar, uvar, Int32(n))
        return x0
    end
end

function _libmad_solve_gpu_bnlp!(
    gpu_bnlp,
    infeasible,
    tol::Float64,
    time_limit::Float64,
    max_iter::Int,
    terminate_check::Ptr{Cvoid},
    terminate_data::Ptr{Cvoid},
    statuses_ptr::Ptr{Clonglong},
    objectives_ptr::Ptr{Cdouble},
    iterations_ptr::Ptr{Clonglong},
)
    solver = MadIPM.UniformBatchMPCSolver(
        gpu_bnlp;
        uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
        tol = tol > 0 ? tol : 1e-6,
        max_wall_time = time_limit > 0 ? time_limit : 1e6,
        max_iter = max_iter > 0 ? max_iter : 3000,
        print_level = MadNLP.ERROR,
        rethrow_error = false,
        check_batch_structure = false,
        fixed_variable_treatment = MadNLP.RelaxBound,
        regularization = MadIPM.FixedRegularization(1e-10, 1e-10),
        termination_callback = _libmad_termination_callback(terminate_check, terminate_data),
    )
    stats = MadIPM.solve!(solver; fetch_solution = false)
    objective = objectives_ptr == C_NULL ? nothing : Array(vec(solver.workspace.obj_val))
    need_results = statuses_ptr != C_NULL || objectives_ptr != C_NULL || iterations_ptr != C_NULL
    infeasible_cpu = need_results ? Array(infeasible) : nothing
    @inbounds for j in eachindex(stats.status)
        if infeasible_cpu !== nothing && !iszero(infeasible_cpu[j])
            _libmad_write_result!(
                statuses_ptr, objectives_ptr, iterations_ptr, j,
                LIBMAD_STATUS_INFEASIBLE, NaN, 0)
        else
            _libmad_write_result!(
                statuses_ptr, objectives_ptr, iterations_ptr, j,
                _libmad_status(stats.status[j]),
                objective === nothing ? NaN : objective[j],
                stats.iter[j])
        end
    end
    return
end
