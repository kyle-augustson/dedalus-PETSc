"""Tools for array manipulations."""

import numpy as np
from scipy import sparse
import scipy.sparse as sp
from scipy.sparse import _sparsetools
from scipy.sparse import linalg as spla
from functools import reduce
import operator
from ..tools.config import config

SPLIT_CSR_MATVECS = config['linear algebra'].getboolean('SPLIT_CSR_MATVECS')


def prod(arg):
    if arg:
        return reduce(operator.mul, arg)
    else:
        return 1


def interleaved_view(data):
    """
    View n-dim complex array as (n+1)-dim real array, where the last axis
    separates real and imaginary parts.
    """
    # Check datatype
    if data.dtype != np.complex128:
        raise ValueError("Complex array required.")
    # Create view array
    iv_shape = data.shape + (2,)
    iv = np.ndarray(iv_shape, dtype=np.float64, buffer=data.data)
    return iv


def reshape_vector(data, dim=2, axis=-1):
    """Reshape 1-dim array as a multidimensional vector."""
    # Build multidimensional shape
    shape = [1] * dim
    shape[axis] = data.size
    return data.reshape(shape)


def axindex(axis, index):
    """Index array along specified axis."""
    if axis < 0:
        raise ValueError("`axis` must be positive")
    # Add empty slices for leading axes
    return (slice(None),)*axis + (index,)


def axslice(axis, start, stop, step=None):
    """Slice array along a specified axis."""
    return axindex(axis, slice(start, stop, step))


def zeros_with_pattern(*args):
    """Create sparse matrix with the combined pattern of other sparse matrices."""
    # Join individual patterns in COO format
    coo = [A.tocoo() for A in args]
    rows = np.concatenate([A.row for A in coo])
    cols = np.concatenate([A.col for A in coo])
    shape = coo[0].shape
    # Create new COO matrix with zeroed data and combined pattern
    data = np.concatenate([A.data*0 for A in coo])
    return sparse.coo_matrix((data, (rows, cols)), shape=shape)


def expand_pattern(input, pattern):
    """Return copy of sparse matrix with extended pattern."""
    # Join input and pattern in COO format
    A = input.tocoo()
    P = pattern.tocoo()
    rows = np.concatenate((A.row, P.row))
    cols = np.concatenate((A.col, P.col))
    shape = A.shape
    # Create new COO matrix with expanded data and combined pattern
    data = np.concatenate((A.data, P.data*0))
    return sparse.coo_matrix((data, (rows, cols)), shape=shape)


def apply_matrix(matrix, array, axis, **kw):
    """Apply matrix along any axis of an array."""
    if sparse.isspmatrix(matrix):
        return apply_sparse(matrix, array, axis, **kw)
    else:
        return apply_dense(matrix, array, axis, **kw)


def apply_dense_einsum(matrix, array, axis, optimize=True, **kw):
    """Apply dense matrix along any axis of an array."""
    dim = len(array.shape)
    # Build Einstein signatures
    mat_sig = [dim, axis]
    arr_sig = list(range(dim))
    out_sig = list(range(dim))
    out_sig[axis] = dim
    out = np.einsum(matrix, mat_sig, array, arr_sig, out_sig, optimize=optimize, **kw)
    return out


def move_single_axis(a, source, destination):
    """Similar to np.moveaxis but faster for just a single axis."""
    order = [n for n in range(a.ndim) if n != source]
    order.insert(destination, source)
    return a.transpose(order)


def apply_dense(matrix, array, axis, out=None):
    """Apply dense matrix along any axis of an array."""
    dim = array.ndim
    # Resolve wraparound axis
    axis = axis % dim
    # Move axis to 0
    if axis != 0:
        array = move_single_axis(array, axis, 0) # May allocate copy
    # Flatten later axes
    if dim > 2:
        array_shape = array.shape
        array = array.reshape((array_shape[0], -1)) # May allocate copy
    # Apply matmul
    temp = np.matmul(matrix, array) # Allocates temp
    # Unflatten later axes
    if dim > 2:
        temp = temp.reshape((temp.shape[0],) + array_shape[1:]) # View
    # Move axis back from 0
    if axis != 0:
        temp = move_single_axis(temp, 0, axis) # View
    # Return
    if out is None:
        return temp
    else:
        out[:] = temp # Copy
        return out


def splu_inverse(matrix, permc_spec="NATURAL", **kw):
    """Create LinearOperator implicitly acting as a sparse matrix inverse."""
    splu = spla.splu(matrix.tocsc(), permc_spec=permc_spec, **kw)
    def solve(x):
        if np.iscomplexobj(x) and matrix.dtype == np.float64:
            return splu.solve(x.real) + 1j*splu.solve(x.imag)
        else:
            return splu.solve(x)
    return spla.LinearOperator(shape=matrix.shape, dtype=matrix.dtype, matvec=solve, matmat=solve)


def apply_sparse(matrix, array, axis, out=None):
    """Apply sparse matrix along any axis of an array."""
    dim = array.ndim
    # Resolve wraparound axis
    axis = axis % dim
    # Move axis to 0
    if axis != 0:
        array = move_single_axis(array, axis, 0) # May allocate copy
    # Flatten later axes
    if dim > 2:
        array_shape = array.shape
        array = array.reshape((array_shape[0], -1)) # May allocate copy
    # Apply matmul
    temp = matrix.dot(array) # Allocates temp
    # Unflatten later axes
    if dim > 2:
        temp = temp.reshape((temp.shape[0],) + array_shape[1:]) # View
    # Move axis back from 0
    if axis != 0:
        temp = move_single_axis(temp, 0, axis) # View
    # Return
    if out is None:
        return temp
    else:
        out[:] = temp # Copy
        return out


def csr_matvec(A_csr, x_vec, out_vec):
    """
    Fast CSR matvec with dense vector skipping output allocation. The result is
    added to the specificed output array, so the output should be manually
    zeroed prior to calling this routine, if necessary.
    """
    # Check format but don't convert
    if A_csr.format != "csr":
        raise ValueError("Matrix must be in CSR format.")
    # Check shapes
    M, N = A_csr.shape
    m, n = out_vec.size, x_vec.size
    if x_vec.ndim > 1 or out_vec.ndim > 1:
        raise ValueError("Only vectors allowed for input and output.")
    if M != m or N != n:
        raise ValueError(f"Matrix shape {(M,N)} does not match input {(n,)} and output {(m,)} shapes.")
    # Apply matvec
    _sparsetools.csr_matvec(M, N, A_csr.indptr, A_csr.indices, A_csr.data, x_vec, out_vec)
    return out_vec


def csr_matvecs(A_csr, x_vec, out_vec):
    """
    Fast CSR matvec with dense vector skipping output allocation. The result is
    added to the specificed output array, so the output should be manually
    zeroed prior to calling this routine, if necessary.
    """
    # Check format but don't convert
    if A_csr.format != "csr":
        raise ValueError("Matrix must be in CSR format.")
    # Check shapes
    M, N = A_csr.shape
    n, kx = x_vec.shape
    m, ko = out_vec.shape
    if x_vec.ndim != 2 or out_vec.ndim != 2:
        raise ValueError("Only matrices allowed for input and output.")
    if M != m or N != n:
        raise ValueError(f"Matrix shape {(M,N)} does not match input {(n,)} and output {(m,)} shapes.")
    if kx != ko:
        raise ValueError("Output size does not match input size.")
    # Apply matvecs
    if SPLIT_CSR_MATVECS:
        for k in range(kx):
            _sparsetools.csr_matvec(M, N, A_csr.indptr, A_csr.indices, A_csr.data, x_vec[:,k], out_vec[:,k])
    else:
        _sparsetools.csr_matvecs(M, N, kx, A_csr.indptr, A_csr.indices, A_csr.data, x_vec, out_vec)
    return out_vec


def add_sparse(A, B):
    """Add sparse matrices, promoting scalars to multiples of the identity."""
    A_is_scalar = np.isscalar(A)
    B_is_scalar = np.isscalar(B)
    if A_is_scalar and B_is_scalar:
        return A + B
    elif A_is_scalar:
        I = sparse.eye(*B.shape, dtype=B.dtype, format=B.format)
        return A*I + B
    elif B_is_scalar:
        I = sparse.eye(*A.shape, dtype=A.dtype, format=A.format)
        return A + B*I
    else:
        return A + B


def sparse_block_diag(blocks, shape=None):
    """Build a block diagonal sparse matrix allowing size 0 matrices."""
    # Collect subblocks
    data, rows, cols = [], [], []
    i0, j0 = 0, 0
    for block in blocks:
        block = sparse.coo_matrix(block)
        if block.nnz > 0:
            data.append(block.data)
            rows.append(block.row + i0)
            cols.append(block.col + j0)
        i0 += block.shape[0]
        j0 += block.shape[1]
    # Build full matrix
    if shape is None:
        shape = (i0, j0)
    if data:
        data = np.concatenate(data)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        return sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    else:
        return sparse.csr_matrix(shape)


def kron(*factors):
    if factors:
        out = factors[0]
        for f in factors[1:]:
            out = np.kron(out, f)
    else:
        out = np.identity(1)
    return out


def nkron(factor, n):
    return kron(*[factor for i in range(n)])


def permute_axis(array, axis, permutation, out=None):
    # OPTIMIZE: currently creates a temporary
    slices = [slice(None) for i in array.shape]
    slices[axis] = permutation
    perm = array[tuple(slices)]
    if out is None:
        return perm
    else:
        np.copyto(out, perm)
        return out


def copyto(dest, src):
    # Seems to be faster than np.copyto
    dest[:] = src


def perm_matrix(perm, M=None, source_index=False, sparse=True):
    """
    Build sparse permutation matrix from permutation vector.

    Parameters
    ----------
    perm : ndarray
        Permutation vector.
    M : int, optional
        Output dimension. Default: len(perm).
    source_index : bool, optional
        False (default) if perm entries indicate destination index:
            output[i,j] = (i == perm[j])
        True if perm entires indicate source index:
            output[i,j] = (j == perm[i])
    sparse : bool, optional
        Whether to return sparse matrix or dense array (default: True).
    """
    N = len(perm)
    if M is None:
        M = N
    if source_index:
        row = np.arange(N)
        col = np.array(perm)
    else:
        row = np.array(perm)
        col = np.arange(N)
    if sparse:
        data = np.ones(N, dtype=int)
        return sp.coo_matrix((data, (row, col)), shape=(M, N))
    else:
        output = np.zeros((M, N), dtype=int)
        output[row, col] = 1
        return output


def drop_empty_rows(mat):
    mat = sparse.csr_matrix(mat)
    nonempty_rows = (np.diff(mat.indptr) > 0)
    return mat[nonempty_rows]


def scipy_sparse_eigs(A, B, N, target, matsolver, eigv, **kw):
    """
    Perform targeted eigenmode search using the scipy/ARPACK sparse solver
    for the reformulated generalized eigenvalue problem

        A.x = λ B.x  ==>  (A - σB)^I B.x = (1/(λ-σ)) x

    for eigenvalues λ near the target σ.

    Parameters
    ----------
    A, B : scipy sparse matrices
        Sparse matrices for generalized eigenvalue problem
    N : int
        Number of eigenmodes to return
    target : complex
        Target σ for eigenvalue search
    matsolver : matrix solver class
        Class implementing solve method for solving sparse systems.

    Other keyword options passed to scipy.sparse.linalg.eigs.
    """
    # Build sparse linear operator representing (A - σB)^I B = C^I B = D
    C = A - target * B
    solver = matsolver(C)
    def matvec(x):
        return solver.solve(B.dot(x))
    D = spla.LinearOperator(dtype=A.dtype, shape=A.shape, matvec=matvec)
    # Solve using scipy sparse algorithm
    evals, evecs = spla.eigs(D, k=N, which='LM', sigma=None, v0=eigv, **kw)
    # Rectify eigenvalues
    evals = 1 / evals + target
    return evals, evecs

def slepc_target_wrapper(comm,A, B, N, target, eigv, solver_type, params, **kw):
    """                                                                                                                               
    Perform targeted eigenmode search using the SLEPc parallel sparse solver                                                          
    for the reformulated generalized eigenvalue problem                                                                               
                                                                                                                                      
        A.x = λ B.x  ==>  (A - σB)^I B.x = (1/(λ-σ)) x                                                                                
                                                                                                                                      
    for eigenvalues λ near the target σ.                                                                                              
                                                                                                                                      
    Parameters                                                                                                                        
    ----------                                                                                                                        
    A, B : scipy sparse matrices                                                                                                      
        Sparse matrices for generalized eigenvalue problem                                                                            
    N : int                                                                                                                           
        Number of eigenmodes to return                                                                                                
    target : complex                                                                                                                  
        Target σ for eigenvalue search                                                                                                
                                                                                                                                      
    Other keyword options passed to scipy.sparse.linalg.eigs.                                                                         
    """
    import os
    import sys
    from mpi4py import MPI
    import petsc4py
    import slepc4py
    from petsc4py import PETSc
    from slepc4py import SLEPc

        #Set options using opts.set_target_option(value)                                                                                  
    print("Setting Parameters in SLEPc")
    #petsc4py.init(comm=comm)                                                                                                         
    opts = PETSc.Options()
    if (params!=None):
        print("Setting solver parameter(s) "+params)
        #These DO NOT work, SLEPc & PETSc team should be contacted.                                                                   
        #petsc4py.init(params)                                                                                                        
        #slepc4py.init(params)                                                                                                        
        #Parsing param string                                                                                                         
        tmp = params[1:-1]
        tmp = tmp.split(" ")
        nterms = len(tmp)
        names = tmp[0:nterms:2]
        vals = tmp[1:nterms:2]
        nparams = len(names)
        icntls=[]
        set_omp_threads=1
        for kk in range(nparams):
            tmp = names[kk].split("-")
            name = tmp[1]
            tmp = name.split("_")
            if (tmp[1]=='mumps' and tmp[2]=='icntl'):
                icntls.append([int(tmp[3]),int(vals[kk])])

            if (tmp[1]=='mumps' and tmp[3]=='omp'):
               set_omp_threads=int(vals[kk])
            name = name[4:]

    # create SLEPc Eigenvalue solver                                                                                                  

    #Setup matrices in petsc format                                                                                                   
    print("Creating global matrix A")
    Ap = PETSc.Mat().createAIJ(size=A.shape,csr=(A.indptr, A.indices,A.data),comm=comm)
    Ap.assemble()
    A=0

    #Setup matrices in petsc format                                                                                                   
    print("Creating global matrix B")
    Bp = PETSc.Mat().createAIJ(size=B.shape,csr=(B.indptr, B.indices,B.data),comm=comm)
    Bp.assemble()
    B=0

    #Setup eigensolver                                                                                                                
    logger.info("Setting up SLEPc eigensolver")
    E = SLEPc.EPS().create(comm)
    E.setOperators(Ap,Bp)
    E.setFromOptions()
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)    # generalized non-Hermitian eigenvalue problem                                   

    #E.setType('krylovschur')                                                                                                         
    solverType = 'krylovschur' #'arnoldi'                                                                                             
    E.setType(solverType)
    E.setDimensions(N,PETSc.DECIDE)     # set number of  eigenvalues to compute                                                       
    E.setTrueResidual(True)
    Niter = 64
    epsilon = 1e-7
    E.setTolerances(epsilon,Niter) # Set absolute tolerance and number of iterations                                                  

    if target is not None:
        E.setTarget(target)     # set the desired eigenvalue                                                                          
        if (np.abs(target)<1e0):
            E.setConvergenceTest(E.Conv.ABS)
        else:
            E.setConvergenceTest(E.Conv.REL)

    #target = evals[0]                                                                                                                
    E.setTarget(target)     # set the desired eigenvalue                                                                              
    if (np.abs(target)<1e0):
        E.setConvergenceTest(E.Conv.ABS)
    else:
        E.setConvergenceTest(E.Conv.REL)

    if eigv is not None:
        vr, wr = Ap.getVecs()
        tmp = vr.getArray()
        tmp[:] = eigv
        vr.setArray(tmp)
	E.setInitialSpace(vr)  # set the initial vector                                                                               

    E.setWhichEigenpairs(E.Which.TARGET_IMAGINARY)
    logger.info("Setting up Spectral Transformer")
    ST = E.getST()
    ST.setFromOptions()
    ST.setType('sinvert')
    ST.setUp()

    logger.info("Setting up " + solverType +  " solver")
    KSP = ST.getKSP()
    KSP.setFromOptions()
    KSP.setType('preonly')

    logger.info("Setting up LU Preconditioner")
    PC =  KSP.getPC()
    PC.setFromOptions()
    PC.setType('lu')
    if (solver_type=='SlepcMumps'):
        PC.setFactorSolverType('mumps')
    elif (solver_type=='SlepcSuperlu_dist'):
        PC.setFactorSolverType('superlu_dist')
	#PC.setFactorShift(shift_type='nonzero',amount=0.1)                                                                           
    else:
        raise NotImplementedError("SLEPc solver type not implemented.")

    logger.info("Setting up LU matrix")
    ST.getOperator()
    PC.setFactorSetUpSolverType()
    K = PC.getFactorMatrix()
    K.setFromOptions()
    if (solver_type=='SlepcMumps'):
        for kk in range(len(icntls)):
            K.setMumpsIcntl(icntls[kk][0],icntls[kk][1])
    
    ST.restoreOperator(K)
    logger.info("Solving")
    E.solve()

    eps_type = E.getType()
    print("Solution method: %s" % eps_type)
    its = E.getIterationNumber()
    print("Number of iterations of the method: %d" % its)

    nev, ncv, mpd = E.getDimensions()
    print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = E.getTolerances()
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    nconv = E.getConverged()
    print("Number of converged eigenpairs %d" % nconv)

    if (nconv > 0):
        # Create the results vectors                                                                                                  
	vr, wr = Ap.getVecs()
        #                                                                                                                             
        print(" ")
        print("        k          ||Ax-kBx||/||kBx|| ")
        print("----------------- ------------------")
        evals = np.zeros(N,dtype=np.complex128)
        #print(evals[0])                                                                                                              
        evecs = np.zeros((vr.getSize(),N),dtype=np.complex128)
        errs = np.zeros(N,dtype=np.float64)
        for i in range(np.min([nconv,N])):
            k = E.getEigenpair(i, vr, wr)
            #print(k.real, k.imag)                                                                                                    
            evals[i] = k.real + 1j*k.imag
            evecs[:,i] = vr.getArray()+1j*wr.getArray()
            error = E.computeError(i)
            errs[i] = error
            if k.imag != 0.0:
                print(" %9f%+9f j %12g" % (k.real, k.imag, error))
            else:
                print(" %12f      %12g" % (k.real, error))
            print(" ")
    else:
        print("Solver did not converge. Try adjusting solver parameters, or increase resolution and try again.")
        evals = np.zeros(N,dtype=np.complex128)
        errs = np.zeros(N,dtype=np.complex128)
        evecs = np.zeros((vr.getSize(),N),dtype=np.complex128)

    K.destroy()
    PC.destroy()
    KSP.destroy()
    ST.destroy()
    E.destroy()
    Ap.destroy()
    Bp.destroy()
    return evals, evecs, errs, its, nconv
        
def interleave_matrices(matrices):
    N = len(matrices)
    if N == 1:
        return matrices[0]
    sum = 0
    P = sparse.lil_matrix((N, N))
    for i, matrix in enumerate(matrices):
        P[i, i] = 1
        sum += sparse.kron(matrix, P)
        P[i, i] = 0
    return sum

