"""Matrix solver wrappers."""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from functools import partial
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc

matsolvers = {}
def add_solver(solver):
    matsolvers[solver.__name__.lower()] = solver
    return solver


class SolverBase:
    """Abstract base class for all solvers."""

    config = {}
    def __init__(self, matrix, solver=None, *args, **kwargs):
        pass

    def solve(self, vector, *args, **kwargs):
        pass
    
    #def __init__(self, matrix, solver=None):
    #    pass

    #def solve(self, vector):
    #    pass


@add_solver
class DummySolver(SolverBase):
    """Dummy solver that returns zeros for testing."""

    def solve(self, vector):
        return 0 * vector


class SparseSolver(SolverBase):
    """Base class for sparse solvers."""
    sparse = True
    banded = False


class BandedSolver(SolverBase):
    """Base class for banded solvers."""
    sparse = False
    banded = True

    @staticmethod
    def sparse_to_banded(matrix, u=None, l=None):
        """Convert sparse matrix to banded format."""
        matrix = sp.dia_matrix(matrix)
        if u is None:
            u = max(0, max(matrix.offsets))
        if l is None:
            l = max(0, max(-matrix.offsets))
        ab = np.zeros((u+l+1, matrix.shape[1]), dtype=matrix.dtype)
        ab[u-matrix.offsets] = matrix.data
        lu = (l, u)
        return lu, ab


class DenseSolver(SolverBase):
    """Base class for dense solvers."""
    sparse = False
    banded = False


@add_solver
class UmfpackSpsolve(SparseSolver):
    """UMFPACK spsolve."""

    def __init__(self, matrix, solver=None):
        from scikits import umfpack
        self.matrix = matrix.copy()

    def update(self, matrix, solver=None):
        self.matrix = matrix.copy()
        
    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, use_umfpack=True)


@add_solver
class SuperluNaturalSpsolve(SparseSolver):
    """SuperLU+NATURAL spsolve."""

    def __init__(self, matrix, solver=None):
        self.matrix = matrix.copy()

    def update(self, matrix, solver=None):
        self.matrix = matrix.copy()
        
    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, permc_spec='NATURAL', use_umfpack=False)


@add_solver
class SuperluColamdSpsolve(SparseSolver):
    """SuperLU+COLAMD spsolve."""

    def __init__(self, matrix, solver=None):
        self.matrix = matrix.copy()

    def update(self, matrix, solver=None):
        self.matrix = matrix.copy()
        
    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, permc_spec='COLAMD', use_umfpack=False)


@add_solver
class UmfpackFactorized(SparseSolver):
    """UMFPACK LU factorized solve."""

    def __init__(self, matrix, solver=None):
        from scikits import umfpack
        self.LU = spla.factorized(matrix.tocsc())

    def update(self, matrix, solver=None):
        self.LU = spla.factorized(matrix.tocsc())
        
    def solve(self, vector):
        return self.LU(vector)


@add_solver
class SuperluNaturalFactorized(SparseSolver):
    """SuperLU+NATURAL LU factorized solve."""

    def __init__(self, matrix, solver=None):
        self.LU = spla.splu(matrix.tocsc(), permc_spec='NATURAL')

    def update(self, matrix, solver=None):
        self.LU = spla.splu(matrix.tocsc(), permc_spec='NATURAL')
        
    def solve(self, vector):
        return self.LU.solve(vector)


@add_solver
class SuperluNaturalFactorizedTranspose(SparseSolver):
    """SuperLU+NATURAL LU factorized solve."""

    def __init__(self, matrix, solver=None):
        self.LU = spla.splu(matrix.T.tocsc(), permc_spec='NATURAL')

    def update(self, matrix, solver=None):
        self.LU = spla.splu(matrix.T.tocsc(), permc_spec='NATURAL')
        
    def solve(self, vector):
        return self.LU.solve(vector, trans='T')


@add_solver
class SuperluColamdFactorized(SparseSolver):
    """SuperLU+COLAMD LU factorized solve."""

    def __init__(self, matrix, solver=None):
        self.LU = spla.splu(matrix.tocsc(), permc_spec='COLAMD')

    def update(self, matrix, solver=None):
        self.LU = spla.splu(matrix.tocsc(), permc_spec='COLAMD')

    def solve(self, vector):
        return self.LU.solve(vector)


@add_solver
class ScipyBanded(BandedSolver):
    """Scipy banded solve."""

    def __init__(self, matrix, solver=None):
        self.lu, self.ab = self.sparse_to_banded(matrix)

    def update(self, matrix, solver=None):
        self.lu, self.ab = self.sparse_to_banded(matrix)
        
    def solve(self, vector):
        return sla.solve_banded(self.lu, self.ab, vector, check_finite=False)


@add_solver
class SPQR_solve(SparseSolver):
    """SuiteSparse QR solve."""

    def __init__(self, matrix, solver=None):
        import sparseqr
        self.matrix = matrix.copy()

    def update(self, matrix, solver=None):
        self.matrix = matrix.copy()
        
    def solve(self, vector):
        return sparseqr.solve(self.matrix, vector)


@add_solver
class BandedQR(BandedSolver):
    """pybanded QR solve."""

    def __init__(self, matrix, solver=None):
        import pybanded
        matrix = pybanded.BandedMatrix.from_sparse(matrix)
        self.QR = pybanded.BandedQR(matrix)

    def update(self, matrix, solver=None):
        matrix = pybanded.BandedMatrix.from_sparse(matrix)
        self.QR = pybanded.BandedQR(matrix)
        
    def solve(self, vector):
        return self.QR.solve(vector)


@add_solver
class SparseInverse(SparseSolver):
    """Sparse inversion solve."""

    def __init__(self, matrix, solver=None):
        self.matrix_inverse = spla.inv(matrix.tocsc())

    def update(self, matrix, solver=None):
        self.matrix_inverse = sla.inv(matrix.A)
        
    def solve(self, vector):
        return self.matrix_inverse @ vector


@add_solver
class DenseInverse(DenseSolver):
    """Dense inversion solve."""

    def __init__(self, matrix, solver=None):
        self.matrix_inverse = sla.inv(matrix.A)

    def update(self, matrix, solver=None):
        self.matrix_inverse = sla.inv(matrix.A)
        
    def solve(self, vector):
        return self.matrix_inverse @ vector


@add_solver
class BlockInverse(BandedSolver):
    """Block inversion solve."""

    def __init__(self, matrix, solver):
        from dedalus.tools.sparse import same_dense_block_diag
        # Check separability
        if solver.domain.bases[-1].coupled:
            raise ValueError("Block solver requires uncoupled problems.")
        block_size = b = len(solver.problem.variables)
        # Produce inverse
        if block_size == 1:
            # Special-case diagonal matrices
            self.matrix_inv_diagonal = 1 / matrix.todia().data[0]
            self.solve = self._solve_diag
        else:
            # Covert to BSR to extract blocks
            bsr_matrix = matrix.tobsr(blocksize=(b, b))
            # Compute block inverses
            inv_blocks = np.linalg.inv(bsr_matrix.data)
            self.matrix_inverse = same_dense_block_diag(list(inv_blocks), format='csr')
            self.solve = self._solve_block

    def update(self, matrix, solver):
        # Check separability
        if solver.domain.bases[-1].coupled:
            raise ValueError("Block solver requires uncoupled problems.")
        block_size = b = len(solver.problem.variables)
        # Produce inverse
        if block_size == 1:
            # Special-case diagonal matrices
            self.matrix_inv_diagonal = 1 / matrix.todia().data[0]
            self.solve = self._solve_diag
        else:
            # Covert to BSR to extract blocks
            bsr_matrix = matrix.tobsr(blocksize=(b, b))
            # Compute block inverses
            inv_blocks = np.linalg.inv(bsr_matrix.data)
            self.matrix_inverse = same_dense_block_diag(list(inv_blocks), format='csr')
            self.solve = self._solve_block
            
    def _solve_block(self, vector):
        return self.matrix_inverse @ vector

    def _solve_diag(self, vector):
        return self.matrix_inv_diagonal * vector

@add_solver
class ScipyDenseLU(DenseSolver):
    """Scipy dense LU factorized solve."""

    def __init__(self, matrix, solver=None):
        self.LU = sla.lu_factor(matrix.A, check_finite=False)

    def update(self, matrix, solver=None):
        self.LU = sla.lu_factor(matrix.A, check_finite=False)

    def solve(self, vector):
        return sla.lu_solve(self.LU, vector, check_finite=False)

@add_solver
class PETScSparseSolver(SparseSolver):

    def __init__(self, matrix, solver=None):
        from ..tools.config import config
        solver_type=config['linear algebra']['MATRIX_EIGENSOLVER']
        params=config['linear algebra']['MATRIX_EIGENSOLVER_PARAMS']
        #No multinode comm for now, until I figure out how to pass it in.
        self.comm = MPI.COMM_SELF
        self.opts = PETSc.Options()
        if (params!=None):
            #logger.debug("Setting solver parameter(s) "+params)
            tmp = params[1:-1]
            tmp = tmp.split(" ")
            nterms = len(tmp)
            names = tmp[0:nterms:2]
            vals = tmp[1:nterms:2]
            nparams = len(names)
            icntls=[]
            cntls=[]
            #set_omp_threads=1
            for kk in range(nparams):
                tmp = names[kk].split("-")
                name = tmp[1]
                tmp = name.split("_")
                if (solver_type=='SlepcMumps'):
                    if (tmp[1]=='mumps' and tmp[2]=='icntl'):
                        self.opts[name]=int(vals[kk])
                    if (tmp[1]=='mumps' and tmp[2]=='cntl'):
                        self.opts[name]=float(vals[kk])
                elif (solver_type=='SlepcSuperlu_dist'):
                    self.opts[name]=vals[kk]
                elif (solver_type=='SlepcSuperlu'):
                    if (tmp[2]=='ilu'):
                        self.opts[name]=float(vals[kk])
                    else:
                        self.opts[name]=vals[kk]

        #Setup matrices in petsc format
        self.Ap = PETSc.Mat().createAIJ(size=matrix.shape,csr=(matrix.indptr, matrix.indices,matrix.data),comm=self.comm)
        self.Ap.assemble()
        matrix = np.zeros(1,dtype=int)

        #logger.info("Setting up solver")
        self.KSP = PETSc.KSP().create(comm=self.comm)
        self.KSP.setFromOptions()
        self.KSP.setType('preonly')
        self.KSP.setOperators(self.Ap)

        #logger.info("Setting up LU Preconditioner")
        self.PC = self.KSP.getPC()
        self.PC.setFromOptions()
        self.PC.setType('lu')
        if (solver_type=='SlepcMumps'):
            self.PC.setFactorSolverType('mumps')
        elif (solver_type=='SlepcSuperlu_dist'):
            self.PC.setFactorSolverType('superlu_dist')
        elif (solver_type=='SlepcSuperlu'):
            self.PC.setFactorSolverType('superlu')
        else:
            raise NotImplementedError("PETSc solver type not implemented.")

        #logger.info("Setting up LU matrix")
        self.PC.setFactorSetUpSolverType()
        self.K = self.PC.getFactorMatrix()
        self.K.setFromOptions()

        self.sol = self.Ap.createVecRight()
        self.rhs = self.Ap.createVecLeft()
        tmp = self.sol.getArray()
        tsize = len(tmp)
        tdtype = tmp.dtype
        tmp = 0
        self.out = np.zeros((tsize,1),dtype=tdtype)

    def update(self,matrix,solver=None):
        self.Ap.destroy()
        self.Ap = PETSc.Mat().createAIJ(size=matrix.shape,csr=(matrix.indptr, matrix.indices,matrix.data),comm=self.comm)
        self.Ap.assemble()
        matrix = np.zeros(1,dtype=int)
        self.KSP.setOperators(self.Ap)
        self.sol = self.Ap.createVecRight()
        self.rhs = self.Ap.createVecLeft()
    
    def solve(self,vector):
        self.rhs.setArray(vector)
        self.KSP.solve(self.rhs,self.sol)
        self.out[:,0] = self.sol.getArray()
        return self.out

class Woodbury(SparseSolver):
    """Solve top & right bordered matrix using Woodbury formula."""

    config = {'bc_top': True}

    def __init__(self, matrix, subproblem, matsolver):
        self.matrix = matrix
        self.matsolver = matsolver
        self.update_rank = R = subproblem.update_rank
        # Form Woodbury factors
        self.U = U = np.zeros((matrix.shape[0], 2*R), dtype=matrix.dtype)
        self.V = V = np.zeros((2*R, matrix.shape[1]), dtype=matrix.dtype)
        # Remove top border, leaving upper left subblock
        U[:R, :R] = np.identity(R)
        V[:R, R:] = matrix[:R, R:].A
        # Remove right border, leaving upper right and lower right subblocks
        U[R:-R, R:] = matrix[R:-R, -R:].A
        V[-R:, -R:] = np.identity(R)
        self.A = matrix - sp.csr_matrix(U) @ sp.csr_matrix(V)
        # Solve A using specified matsolver
        self.A_matsolver = matsolver(self.A)
        self.Ainv = self.A_matsolver.solve
        self.Ainv_U = self.Ainv(U)
        # Solve S using scipy dense inverse
        S = np.identity(2*R) + V @ self.Ainv_U
        self.Sinv_ = sla.inv(S)
        self.Sinv = lambda Y, Sinv=self.Sinv_: Sinv @ Y

    def solve(self, Y):
        Ainv_Y = self.Ainv(Y)
        return Ainv_Y - self.Ainv_U @ self.Sinv(self.V @ Ainv_Y)


woodbury_matsolvers = {}
for name, matsolver in matsolvers.items():
    woodbury_matsolvers['woodbury' + name] = partial(Woodbury, matsolver=matsolver)
matsolvers.update(woodbury_matsolvers)

