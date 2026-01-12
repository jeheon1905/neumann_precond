from gospel.Eigensolver.CG import CG
from gospel.Eigensolver.Tucker import Tucker
from gospel.Eigensolver.Scipy import Scipy
from gospel.Eigensolver.Davidson import Davidson
from gospel.Eigensolver.ParallelDavidson import Davidson as ParallelDavidson
from gospel.Eigensolver.lobpcg import LOBPCG
from gospel.Eigensolver.Eigensolver import parallel_orthonormalize
from gospel.Eigensolver.Eigensolver import Eigensolver as BaseEigensolver

def create_eigensolver(params):
    """Create Eigensolver object from input parameters.

    :type  params: dict
    :param params:
        dictionary of eigensolver options

    *Example*
    >>> params={
    ...     "type": "davidson",
    ...     "maxiter": 10,
    ...     "locking": True,
    ... }
    >>> eigensolver = create_eigensolver(params)
    """
    supported_eigensolver = {
        "cg": CG,
        "scipy": Scipy,
        "tucker": Tucker,
        "davidson": Davidson,
        "parallel_davidson": ParallelDavidson,
        "lobpcg": LOBPCG,
    }
    if params is None:
        # eigensolver = LOBPCG()
        eigensolver = ParallelDavidson(locking=False)
        print(f"WARNING: Default eigensolver is set to ParallelDavidson.")
    elif isinstance(params, dict):
        _params = params.copy()
        eigensolver = supported_eigensolver[_params.pop("type").lower()](**_params)
    else:
        print("Custom eigensolver is used.")
        eigensolver = params
    return eigensolver
