from gospel.util import timer
from gospel.Poisson.ISF_solver import ISF_solver
from gospel.ParallelHelper import ParallelHelper as PH


@timer
def create_poisson(grid, params=None, device=None):
    """Create Poisson_solver class object from input parameters.
    :type  grid: gospel.Grid
    :param grid:
        Grid class object
    :type  params: dict or None, optional
    :param params:
        parameters of Poisson_solver
    :type  device: torch.device, optional
    :param device:
        device to use for computation, defaults to None

    :rtype: gospel.Poisson.Poisson_solver
    :return:
        Poisson_solver class object

    **Example**

    >>> # create ISF type poisson solver
    >>> from ase.build import bulk
    >>> from gospel.Grid import grid
    >>> from gospel.Poisson import create_poisson
    >>> atoms = bulk('Si', 'fcc', a=5.43)
    >>> grid = Grid(atoms, spacing=0.2)
    >>> params = {'type':'ISF', 'ISF_cutoff':None}
    >>> poisson_solver = create_poisson(grid, params)
    """
    supported_poisson = {"ISF": ISF_solver}
    if params is None:
        poisson_solver = ISF_solver(grid, device=device)
    elif isinstance(params, dict):
        if params.get("no_poisson") == True:
            poisson_solver = None
        else:
            params['device'] = device
            poisson_solver = supported_poisson[params.pop("type")](**params)
    else:
        if type(params) in supported_poisson.values():
            poisson_solver = params
        else:
            assert (
                False
            ), "An input parameter for poisson should be dictionary including parameters or directly defined object."
    return poisson_solver
