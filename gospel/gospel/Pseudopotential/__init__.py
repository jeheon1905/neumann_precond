from gospel.Pseudopotential.NCPP import NCPP
from gospel.Pseudopotential.LUSP import LUSP
from gospel.Pseudopotential.UPF import UPF
from gospel.util import Timer


@Timer.timeit
def create_Pseudopotential(grid, params, kpts=[[0, 0, 0]], device=None):
    """Create Pseudopotential object.

    :type  grid: gospel.Grid
    :param grid:
        Grid object
    :type  params: dict
    :param params:
        dictionary containing input parameters
    :type  kpts: list[list[float]], optional
    :param kpts:
        list of k-points, defaults to [[0, 0, 0]]
    :type device: torch.device, optional
    :param device:
        device to use for computation, defaults to None

    :rtype: gospel.Pseudopotential
    :return:
        Pseudopotential object
    """
    _params = params.copy()

    if _params.get("upf") is None:
        assert 0, "No 'upf' file paths."

    upf = UPF(_params.pop("upf"), device=device)
    if True in grid.get_pbc():
        assert _params.get("use_comp"), "use_comp is necessary for PBC calculation."

    if upf.pseudo_type == "NC":
        pp = NCPP(grid, upf, kpts=kpts, device=device, **_params)
    elif upf.pseudo_type == "LUSP":
        if "use_dense_proj" in _params:
            _params.pop("use_dense_proj")
        pp = LUSP(grid, upf, kpts=kpts, **_params)
    else:
        raise NotImplementedError(
            f"{upf.pseudo_type} is not supported pseudopotential type"
        )
    return pp
