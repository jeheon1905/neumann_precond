import torch
import math
import numpy as np

boltzmann_factor = 3.1668114 * 1e-6  # kelvin/hartree


def fermi_dirac_smearing_function(energy_diff, temperature, order=None):
    if temperature < 1e-7:
        beta = torch.inf
    else:
        beta = 1 / (boltzmann_factor * temperature)
    val = torch.nan_to_num(energy_diff * beta)
    return 1 / (torch.exp(val) + 1)


def gaussian_smearing_function(energy_diff, temperature, order=None):
    if temperature < 1e-7:
        beta = torch.inf
    else:
        beta = 1 / (boltzmann_factor * temperature)
    val = torch.nan_to_num(energy_diff * beta)
    return 0.5 * (1 - torch.erf(val))


def cold_smearing_function(energy_diff, temperature, order=None):
    if temperature < 1e-7:
        beta = torch.inf
    else:
        beta = 1 / (boltzmann_factor * temperature)
    val = torch.nan_to_num(energy_diff * beta)
    return 0.5 * (
        math.sqrt(2 / math.pi) * torch.exp(-(val**2) - math.sqrt(2) * val - 0.5)
        + 1
        - torch.erf(val + 1 / math.sqrt(2))
    )


def mp_smearing_function(energy_diff, temperature, order=2):
    from scipy.special import eval_hermite

    if temperature < 1e-7:
        beta = torch.inf
    else:
        beta = 1 / (boltzmann_factor * temperature)
    val = torch.nan_to_num(energy_diff * beta)
    ret_val = 0.5 * (1 - torch.erf(val))
    for m in range(1, order + 1):
        A_m = (-1) ** m / (math.factorial(m) * 4**m * math.sqrt(math.pi))
        H_2m_1 = eval_hermite(2 * m - 1, val)
        ret_val += A_m * H_2m_1 * torch.exp(-(val**2))
    return ret_val


if __name__ == "__main__":
    eigval = torch.arange(10) * 0.1
    fermi_level = 0.0
    e_diff = eigval - fermi_level
    temp = 1160.0
    fermi_dirac_smearing_function(e_diff, temp)
    gaussian_smearing_function(e_diff, temp)
    cold_smearing_function(e_diff, temp)
    mp_smearing_function(e_diff, temp)
