from typing import Tuple
import numpy as np
import torch


# Precomputed coefficients for real spherical harmonics Y_l^m(θ, φ)
# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
coef_list = [
    [0.5 * np.sqrt(1 / np.pi)],
    [0.4886025119029199, 0.4886025119029199, 0.4886025119029199],  # sqrt(3 / (4 * pi))
    [
        0.5 * np.sqrt(15 / np.pi),
        0.5 * np.sqrt(15 / np.pi),
        0.25 * np.sqrt(5 / np.pi),
        0.5 * np.sqrt(15 / np.pi),
        0.25 * np.sqrt(15 / np.pi),
    ],
    [
        0.25 * np.sqrt(35 / (2 * np.pi)),
        0.5 * np.sqrt(105 / np.pi),
        0.25 * np.sqrt(21 / 2 / np.pi),
        0.25 * np.sqrt(7 / np.pi),
        0.25 * np.sqrt(21 / 2 / np.pi),
        0.25 * np.sqrt(105 / np.pi),
        0.25 * np.sqrt(35 / (2 * np.pi)),
    ],
]


def Y_lm_real_batch(
    point: torch.Tensor,
    l: int,
) -> torch.Tensor:
    """
    Compute the real spherical harmonics Y_l^m(θ, φ) on grid.

    Args:
        point: (ngpts, 3) tensor of Cartesian coordinates
        l: degree of the harmonic

    Returns:
        Y_lm_real: (2 * el + 1, ngpts) tensor of real spherical harmonics
    """
    coef = torch.tensor(coef_list[l], dtype=point.dtype, device=point.device).unsqueeze(
        -1
    )
    if l == 0:
        result = coef * torch.ones_like(point[:, 0])
    elif l == 1:
        x, y, z = point[:, 0], point[:, 1], point[:, 2]
        y_lm = coef * torch.stack([y, z, x]) / point.norm(dim=1)
        result = y_lm
    elif l == 2:
        x, y, z = point[:, 0], point[:, 1], point[:, 2]
        r2 = point.norm(dim=1).square()
        y_lm = torch.stack([x * y, y * z, 3 * z**2 - r2, z * x, x**2 - y**2])
        result = coef * y_lm / r2
    elif l == 3:
        x, y, z = point[:, 0], point[:, 1], point[:, 2]
        x2, y2, z2 = x.square(), y.square(), z.square()
        r = point.norm(dim=1)
        r2 = r.square()
        y_lm = torch.stack(
            [
                y * (3 * x2 - y2),
                x * y * z,
                y * (5 * z2 - r2),
                z * (5 * z2 - 3 * r2),
                x * (5 * z2 - r2),
                z * (x2 - y2),
                x * (x2 - 3 * y2),
            ],
        )
        result = coef * y_lm / r2 / r
    else:
        raise ValueError("l must be 0, 1, 2, or 3; unsupported l value")
    return torch.nan_to_num(result, nan=0.0)


def Y_lm_real(
    point: torch.Tensor,
    l: int,
    m: int,
) -> torch.Tensor:
    """
    Compute the real spherical harmonics Y_l^m(θ, φ) on grid.

    Args:
        point: (ngpts, 3) tensor of Cartesian coordinates
        l: degree of the harmonic
        m: order of the harmonic

    Returns:
        Y_lm_real: (ngpts,) tensor of real spherical harmonics
    """
    theta, phi = cartesian_to_spherical(point)
    return real_sph_harm(l, m, theta, phi)


def cartesian_to_spherical(point: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cartesian coordinates to spherical coordinates.

    Args:
        point: (ngpts, 3) tensor of Cartesian coordinates

    Returns:
        theta: (ngpts,) tensor of polar angles in radians
        phi: (ngpts,) tensor of azimuthal angles in radians
    """
    x = point[:, 0]
    y = point[:, 1]
    z = point[:, 2]
    phi = torch.arctan2(y, x)
    theta = torch.arctan2(torch.sqrt(x**2 + y**2), z)
    return theta, phi


def real_sph_harm(
    l: int, m: int, theta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """
    Compute the real spherical harmonics Y_l^m(θ, φ).

    This function transforms the complex spherical harmonics into a real basis using:

        For m > 0:
            Y_l^m_real(θ, φ) = sqrt(2) * (-1)^m * Re[Y_l^m(θ, φ)]

        For m = 0:
            Y_l^0_real(θ, φ) = Y_l^0(θ, φ)

        For m < 0:
            Y_l^m_real(θ, φ) = sqrt(2) * (-1)^m * Im[Y_l^{|m|}(θ, φ)]

    Where:
        - l: degree (l ≥ 0)
        - m: order (−l ≤ m ≤ l)
        - θ: polar angle in radians, θ ∈ [0, π]
        - φ: azimuthal angle in radians, φ ∈ [0, 2π]
        - Re[], Im[] denote real and imaginary parts of the complex spherical harmonics

    For l <= 3 cases, the real spherical harmonics can be computed directly using the precomputed formulas.
    Reference: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics

    Args:
        l: degree of the harmonic
        m: order of the harmonic
        theta: polar angle (0 <= theta <= pi)
        phi: azimuthal angle (0 <= phi < 2 * pi)
    """
    if l == 0:
        return 0.5 * np.sqrt(1 / np.pi) * torch.ones_like(theta)
    elif l == 1:
        if m == -1:
            return np.sqrt(3 / (4 * np.pi)) * torch.sin(theta) * torch.sin(phi)
        elif m == 0:
            return np.sqrt(3 / (4 * np.pi)) * torch.cos(theta)
        elif m == 1:
            return np.sqrt(3 / (4 * np.pi)) * torch.sin(theta) * torch.cos(phi)
    elif l == 2:
        if m == -2:
            return (
                0.25 * np.sqrt(15 / np.pi) * torch.sin(theta) ** 2 * torch.sin(2 * phi)
            )
        elif m == -1:
            return 0.25 * np.sqrt(15 / np.pi) * torch.sin(2 * theta) * torch.sin(phi)
        elif m == 0:
            return 0.25 * np.sqrt(5 / np.pi) * (3 * torch.cos(theta) ** 2 - 1)
        elif m == 1:
            return 0.25 * np.sqrt(15 / np.pi) * torch.sin(2 * theta) * torch.cos(phi)
        elif m == 2:
            return (
                0.25 * np.sqrt(15 / np.pi) * torch.sin(theta) ** 2 * torch.cos(2 * phi)
            )
    elif l == 3:
        if m == -3:
            return (
                0.25
                * np.sqrt(35 / (2 * np.pi))
                * torch.sin(theta) ** 3
                * torch.sin(3 * phi)
            )
        elif m == -2:
            return (
                0.25
                * np.sqrt(105 / np.pi)
                * (torch.sin(theta) ** 2)
                * torch.cos(theta)
                * torch.sin(2 * phi)
            )
        elif m == -1:
            return (
                0.25
                * np.sqrt(21 / 2 / np.pi)
                * torch.sin(theta)
                * (5 * torch.cos(theta) ** 2 - 1)
                * torch.sin(phi)
            )
        elif m == 0:
            return (
                0.25
                * np.sqrt(7 / np.pi)
                * (5 * torch.cos(theta) ** 3 - 3 * torch.cos(theta))
            )
        elif m == 1:
            return (
                0.25
                * np.sqrt(21 / 2 / np.pi)
                * torch.sin(theta)
                * (5 * torch.cos(theta) ** 2 - 1)
                * torch.cos(phi)
            )
        elif m == 2:
            return (
                0.25
                * np.sqrt(105 / np.pi)
                * (torch.sin(theta) ** 2)
                * torch.cos(theta)
                * torch.cos(2 * phi)
            )
        elif m == 3:
            return (
                0.25
                * np.sqrt(35 / (2 * np.pi))
                * torch.sin(theta) ** 3
                * torch.cos(3 * phi)
            )
    else:
        raise ValueError("l must be 0, 1, 2, or 3; unsupported l value")


def sph2cart(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z


def plot_real_sph(l, m, title):
    import matplotlib.pyplot as plt

    res = 200
    theta = torch.linspace(0, np.pi, res)
    phi = torch.linspace(0, 2 * np.pi, res)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    r = real_sph_harm(l, m, theta, phi)

    # scale radius for plotting (absolute value)
    r_plot = r.abs()

    # convert to cartesian
    x, y, z = sph2cart(r_plot, theta, phi)
    r_val = r.numpy()

    # 3D plot
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x.numpy(),
        y.numpy(),
        z.numpy(),
        facecolors=plt.cm.coolwarm((r_val - r_val.min()) / (r_val.max() - r_val.min())),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


def plot_cartesian_sph(l, m, title, grid_size=50):
    import matplotlib.pyplot as plt

    # Generate a grid of points in Cartesian coordinates
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)  # [N, 3]

    r = torch.norm(xyz, dim=1)
    mask = r <= 1.0  # inside unit sphere

    val = Y_lm_real(xyz, l, m)

    # Visualize: points + color
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")
    color = val[mask].numpy()
    norm = plt.Normalize(vmin=color.min(), vmax=color.max())
    ax.scatter(
        xyz[mask, 0].numpy(),
        xyz[mask, 1].numpy(),
        xyz[mask, 2].numpy(),
        c=plt.cm.coolwarm(norm(color)),
        alpha=0.6,
        s=5,
    )
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


def spherical_jn(
    n: int,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Compute the spherical Bessel function of the first kind j_n(x).
    Refer to: https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions

    Args:
        n: order of the Bessel function
        x: input values, points on 1D grid, i.e. ascending order

    Returns:
        j_n: spherical Bessel function of the first kind of order n at x
    """
    out = torch.empty_like(x)

    # mask for processing the zero division case
    mask0 = x == 0
    mask = ~mask0
    _x = x[mask]

    if n == 0:
        out[mask] = torch.sin(_x) / _x
        out[mask0] = 1.0
    elif n == 1:
        out[mask] = (torch.sin(_x) / _x - torch.cos(_x)) / _x
        out[mask0] = 0.0
    elif n == 2:
        x2 = _x**2
        out[mask] = ((3 - x2) * torch.sin(_x) / _x - 3 * torch.cos(_x)) / x2
        out[mask0] = 0.0
    elif n == 3:
        x2 = _x**2
        tmp = torch.sin(_x) / x2
        out[mask] = (15 - x2) / x2 * (tmp - torch.cos(_x) / _x) - 5 * tmp
        out[mask0] = 0.0
    else:
        raise ValueError("n must be 0, 1, 2, or 3; unsupported n value")
    return out


if __name__ == "__main__":
    # Plot real spherical harmonics
    for l in range(4):
        for m in range(-l, l + 1):
            # plot_real_sph(l, m, f"Y_l^m (l={l}, m={m})")
            plot_cartesian_sph(l, m, f"Y_l^m (l={l}, m={m})")
