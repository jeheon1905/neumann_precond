from math import ceil
import numpy as np
import torch
from scipy.interpolate import interp1d

from gospel.Pseudopotential.UPF import get_zero_convg_idx
from gospel.util import Timer
from gospel.special_functions import spherical_jn


# TODO: Filtering의 결과가 radial grid의 길이에 dependency를 가짐. 똑같은 pseudopotential이어도 정의된 radial grid의 길이에 따라 다른 filtering결과가 나올 수 있음. Filtering 시에 cutoff + alpha 로 radial grid를 재정의하고나서 filtering을 수행하도록 수정하기
# TODO: Replace numpy with torch (Modification of UPF.py should be preceded)
# TODO: After that, remove explicit device argument.
class Filtering:
    def __init__(self, spacing):
        """Filtering pseudopotential
        Citation)
            https://doi.org/10.1063/1.2193514
            https://doi.org/10.1103/PhysRevB.64.201107

        Arguments)
            spacing (float): target grid spacing (unit: a.u.)
        """
        self.alpha_local = 1.1
        self.alpha_nonlocal = 1.1
        self.gamma_local = 2.0
        self.gamma_nonlocal = 2.0

        self.max_freq = np.pi / spacing
        self.dq = 0.005  # self.max_freq / 1000
        self.q = torch.arange(0, self.max_freq, self.dq)
        self.tol = 1e-6  # tolerance considered to be zero
        # self.tol = 1e-7  # tolerance considered to be zero

        self.mask_grid, self.mask_function = self.initialize_mask()
        return

    @Timer.timeit
    def filtering(self, upf, what="local"):
        """Return two dictionaries of filtered functions and its cutoff radius, respectively. Here, element symbol is key.

        Arguments)
            upf (gospel.Pseudopotential.UPF)
            what (str): whether the function to be filted is 'local' or 'nonlocal'

        Return)
            P_list (dict of np.ndarray): filtered functions
            cutoff_list (dict of np.ndarray): cutoff radii of filtered functions
        """
        assert what in ["nonlocal", "local"]
        P_list = {}
        cutoff_list = {}

        if what == "local":
            for i_atom, i_sym in enumerate(upf.elements):
                P_filt = self.filter(
                    upf[i_sym].R,
                    upf[i_sym].RAB,
                    upf[i_sym].local,
                    upf[i_sym].local_cutoff,
                    l=0,
                    what=what,
                    device=upf.device,
                ).cpu().numpy()
                P_list[i_sym] = P_filt
                cutoff = upf[i_sym].R[get_zero_convg_idx(P_filt, tol=self.tol)]
                cutoff = ceil(cutoff * 1e2) / 1e2
                cutoff_list[i_sym] = cutoff
        elif what == "nonlocal":
            for i_atom, i_sym in enumerate(upf.elements):
                P_list[i_sym] = []
                cutoff_list[i_sym] = []
                for i_proj in range(len(upf[i_sym].beta)):
                    P_filt = self.filter(
                        upf[i_sym].R,
                        upf[i_sym].RAB,
                        upf[i_sym].beta[i_proj],
                        upf[i_sym].beta_cutoff[i_proj],
                        l=upf[i_sym].angmom[i_proj],
                        what=what,
                        device=upf.device,
                    ).cpu().numpy()
                    P_list[i_sym].append(P_filt)
                    cutoff = upf[i_sym].R[get_zero_convg_idx(P_filt, tol=self.tol)]
                    cutoff = ceil(cutoff * 1e2) / 1e2
                    cutoff_list[i_sym].append(cutoff)
        return P_list, cutoff_list

    def sph_bessel_transform(self, r, P, l, q, dr):
        qr = torch.outer(q, r)
        bessel = spherical_jn(l, qr)
        return (P * r**2 * bessel * dr).sum(dim=1) * np.sqrt(2 / np.pi)

    def fourier_filter(
        self,
        q: torch.Tensor,
        q_cut: float,
        beta: float
    ) -> torch.Tensor:
        F_fs = torch.ones_like(q)
        mask = q > q_cut
        F_fs[mask] = torch.exp(-beta * (q[mask] / q_cut - 1) ** 2)
        return F_fs

    # TODO: Remove explicit usage of device
    def filter(self, r, dr, P, cutoff, l=0, what="local", device=None):
        """
        r : radial grid points
        dr :
        P : function to be filtered (e.g., local potential or nonlocal projector)
        cutoff : cutoff radius of P
        l : angular momentum quantum number
        what : object to be filtered
        """
        assert len(r) == len(dr) == len(P)
        assert isinstance(l, int)
        assert what in ["local", "nonlocal"]

        if what == "local":
            q_cut = self.max_freq / self.alpha_local
            r_cut = cutoff * self.gamma_local
            beta_fs = 5 * np.log(10) / (self.alpha_local - 1) ** 2
        elif what == "nonlocal":
            q_cut = self.max_freq / self.alpha_nonlocal
            r_cut = cutoff * self.gamma_nonlocal
            beta_fs = 5 * np.log(10) / (self.alpha_nonlocal - 1) ** 2

        inside_r_cut = r <= r_cut
        _r, _P, _dr = r[inside_r_cut], P[inside_r_cut], dr[inside_r_cut]

        # m(r / r_cut)
        mask = self.mask(_r / r_cut)

        # NOTE: temporary fix for torch
        # numpy to torch
        _r = torch.from_numpy(_r).to(device)
        _P = torch.from_numpy(_P).to(device)
        _dr = torch.from_numpy(_dr).to(device)
        mask = torch.from_numpy(mask).to(device)
        self.q = self.q.to(device)

        # P(q) = F_fs(q) \int P(r) / m(r/r_cut) j_l(qr) r^2 dr
        P_q = self.sph_bessel_transform(_r, _P / mask, l, self.q, _dr)
        F_fs_q = self.fourier_filter(self.q, q_cut, beta_fs)
        assert F_fs_q[-1] < 1e-4, f"F_fs_q[-1] = {F_fs_q[-1]}"
        P_q = F_fs_q * P_q

        # P(r) = m(r/r_cut) \int P(q) j_l(qr) q^2 dq
        P_r = self.sph_bessel_transform(self.q, P_q, l, _r, self.dq)
        P_r = mask * P_r
        P_r = torch.nn.functional.pad(
            P_r, (0, len(r) - len(P_r)), mode="constant", value=0.0
        )
        return P_r

    def mask(self, x):
        interp = interp1d(
            self.mask_grid,
            self.mask_function,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return interp(x)

    def initialize_mask(self):
        ## numerical function
        mask_grid = np.linspace(0.0, 1.0, 201)
        # fmt: off
        mask_function = np.array([
            0.10000000E+01,0.10000000E+01,0.99948662E+00,0.99863154E+00,0.99743557E+00,
            0.99589985E+00,0.99402586E+00,0.99181538E+00,0.98927052E+00,0.98639370E+00,
            0.98318766E+00,0.97965544E+00,0.97580040E+00,0.97162618E+00,0.96713671E+00,
            0.96233623E+00,0.95722924E+00,0.95182053E+00,0.94611516E+00,0.94011842E+00,
            0.93383589E+00,0.92727338E+00,0.92043693E+00,0.91333282E+00,0.90596753E+00,
            0.89834777E+00,0.89048044E+00,0.88237263E+00,0.87403161E+00,0.86546483E+00,
            0.85667987E+00,0.84768450E+00,0.83848659E+00,0.82909416E+00,0.81951535E+00,
            0.80975838E+00,0.79983160E+00,0.78974340E+00,0.77950227E+00,0.76911677E+00,
            0.75859548E+00,0.74794703E+00,0.73718009E+00,0.72630334E+00,0.71532544E+00,
            0.70425508E+00,0.69310092E+00,0.68187158E+00,0.67057566E+00,0.65922170E+00,
            0.64781819E+00,0.63637355E+00,0.62489612E+00,0.61339415E+00,0.60187581E+00,
            0.59034914E+00,0.57882208E+00,0.56730245E+00,0.55579794E+00,0.54431609E+00,
            0.53286431E+00,0.52144984E+00,0.51007978E+00,0.49876105E+00,0.48750040E+00,
            0.47630440E+00,0.46517945E+00,0.45413176E+00,0.44316732E+00,0.43229196E+00,
            0.42151128E+00,0.41083069E+00,0.40025539E+00,0.38979038E+00,0.37944042E+00,
            0.36921008E+00,0.35910371E+00,0.34912542E+00,0.33927912E+00,0.32956851E+00,
            0.31999705E+00,0.31056799E+00,0.30128436E+00,0.29214897E+00,0.28316441E+00,
            0.27433307E+00,0.26565709E+00,0.25713844E+00,0.24877886E+00,0.24057988E+00,
            0.23254283E+00,0.22466884E+00,0.21695884E+00,0.20941357E+00,0.20203357E+00,
            0.19481920E+00,0.18777065E+00,0.18088790E+00,0.17417080E+00,0.16761900E+00,
            0.16123200E+00,0.15500913E+00,0.14894959E+00,0.14305240E+00,0.13731647E+00,
            0.13174055E+00,0.12632327E+00,0.12106315E+00,0.11595855E+00,0.11100775E+00,
            0.10620891E+00,0.10156010E+00,0.97059268E-01,0.92704295E-01,0.88492966E-01,
            0.84422989E-01,0.80492001E-01,0.76697569E-01,0.73037197E-01,0.69508335E-01,
            0.66108380E-01,0.62834685E-01,0.59684561E-01,0.56655284E-01,0.53744102E-01,
            0.50948236E-01,0.48264886E-01,0.45691239E-01,0.43224469E-01,0.40861744E-01,
            0.38600231E-01,0.36437098E-01,0.34369520E-01,0.32394681E-01,0.30509780E-01,
            0.28712032E-01,0.26998673E-01,0.25366964E-01,0.23814193E-01,0.22337676E-01,
            0.20934765E-01,0.19602844E-01,0.18339338E-01,0.17141711E-01,0.16007467E-01,
            0.14934157E-01,0.13919377E-01,0.12960772E-01,0.12056034E-01,0.11202905E-01,
            0.10399183E-01,0.96427132E-02,0.89313983E-02,0.82631938E-02,0.76361106E-02,
            0.70482151E-02,0.64976294E-02,0.59825322E-02,0.55011581E-02,0.50517982E-02,
            0.46327998E-02,0.42425662E-02,0.38795566E-02,0.35422853E-02,0.32293218E-02,
            0.29392897E-02,0.26708663E-02,0.24227820E-02,0.21938194E-02,0.19828122E-02,
            0.17886449E-02,0.16102512E-02,0.14466132E-02,0.12967606E-02,0.11597692E-02,
            0.10347601E-02,0.92089812E-03,0.81739110E-03,0.72348823E-03,0.63847906E-03,
            0.56169212E-03,0.49249371E-03,0.43028657E-03,0.37450862E-03,0.32463165E-03,
            0.28016004E-03,0.24062948E-03,0.20560566E-03,0.17468305E-03,0.14748362E-03,
            0.12365560E-03,0.10287226E-03,0.84830727E-04,0.69250769E-04,0.55873673E-04,
            0.44461100E-04,0.34793983E-04,0.26671449E-04,0.19909778E-04,0.14341381E-04,
            0.98138215E-05,
            ])
        # fmt: on
        return mask_grid, mask_function


if __name__ == "__main__":
    """Plot local potential and projectors of before and after filtering."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="Pseudopotential file path (.UPF)")
    parser.add_argument(
        "--spacing", type=float, help="grid spacing (Bohr)", default=0.4
    )
    args = parser.parse_args()
    print("args=", args)

    from UPF import UPF
    import matplotlib.pyplot as plt

    def _calc_comp_pot(Z, r, r_g=1.0):
        import torch
        from math import sqrt, pi

        r = torch.from_numpy(r)

        R = r / r_g
        V_c_0 = 2 * Z / r_g / sqrt(pi)
        V_c = Z * torch.erf(R)
        V_c = torch.div(V_c, r)
        V_c[r == 0] = V_c_0
        return V_c.numpy()

    upf = UPF([args.filepath])
    element = upf.elements[0]

    ## Add compensation
    V_comp = _calc_comp_pot(upf[element].zval, upf[element].R)
    upf[element].local += V_comp

    R = upf[element].R
    beta = upf[element].beta
    local = upf[element].local

    ## Do filtering
    _filter = Filtering(spacing=args.spacing)

    filtered_local, filtered_local_cutoff = _filter.filtering(upf, "local")
    filtered_local = filtered_local[element]
    filtered_local_cutoff = filtered_local_cutoff[element]

    filtered_beta, filtered_beta_cutoff = _filter.filtering(upf, "nonlocal")
    filtered_beta = filtered_beta[element]
    filtered_beta_cutoff = filtered_beta_cutoff[element]

    if len(beta) == 0:
        plt.figure()
        plt.plot(R, local, label="local")
        plt.plot(R, filtered_local, label="filtered_local")
        plt.axvline(x=filtered_local_cutoff, color="b")
        plt.axvline(0.0, color="k")
        plt.axhline(0.0, color="k")
        plt.title(f"{args.filepath}, spacing={args.spacing}")
        plt.tight_layout()
        plt.legend(loc="upper right")
        plt.show()
    else:
        ## plot subplot
        fig, axs = plt.subplots(len(beta) + 1)
        for i in range(len(beta) + 1):
            if i == 0:
                axs[i].plot(R, local, label="local")
                axs[i].plot(R, filtered_local, label="filtered_local")
                axs[i].axvline(x=filtered_local_cutoff, color="b")
            else:
                axs[i].plot(R, beta[i - 1], label=f"beta ({i})")
                axs[i].plot(R, filtered_beta[i - 1], label=f"filtered_beta ({i})")
                axs[i].axvline(x=filtered_beta_cutoff[i - 1], color="b")
            axs[i].legend(loc="upper right")
            axs[i].axvline(x=0, color="k")
            axs[i].axhline(y=0, color="k")
        fig.suptitle(f"{args.filepath}, spacing={args.spacing}")
        fig.tight_layout()
        plt.show()
