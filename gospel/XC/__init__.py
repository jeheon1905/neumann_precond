from gospel.XC.XC_Functional import XC_Functional

__all__ = ['XC_Functional']


def create_xc(params, calc):
    if params is None:
        xc = XC_Functional('gga_x_pbe + gga_c_pbe', calc=calc)
    elif isinstance(params, dict):
        xc_type = params.get('type').lower()
        xc_weight = params.get('weight')
        if xc_type == 'lda':
            xc = XC_Functional('lda_x + lda_c_pz', calc=calc)
        elif xc_type == 'gga':
            xc = XC_Functional('gga_x_pbe + gga_c_pbe', calc=calc)
        elif xc_type == 'exx':
            xc = XC_Functional('exx', calc=calc)
        elif xc_type == 'pbe0':
            xc = XC_Functional('gga_x_pbe + exx + gga_c_pbe',
                               '0.75, 0.25, 1.', calc=calc)
        elif xc_type == 'b3lyp':
            xc = XC_Functional('HYB_GGA_XC_B3LYP + exx',
                               '0.8, 0.2', calc=calc)
            raise NotImplementedError
        elif xc_type == 'hse':
            raise NotImplementedError
        elif xc_type == 'lc-wpbe(2gau)':
            raise NotImplementedError
        else:
            xc = XC_Functional(xc_type, xc_weight, calc=calc)
    else:
        assert False, "An input parameter's type for xc should be dictionary."
    return xc
