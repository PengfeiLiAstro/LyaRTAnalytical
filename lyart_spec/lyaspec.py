import numpy as np
import warnings
from .utils import *

def lya_spec_cls(x = None, tau0 = 1e4, taus2tau0 = 0.0, xi = 0.0, geometry = "sla", 
                 v2b = 0.0, recoil_flag = False, T = 10.0):
    """ Evaluate Lya spectra with radiative transfer using closed-form solutions/formulae. 
    Kwargs:
        x (numpy.ndarray): A 1D array of frequency parameter.
        tau0 (float): Optical depth
        taus2tau0 (float): The ratio of source optical depth to cloud optical depth, between 0.0 and 1.0.
        xi (float): Initial frequency. 
        geometry (str): "sla", "cyl", "sph" for slab, cylindrical, and spherical geometry, respectively. 
        v2b (float): The ratio of cloud edge velocity to thermal velocity. 
        recoil_flag (bool): True: with recoil; False: without recoil. 
        T (float): Temperature of the gas cloud in K.
    Returns:
        x (numpy.ndarray): A 1D array of frequency parameter. If the input x is not None, the return x will be the same as the input x. Otherwise, the output x will be initiated inside the function. 
        y (numpy.ndarray): Emergent Lya intensity at each given frequency parameter x.
    """

    const = PhyConst()
    b_th = np.sqrt(2*const.kB*T/const.m_H)
    vE = v2b*b_th
    taus = tau0*taus2tau0
    av = anu_voigt(T)

    if x is None:
        xnorm_bin = np.linspace(-10.0, 10.0, 401)
        xnorm = (xnorm_bin[1:] + xnorm_bin[:-1])*0.5    
        x = xnorm*np.power(av*tau0, 1/3)


    if np.isclose(v2b, 0.0) == True:
        if geometry == "sla":
            y = pla_ana(x, tau0, av = av, taus = taus, xi = xi)
        elif geometry == "cyl":
            y = cyl_fit(x, tau0, av = av, taus = taus, xi = xi)
        elif geometry == "sph":
            y = sph_ana(x, tau0, av = av, taus = taus, xi = xi)
        else:
            raise ValueError("Please choose the correct geometry among sla, cyl, and sph.")
    else:
        if np.isclose(taus2tau0, 0.0) == False or np.isclose(xi, 0.0) == False:
            raise ValueError("The closed-form solution/formula for non-zero velocity gradients is only valid "\
                             "for taus2tau0=0 and xi=0."\
                             "Switch to series solutions using function lya_spec_ses for non-zero "\
                             "taus2tau0 or non-zero xi. "\
                             "Notice that the series solution for non-zero velocity gradients is "\
                             "not accurate for abs(v2b)>1.")
        if np.log10(np.abs(v2b)) > 2.0:
            warnings.warn("The closed-form solution/formula for abs(v2b)>100 is not "\
                          "tested again Monte Carlo radiative transfer simulation. "\
                          "This scenario is executable but please proceed with caution.", UserWarning)
        if np.log10(av*tau0) < 2.2 or np.log10(av*tau0) > 4.2:
            warnings.warn("The closed-form solution/formula for log10(av*tau0) outside [2.2, 4.2] is not "\
                          "tested again Monte Carlo radiative transfer simulation. "\
                          "This scenario is executable but please proceed with caution.", UserWarning)

        pcl = FitFuncVTau()
        if geometry == "sla":
            y = pcl.spec_jointfit(x, vratio = v2b, tau0 = tau0, T = T, geo_flag = 0)
        elif geometry == "cyl":
            y = pcl.spec_jointfit(x, vratio = v2b, tau0 = tau0, T = T, geo_flag = 1)
        elif geometry == "sph":
            y = pcl.spec_jointfit(x, vratio = v2b, tau0 = tau0, T = T, geo_flag = 2)
        else:
            raise ValueError("Please choose the correct geometry among sla, cyl, and sph.")

    if recoil_flag == True:
        if np.isclose(v2b, 0.0) == False:
            warnings.warn("The recoil correction for non-zero velocity gradients is not "\
                          "tested again Monte Carlo radiative transfer simulation. "\
                          "This scenario is executable but please proceed with caution.", UserWarning)
        y = recoil_corr(x, y, T = T)
    return x, y


def lya_spec_ses(x = None, tau0 = 1e4, taus2tau0 = 0.0, xi = 0.0, geometry = "sla", 
                 v2b = 0.0, recoil_flag = False, T = 10.0, Nses = 1000):
    """ Evaluate Lya spectra with radiative transfer using series solutions. 
    Kwargs:
        x (numpy.ndarray): A 1D array of frequency parameter.
        tau0 (float): Optical depth
        taus2tau0 (float): The ratio of source optical depth to cloud optical depth, between 0.0 and 1.0.
        xi (float): Initial frequency. 
        geometry (str): "sla", "cyl", "sph" for slab, cylindrical, and spherical geometry, respectively. 
        v2b (float): The ratio of cloud edge velocity to thermal velocity. 
        recoil_flag (bool): True: with recoil; False: without recoil. 
        T (float): Temperature of the gas cloud in K.
        Nses (int): The number of terms to be evaluated in the series. For Nses=1000, the series solution will be evaluated from n=0 to n=999, 1000 terms in total. 
    Returns:
        x (numpy.ndarray): A 1D array of frequency parameter. If the input x is not None, the return x will be the same as the input x. Otherwise, the output x will be initiated inside the function. 
        y (numpy.ndarray): Emergent Lya intensity at each given frequency parameter x.
    """


    const = PhyConst()
    b_th = np.sqrt(2*const.kB*T/const.m_H)
    vE = v2b*b_th
    taus = tau0*taus2tau0
    av = anu_voigt(T)

    if np.abs(v2b) > 1.0:
        warnings.warn("abs(v2b)>1.0 detected. The series solution is no longer accurate. "\
                      "Switch to closed-form solutions/formulae using function lya_spec_cls for better accuracy. "\
                      "Notice that the closed-form solution/formula for non-zero velocity gradients is "\
                      "only functional for taus2tau0=0 and xi=0.", UserWarning)

    if x is None:
        xnorm_bin = np.linspace(-10.0, 10.0, 401)
        xnorm = (xnorm_bin[1:] + xnorm_bin[:-1])*0.5    
        x = xnorm*np.power(av*tau0, 1/3)


    if geometry == "sla":
        y = pla_ses(x, tau0, T = T, vE = vE, taus = taus, xi = xi, N = Nses)

    elif geometry == "cyl":
        y = cyl_ses(x, tau0, T = T, vE = vE, taus = taus, xi = xi, N = Nses)

    elif geometry == "sph":
        y = sph_ses(x, tau0, T = T, vE = vE, taus = taus, xi = xi, N = Nses)
    else:
        raise ValueError("Please choose the correct geometry among sla, cyl, and sph.")

    if recoil_flag == True:
        if np.isclose(v2b, 0.0) == False:
            warnings.warn("The recoil correction for non-zero velocity gradients is not "\
                          "tested again Monte Carlo radiative transfer simulation. "\
                          "This scenario is executable but please proceed with caution.", UserWarning)
        y = recoil_corr(x, y, T = T)

    return x, y

def lya_spec_sim(tau0 = 1e4, taus2tau0 = 0.0, xi = 0.0, geometry = "sla", 
                 v2b = 0.0, recoil_flag = False, T = 10.0):
    """ Read simulated Lya spectra with radiative transfer. Only certain discrete values of the input parameters have corresponding RT simulations. Please check the table in README.md file for the available parameters. Input parameters other than the available ones will raise an error. 
    Kwargs:
        tau0 (float): Optical depth
        taus2tau0 (float): The ratio of source optical depth to cloud optical depth, between 0.0 and 1.0.
        xi (float): Initial frequency. 
        geometry (str): "sla", "cyl", "sph" for slab, cylindrical, and spherical geometry, respectively. 
        v2b (float): The ratio of cloud edge velocity to thermal velocity. 
        recoil_flag (bool): True: with recoil; False: without recoil. 
        T (float): Temperature of the gas cloud in K.
    Returns:
        x (numpy.ndarray): A 1D array of frequency parameter. The output x is determined when simulated Lya spectra are created. 
        y (numpy.ndarray): Emergent Lya intensity at each given frequency parameter x.
    """

    geo_num = ["sla", "cyl", "sph"].index(geometry)
    if recoil_flag == True:
        recoil_num = 1
    else:
        recoil_num = 0

    pname = ["tau0", "taus2tau0", "xi", "geo_num", "v2b", "recoil_num", "T"]
    parr = np.array([tau0, taus2tau0, xi, geo_num, v2b, recoil_num, T])

    xnorm_bin = np.linspace(-10.0, 10.0, 401)
    xnorm = (xnorm_bin[1:] + xnorm_bin[:-1])*0.5

    par_tab = np.load("./data/SpecSimParameter.npy")
    spec_tab = np.load("./data/SpecSimProfile.npy")
    diffsq = np.sum((par_tab - parr)**2, axis = 1)
    ind = np.argmin(diffsq)
    if diffsq[ind] > 1e-10:
        raise ValueError("The input parameters do not have a corresponding simulated spectra. "\
                         "Please refer to the README.md for the parameters with simulated spectra.")
    y = spec_tab[ind]
    tau0 = parr[pname.index("tau0")]
    T = parr[pname.index("T")]
    av = anu_voigt(T)
    x = xnorm*np.power(av*tau0, 1/3)
    return x, y
