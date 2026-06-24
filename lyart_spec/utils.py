import numpy as np
from scipy.special import erf
import scipy as sp

DPATH = "./data/"

def anu_voigt(T):
    return 4.7e-4/np.sqrt(T/1e4)

class PhyConst:
    def __init__(self):
        self.nu_0 = 2.4660677e15 #s^{-1}
        self.c_cgs = 2.99792458e10 # cm/s
        self.m_H = 1.672623e-24  # mass of H atom, g
        self.h_plank = 6.6260755e-27 # Planck constant, cm^2 g/s
        self.kB = 1.38e-16 #erg/K

def pla_ses(xarr, tau0, T = 10.0, vE = 0.0, taus = 0.0, xi = 0.0, N = 1000):
    av = anu_voigt(T)
    const = PhyConst()
    b_th = np.sqrt(2*const.kB*T/const.m_H)
    gamma_tau0 = np.sqrt(6)*vE/3.0/b_th # gamma * tau0

    tt = taus/tau0
    coeff = np.sqrt(6)*xarr**2/6/np.sqrt(np.pi)/av/tau0
    y = np.sqrt(2*np.pi/27)*xarr**3/av
    yi = np.sqrt(2*np.pi/27)*xi**3/av
    dyarr = np.abs(y - yi)
    dyarr_noabs = y - yi
    farr = np.full(len(xarr), 0.0)
    lnt = (np.arange(N) + 1/2)*np.pi #lambda_n * tau0
    gnt = np.sqrt(lnt**2 + gamma_tau0**2/4.0) #gamma_n * tau0
    Narr = np.arange(N)
    m1_arr = np.power(-1, Narr)
    for idy, dy in enumerate(dyarr):
        fsum = np.full(N, 0.0)
        dy_noabs = dyarr_noabs[idy]
        fsum = m1_arr*np.cos(lnt*tt)*lnt/gnt*np.exp(-gnt/tau0*dy - gamma_tau0/tau0/2.0*dy_noabs)
        farr[idy] = np.sum(fsum)
    return farr*coeff

def cyl_ses(xarr, tau0, T = 10.0, vE = 0.0, taus = 0.0, xi = 0.0, N = 1000):
    av = anu_voigt(T)
    const = PhyConst()
    b_th = np.sqrt(2*const.kB*T/const.m_H)
    gamma_tau0 = 2.0*np.sqrt(6)*vE/3.0/b_th # gamma * tau0

    tt = taus/tau0
    coeff = np.sqrt(6)*xarr**2/6/np.sqrt(np.pi)/av/tau0
    y = np.sqrt(2*np.pi/27)*xarr**3/av
    yi = np.sqrt(2*np.pi/27)*xi**3/av
    dyarr = np.abs(y - yi)
    dyarr_noabs = y - yi
    farr = np.full(len(xarr), 0.0)
    lnt = (np.arange(N) + 3/4)*np.pi #lambda_n * tau0
    gnt = np.sqrt(lnt**2 + gamma_tau0**2/4.0) #gamma_n * tau0
    for idy, dy in enumerate(dyarr):
        fsum = np.full(N, 0.0)
        dy_noabs = dyarr_noabs[idy]
        fsum = sp.special.j0(lnt*tt)/sp.special.j1(lnt)*lnt/gnt*np.exp(-gnt/tau0*dy - gamma_tau0/tau0/2.0*dy_noabs)
        farr[idy] = np.sum(fsum)
    return farr*coeff

def sph_ses(xarr, tau0, T = 10.0, vE = 0.0, taus = 0.0, xi = 0.0, N = 1000):
    av = anu_voigt(T)
    const = PhyConst()
    b_th = np.sqrt(2*const.kB*T/const.m_H)
    gamma_tau0 = np.sqrt(6)*vE/b_th # gamma * tau0

    tt = taus/tau0
    coeff = np.sqrt(6)*xarr**2/6/np.sqrt(np.pi)/av/tau0
    y = np.sqrt(2*np.pi/27)*xarr**3/av
    yi = np.sqrt(2*np.pi/27)*xi**3/av
    dyarr = np.abs(y - yi)
    dyarr_noabs = y - yi
    farr = np.full(len(xarr), 0.0)
    lnt = (np.arange(N) + 1)*np.pi #lambda_n * tau0
    gnt = np.sqrt(lnt**2 + gamma_tau0**2/4.0) #gamma_n * tau0
    Narr = np.arange(N)
    m1_arr = np.power(-1, Narr)
    if tt < 1e-3:
        for idy, dy in enumerate(dyarr):
            fsum = np.full(N, 0.0)
            dy_noabs = dyarr_noabs[idy]
            fsum = m1_arr*lnt*lnt/gnt*np.exp(-gnt/tau0*dy - gamma_tau0/tau0/2.0*dy_noabs)
            farr[idy] = np.sum(fsum)
    else:
        for idy, dy in enumerate(dyarr):
            fsum = np.full(N, 0.0)
            dy_noabs = dyarr_noabs[idy]
            fsum = m1_arr*np.sin(lnt*tt)/tt*lnt/gnt*np.exp(-gnt/tau0*dy - gamma_tau0/tau0/2.0*dy_noabs)
            farr[idy] = np.sum(fsum)
    return farr*coeff

def pla_ana(x, tau0, av = 1.49e-2, taus = 0.0, xi = 0.0):
    tt = taus/tau0
    y = np.sqrt(2*np.pi/27)*x**3/av
    yi = np.sqrt(2*np.pi/27)*xi**3/av
    dy = np.abs(y - yi)

    coeff = np.sqrt(6)*x**2/12/np.sqrt(np.pi)/av/tau0
    num = np.cosh(np.pi*dy/2/tau0)*np.cos(np.pi*tt/2)
    den = np.cosh(np.pi*dy/tau0) + np.cos(np.pi*tt)
    return coeff*num/den*2 # parallel plane has two surfaces.

def sph_ana(x, tau0, av = 1.49e-2, taus = 0.0, xi = 0.0):
    tt = taus/tau0
    y = np.sqrt(2*np.pi/27)*x**3/av
    yi = np.sqrt(2*np.pi/27)*xi**3/av
    dy = np.abs(y - yi)

    coeff = np.sqrt(6)*x**2/12/np.sqrt(np.pi)/av
    if taus < 1e-20:
        num = np.pi/tau0
    else:
        num = np.sin(np.pi*tt)/taus
    den = np.cos(np.pi*tt) + np.cosh(np.pi/tau0*dy)
    return coeff*num/den

def cyl_fit(x, tau0, av = 1.49e-2, taus = 0.0, xi = 0.0):
    tt = taus/tau0
    coeff = np.sqrt(6)*x**2/12/np.sqrt(np.pi)/tau0/av
    y = np.sqrt(2*np.pi/27)*x**3/av
    yi = np.sqrt(2*np.pi/27)*xi**3/av
    dy = np.abs(y - yi)
    num = np.exp(np.pi*dy/4/tau0)*np.cos(3*np.pi*tt/4 - np.pi/4)\
        + np.exp(-3*np.pi*dy/4/tau0)*np.cos(np.pi*tt/4 + np.pi/4)
    den = np.cos(np.pi*tt) + np.cosh(np.pi*dy/tau0)

    frac = np.exp(-2.96*np.power(dy/tau0, 1.0 + np.power(tt, 4.25)))
    corr_lo = 1.86 - 0.84*np.power(tt, 0.64) # Divide the original fitting by sqrt(tt)
    corr_lo *= frac
    corr_hi = sp.special.j0(3/4*np.pi*tt)/sp.special.j1(3/4*np.pi)\
            / np.sqrt(2/np.pi/(3/4*np.pi))/np.cos(3/4*np.pi*tt - np.pi/4)\
            * np.sqrt(2/np.pi/(3/4*np.pi))*np.sin(3/4*np.pi - np.pi/4)\
            * (1.0 - frac) # Divide the original fitting by sqrt(tt)
    return coeff*num/den*(corr_lo + corr_hi)

def recoil_corr(x, y, T = 10.0, alpha = 0.78):
    const = PhyConst()
    dnuD = np.sqrt(2*const.kB*T/const.m_H)/const.c_cgs*const.nu_0
    xT = const.kB*T/const.h_plank/dnuD

    dx = x[1] - x[0]
    return np.nansum(y*dx)/np.nansum(y*dx*np.exp(-alpha*x/xT))*y*np.exp(-alpha*x/xT)

class FitFuncVTau:
    def __init__(self, vratio = 0.1, tau0 = 1e6, T = 10.0, geo_flag = 0):
        self.T = T
        anu = anu_voigt(T)
        anu10 = anu_voigt(10.0)
        self.anu = anu
        self.anu10 = anu10
        self.vratio = vratio
        self.tau0 = tau0
        self.geo_flag = geo_flag
        self.pname_list = ["a", "b", "c", "alpha", "A", "beta"]
        self.pfunc_list = [self.afunc, self.bfunc, self.cfunc, self.alphafunc, self.Afunc, self.betafunc]

        FitFileName = DPATH +"vgrad_fitparam.dat"
        pfit = np.loadtxt(FitFileName)[:, self.geo_flag]
        a_a0, a_a1, a_a2, a_n0, a_n1, a_n2, \
        b_a0, b_a1, b_a2, b_n0, b_n1, b_n2, \
        c_a0, c_mu0, c_sigma0, c_n0, c_n1, \
        alpha_a0, alpha_a1, alpha_n0, alpha_n1, alpha_n2, \
        A_a0, A_a1, A_a2, A_n0, A_n1, A_n2, A_n3, \
        beta_a0, beta_a1, beta_n0 = pfit
        self.a_fit = np.array([a_a0, a_a1, a_a2, a_n0, a_n1, a_n2])
        self.b_fit = np.array([b_a0, b_a1, b_a2, b_n0, b_n1, b_n2])
        self.c_fit = np.array([c_a0, c_mu0, c_sigma0, c_n0, c_n1])
        self.alpha_fit = np.array([alpha_a0, alpha_a1, alpha_n0, alpha_n1, alpha_n2])
        self.A_fit = np.array([A_a0, A_a1, A_a2, A_n0, A_n1, A_n2, A_n3])
        self.beta_fit = np.array([beta_a0, beta_a1, beta_n0])

    def set_param(self, vratio = 0.1, tau0 = 1e6, T = 10.0, geo_flag = 0):
        self.__init__(vratio = vratio, tau0 = tau0, T = T, geo_flag = geo_flag)
        return True

    def afunc(self, a0, a1, a2, n0, n1, n2):
        vratio = np.abs(self.vratio)
        tau0 = self.tau0
        anu = self.anu
        anu10 = self.anu10

        x_a = vratio/np.power(anu*tau0, n0)
        a = a0/(1.0 + a1*np.power(x_a, n1)) + a2*np.power(1e6*anu10/tau0/anu, n2)
        return a

    def bfunc(self, a0, a1, a2, n0, n1, n2):
        vratio = np.abs(self.vratio)
        tau0 = self.tau0
        anu = self.anu
        anu10 = self.anu10

        b_amp = a0 + n0*np.log10(anu*tau0/1e6/anu10)
        b_scale = a1*np.power(anu*tau0/1e6/anu10, n1)
        b_index = a2 + n2*np.log10(anu*tau0/1e6/anu10)
        b = b_amp*(1.0 + np.power(vratio/b_scale, b_index))
        return b

    def cfunc(self, a0, mu0, sigma0, n0, n1):
        vratio = np.abs(self.vratio)
        tau0 = self.tau0
        anu = self.anu
        anu10 = self.anu10
        
        c = (a0 - n0*np.log10(anu*tau0/1e6/anu10)) * np.exp( -np.power(np.abs(np.log10(vratio) - mu0), n1)/sigma0 )

        return c

    def alphafunc(self, a0, a1, n0, n1, n2):
        vratio = np.abs(self.vratio)
        tau0 = self.tau0
        anu = self.anu
        anu10 = self.anu10
        
        x_alpha = vratio/np.power(anu*tau0, n0)
        alpha = a0/(1.0 - np.power(x_alpha, n1) + a1*np.power(x_alpha, n2))
        return alpha

    def Afunc(self, a0, a1, a2, n0, n1, n2, n3):
        vratio = np.abs(self.vratio)
        tau0 = self.tau0
        anu = self.anu
        anu10 = self.anu10

        A_scale = a0*np.power(anu*tau0/1e6/anu10, n0)
        A = a1*(1.0 + np.power(vratio/a2, n1) + np.power(vratio/A_scale, n3))/(1.0 + np.power(vratio/a2, n1 + n2))

        return A

    def betafunc(self, a0, a1, n0):
        vratio = np.abs(self.vratio)
        tau0 = self.tau0
        anu = self.anu
        anu10 = self.anu10

        x_beta = vratio/np.power(anu*tau0, n0)
        beta = a0*x_beta/(1.0 + a1*x_beta)
        return beta

    def fitfunc_vgrad_t16(self, x, a, b, c, alpha, A, beta, vratio = 1.0, tau0 = 1e6, T = 10.0, geo_flag = 0):
        anu = anu_voigt(T)
        x0 = 0.5*vratio
        lam = np.sqrt(2*np.pi**3/27)/anu/tau0
        if geo_flag == 0:
            gam = np.sqrt(6)/2*vratio/tau0*np.sqrt(2*np.pi/27)/anu/3
            xt = b*lam*np.power(np.abs(x), alpha)
            return A*(1.0 + erf(-gam*(x/a)**3/np.power(np.abs(x/a), beta) + c))/np.cosh(0.5*xt)*(x - x0)**2
        elif geo_flag == 1:
            gam = np.sqrt(6)/2*vratio/tau0*np.sqrt(2*np.pi/27)/anu*2/3
            xt = b*lam*np.power(np.abs(x), alpha)
            return A*(1.0 + erf(-gam*(x/a)**3/np.power(np.abs(x/a), beta) + c))\
                    *(1.92 + 1.32*np.exp(-xt) - 0.609*np.exp(-2*xt))\
                    /(np.exp(-0.25*xt) + 0.5*np.exp(0.75*xt) + 0.5*np.exp(-1.25*xt))\
                    *(x - x0)**2
        elif geo_flag == 2:
            gam = np.sqrt(6)/2*vratio/tau0*np.sqrt(2*np.pi/27)/anu
            xt = b*lam*np.power(np.abs(x), alpha)
            return A*(1.0 + erf(-gam*(x/a)**3/np.power(np.abs(x/a), beta) + c))*np.pi/(1 + np.cosh(xt))*(x - x0)**2

    def spec_jointfit(self, x, vratio = 1.0, tau0 = 1e6, T = 10.0, geo_flag = 0):
        anu = anu_voigt(T)
        self.set_param(vratio = vratio, tau0 = tau0, T = T, geo_flag = geo_flag)
        a = self.afunc(*self.a_fit)
        b = self.bfunc(*self.b_fit)
        c = self.cfunc(*self.c_fit)
        alpha = self.alphafunc(*self.alpha_fit)
        A = self.Afunc(*self.A_fit)
        beta = self.betafunc(*self.beta_fit)
        ratio = self.fitfunc_vgrad_t16(x, a, b, c, alpha, A, beta, vratio = vratio,
                                       tau0 = tau0, T = T, geo_flag = geo_flag)
        xcoeff = np.sqrt(6)/12/np.sqrt(np.pi)/anu/tau0
        return ratio*xcoeff

