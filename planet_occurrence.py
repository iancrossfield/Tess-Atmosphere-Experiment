from scipy import integrate
from scipy import interpolate
import tools
import numpy as np


def h12_p24(period, freq_percutoff=50, log10=False):  
    if log10:       period = 10**period
    period = np.array([period]).ravel()
    ret = 0.25 * 0.0640 * period**0.27 * (1. - np.exp(-(period/7.0)**2.6))
    ret[period>freq_percutoff] = 0.25*0.184
    return ret

def h12_p48(period, freq_percutoff=50, log10=False):  
    if log10:       period = 10**period
    period = np.array([period]).ravel()
    ret = 0.25 * 0.0020 * period**0.79 * (1. - np.exp(-(period/2.2)**4.0))
    ret[period>freq_percutoff] = 0.25*0.044
    return ret

def h12_p832(period, freq_percutoff=50, log10=False): 
    if log10:       period = 10**period
    period = np.array([period]).ravel()
    ret =  0.25 * 0.0025 * period**0.37 * (1. - np.exp(-(period/1.7)**4.1))
    ret[period>freq_percutoff] = 0.25*0.0106
    return ret






def setup_howard2012(maxper=365, nper=1000, freq_percutoff=50):
    """
    :INPUTS:
      maxper : scalar
         maximum simulated period, in days. 

      nper : int
         Resolution of period distribution

      freq_percutoff : scalar
          f(P) flattens out at this orbital period

    """
    per = np.logspace(np.log10(0.5), np.log10(maxper), nper)
    prob24 = h12_p24(per, freq_percutoff=freq_percutoff)
    prob48 = h12_p48(per, freq_percutoff=freq_percutoff)
    prob832 = h12_p832(per, freq_percutoff=freq_percutoff)
    
    teff_mod = np.concatenate(([0], np.linspace(3000, 10000, 200), [50000]))
    f24_temp = 0.165 - 0.081 * (teff_mod - 5100.)/1000.
    f24_temp[teff_mod <= 3600.] = np.interp(3600, teff_mod, f24_temp)
    f24_temp[teff_mod >= 7100.] = np.interp(7100, teff_mod, f24_temp)
    teff_interp = interpolate.interp1d(teff_mod, f24_temp)
    stats = dict(per=per, prob24=prob24, prob48=prob48, prob832=prob832, \
                 teff_mod=teff_mod, f24_temp=f24_temp, teff_interp=teff_interp)
    return stats





def sampleplanet(teff, ndraw=1, sampsmall=0, freq_percutoff=50, sshj=True, mdwarf_bonfils=False, howard=None):
    """
    :INPUTS:
      sampsmall : scalar
        Planets 1-2 Re frequency.   0: none. 1: 1-2Re same as 2-4Re. 2: double, etc.

      freq_percutoff : scalar
          f(P) flattens out at this orbital period

    """
    if howard is None: howard = setup_howard2012(freq_percutoff=freq_percutoff)
    
    prob24 = howard['prob24']
    prob48 = howard['prob48']
    prob832 = howard['prob832']
    teff_interp = howard['teff_interp']
    #f24_temp = howard['f24_temp']
    per = howard['per']
    teff_mod = howard['teff_mod']
        
    if mdwarf_bonfils and teff<=3600: # Use Bonfils+2013 statistics:
        msini, periods = tools.sample_2dcdf(fbon, log10(mbon), log10(pbon), nsamp=ndraw*fbon.sum())
        periods = 10**periods
        sizes = (10**msini)**(1./2.06)
        sizes[sizes > 1.3*an.rjup/an.rearth] = 1.3*an.rjup/an.rearth
    else:  # Use Howard et al. 2012 statistics:
        # Frequencies of planets with P<50 days:
        f48, f832 = 0.017, 0.0079
        ##if teff<=3600: f48, f832 = 1e-4, 1e-4
        #f24 = np.interp(teff, teff_mod, f24_temp)
        f24 = teff_interp(teff)
        f24 = f24 * integrate.quad(h12_p24, np.log10(per.min()), np.log10(per.max()), args=(freq_percutoff, True))[0] / integrate.quad(h12_p24, np.log10(per.min()), np.log10(50), args=(freq_percutoff, True))[0]
        f48 = f48 * integrate.quad(h12_p48, np.log10(per.min()), np.log10(per.max()), args=(freq_percutoff, True))[0] / integrate.quad(h12_p48, np.log10(per.min()), np.log10(50), args=(freq_percutoff, True))[0]
        f832= f832* integrate.quad(h12_p832,np.log10(per.min()), np.log10(per.max()), args=(freq_percutoff, True))[0] / integrate.quad(h12_p832,np.log10(per.min()), np.log10(50), args=(freq_percutoff, True))[0]
        #f48 = f48 * integrate.quad(p48, p.min(), p.max())[0] / integrate.quad(p48, p.min(), 50)[0]
        #f832= f832* integrate.quad(p832,p.min(), p.max())[0] / integrate.quad(p832,p.min(), 50)[0]
        if (not sshj) and teff < 4600:
            f832 = 0.
        expectation_value = f24 + f48 + f832
        n832 = np.round(f832 * ndraw)
        n48 = np.round(f48 * ndraw)
        n24 = np.round(f24 * ndraw)
        size24 =  np.random.uniform((2), (4), size=int(n24))
        size48 =  np.random.uniform((4), (8), size=int(n48))
        size832 = np.random.uniform((8), (32), size=int(n832))
        per24  = 10**tools.sample_1dcdf(prob24,  np.log10(per), nsamp=int(n24))
        per48  = 10**tools.sample_1dcdf(prob48,  np.log10(per), nsamp=int(n48))
        per832 = 10**tools.sample_1dcdf(prob832, np.log10(per), nsamp=int(n832))
        if sampsmall>0:
            size12 =  np.random.uniform(1, 2, size=int(n24*sampsmall))
            per12  = 10**tools.sample_1dcdf(prob24,  np.log10(per), nsamp=int(n24*sampsmall))
        else:
            size12, per12 = [], []
        sizes   = np.concatenate((size12, size24, size48, size832))
        periods = np.concatenate((per12, per24,  per48,  per832))
    return sizes, periods
