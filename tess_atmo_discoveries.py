import pandas as pd
import numpy as np
import planet_occurrence as po
import os
import analysis as an
import tess_yield as ty
import progressbar
import sys
import gc
"""
Currently setting eccentricity to zero

T14s calculated assuming eccentricity is zero

2017-09-06 11:40 IJMC: Documentation/comments added
"""

nmc = 10000 # Number of Monte Carlo planets per system. Needs to be >~1e4 
tessDetectionSN = 15  # Don't even both if TESS can't get this S/N on it 
biggestInterestingPlanet = 6. #Biggest planet we care about (I like Neptunes)

#input/output 
_ticdir = os.path.expanduser('~/proj/tess/tic/')
_ticfn = _ticdir + 'tic_dec90_00S__88_00S.csv'

_outdir = os.path.expanduser('~/proj/tess/atmoexperiment/') 

def getTESSprecision(imag):
    """In ppm per hr^{-1/2}"""
    crudepolyfit = np.array([ -1.28518197e-05,   5.92161792e-04,   7.60303196e-03, -9.40552495e-02,   1.99984004e+00])
    return 10**np.polyval(crudepolyfit, imag)

def getTESScoverage(eclat, eclong):
    skycoverages = np.array([(78, 351), (65, 54), (7, 27), (0, 0)])
    np.abs(eclat)
    skycoveragebin = (np.abs(eclat) >=skycoverages[:,0]).nonzero()[0][0]
    duration = skycoverages[skycoveragebin, 1]
    return duration

def estimatePlanetParams(teff, rstar, per, a, rp, redist=0.3, ab=0.2, mrmode='wolfgang2016'):
    teq = teff/np.sqrt(215.1*a/rstar) * (redist * (1. - ab))**0.25
    if mrmode.lower()=='wolfgang2016':
        c, gamma, sm1, beta = 2.7, 1.3, 1.9, 0.
        c, gamma, sm1, beta = 1.6, 1.8, 2.9, 0.
        mp0 = c*rp**gamma
        u_mp0 = np.sqrt(sm1**2 + beta*(rp-1))
        mp = np.random.normal(mp0, u_mp0)
    res = pd.DataFrame(dict(per=per, Rplanet=rp))
    res['a']   = a
    res['Teq'] = teq
    res['Mplanet']  = mp
    res['gplanet']  = 9.8*mp/res.Rplanet**2
    res['Teff'] = teff
    res['Rstar'] = rstar
    return res

# Statistics for planet occurrence
howardstats = po.setup_howard2012(maxper=365)

# Columns to load from TIC:
ticcols = ['rad' ,'lumclass' ,'mass', 'eclat' ,'Tmag' ,'Jmag', 'Vmag', 'rho' ,'Teff', 'name', 'ID']

# Load the entire TESS Input Catalog
tic = pd.read_csv(_ticfn, usecols=ticcols)

# Set our basic target requirements to downsize the catalog:
req1 = tic.rad > 0
req2 = tic.lumclass=='DWARF'
req3 = tic.mass > 0
ticind =  np.logical_and(req1, req2)
ticind =  np.logical_and(req3, ticind)
validtic = tic[ticind]
nvalid = len(validtic)

#In the future, check if this file exists and skip loading the entire catalog
outfile0 = _ticfn.replace(_ticdir, _outdir).replace('.csv', '_TSNR=%i.csv' % (tessDetectionSN))
validtic.to_csv(outfile0, index=False)


# TESS params: How long does TESS observe it, and what photometric
#   precision is achieved?
obsDurations = np.array([getTESScoverage(validtic.iloc[ii].eclat, None) for ii in range(nvalid)])
tessnoise = getTESSprecision(validtic.Tmag)

# Find stars for which we can detect ANYTHING interesting for
# P=1d. Then downsize catalog again.
minduration = 3.91 * (0.1 / validtic.rho)**0.3333
canDetectAnything = (1e6 * (biggestInterestingPlanet*an.rearth/(validtic.rad*an.rsun))**2) >= (tessDetectionSN * (tessnoise / (minduration * obsDurations**0.5)))
validtic = validtic[canDetectAnything]
nvalid = len(validtic)


# Loop through all stars in remaning catalog:
results = []
allresults = []
ii = 0

print "Starting to iterate through all %i stars:" % nvalid
pbar = progressbar.ProgressBar(widgets= \
                   ['Processed: ', progressbar.Counter(), '/%i stars ('%nvalid , progressbar.Timer(), ')'])

for ii in pbar(range(nvalid)): #1200): #nvalid):   
    
    # Monte Carlo set of radii and periods (and other planet params):
    thisr, thisp = po.sampleplanet(validtic.iloc[ii].Teff, ndraw=nmc, howard=howardstats)
    nthisplanet = thisr.size
    thisecc = np.zeros(nthisplanet, float)
    thisa = ((thisp/365.)**2 * validtic.iloc[ii].mass)**0.3333

    # Figure out which fake planets transit >=2 times (b/c of period),
    # and which transit at all (b/c of geometry):
    transitprob = (validtic.iloc[ii].rad*an.rsun/(thisa*an.AU)) / (1. - thisecc**2)
    thesetransit = np.random.uniform(size=nthisplanet) <= transitprob
    thesetransit2x = np.logical_and(thesetransit,  (((obsDurations[ii] / thisp) - 1) >= np.random.uniform(size=nthisplanet)))

    transitdepth = np.zeros(nthisplanet, float)
    transitdepth[thesetransit2x] = (thisr[thesetransit2x]*an.rearth/(validtic.iloc[ii].rad*an.rsun))**2
    theseimpactparams = np.abs(np.random.uniform(-0.9, 0.9, size=thesetransit2x.sum()))

    transitpers = thisp[thesetransit2x]
    # Use relation from G. Ricker's TESS paper:
    thesedurations = 3.91 * ((transitpers/10.) / validtic.iloc[ii].rho)**0.3333 * \
                     np.sqrt(1. - theseimpactparams**2)

    ## Estimate Multiple-Event-Statistic (essentially S/N on each
    ## planet's folded transit):
    ntransitInTESS = np.floor(obsDurations[ii] / transitpers)
    MES = (transitdepth[thesetransit2x]*1e6) / (tessnoise[ii] / (ntransitInTESS * thesedurations)**0.5)
    theseplanets = MES > tessDetectionSN

    nfoundhere = transitpers[theseplanets].size

    thisres = pd.DataFrame(dict(p=transitpers[theseplanets], a=thisa[thesetransit2x][theseplanets], \
                                r=thisr[thesetransit2x][theseplanets]))
    results.append(thisres)

    # Infer yet more planet parameters for our fake population:
    planetparams = estimatePlanetParams(validtic.iloc[ii].Teff, validtic.iloc[ii].rad, \
                                        thisres.p, thisres.a, thisres.r)
    ty.atmoparams(planetparams);
    
    # Apply CK2017 relation to see which JWST/NIRISS could observe:
    a = ((1.3 - (planetparams.Teq - 360)**0.0035 + 0.25*planetparams.mh/planetparams.Teq) * \
         (planetparams.gplanet/10.) * (planetparams.Rstar*6.955)**2/(planetparams.Rplanet*0.638))**2
    fstar = 10**(-0.4*(validtic.Jmag.iloc[ii]-10))
    planetparams['niriss_hr'] = a**2 / np.sqrt(fstar)
    planetparams['ind'] = ii
    planetparams['Tmag'] = validtic.Tmag.iloc[ii]
    planetparams['name'] = validtic.name.iloc[ii]
    allresults.append(planetparams)
        
allresults = pd.concat(allresults)

outfile = _ticfn.replace(_ticdir, _outdir).replace('.csv', '_nmc=%i_TSNR=%i.csv' % (nmc, tessDetectionSN))
allresults.to_csv(outfile, index=False)


## Now run basic summary analysis on the whole run.

maxniriss = 20 # maximum # of JWST hours we're willing to invest

ind = (allresults.Rplanet < biggestInterestingPlanet) * (allresults.niriss_hr < maxniriss)
potential_targets = np.unique(allresults.iloc[ind.nonzero()].name)
#potential_targets = np.unique(allresults.name)
target_likelihood = 1.0*np.array([(allresults.iloc[ind.nonzero()].name==pottar).sum() for pottar in potential_targets]) / nmc
#target_likelihood = 1.0*np.array([(allresults.name == pottar).sum() for pottar in potential_targets]) / nmc

good_results = pd.concat([validtic.iloc[(validtic.name==pottar).nonzero()] for pottar in potential_targets])
good_results['likelihood'] = target_likelihood

outfile2 = outfile.replace('.csv', ('_best_R-lt-%1.1f_t-lt-%i' % (biggestInterestingPlanet, maxniriss)).replace('.',',')) + '.csv'
good_results.to_csv(outfile2, index=False)




#  150 7.4
#  300 12.1
#  600 21.3
# 1200 40.8 44.0
