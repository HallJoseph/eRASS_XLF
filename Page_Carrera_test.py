import numpy as np
import tqdm
from scipy.integrate import dblquad
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import NearestNDInterpolator as NearND
import pandas as pd
from astropy import units as u

Lx_z = pd.read_csv('data/koens_2013/warps_xlf.dat', delim_whitespace=True, header=0)
sf_flux = pd.read_csv('data/koens_2013/sf_fluxobex_rc100kpc.dat', delim_whitespace=True)
sf_flux.to_csv('data/koens_2013/sf_fluxobex_rc100kpc.csv', index=False)
coverage = sf_flux["coverage"]
flux = sf_flux["flux"] 

kcorr = pd.read_csv('data/koens_2013/kcorr_lz_large.dat', delim_whitespace=True, header=0)
k_z = kcorr["z"]
k_Lx = kcorr["Lx"]
k_kcorrect = kcorr["kcorr"]
kcorr_interp = NearND(list(zip(k_Lx, k_z)), k_kcorrect, rescale = False, tree_options = None)

z_mask = Lx_z['z'] <= 0.3
Lx_mask = Lx_z["Lx"][z_mask] * 1e44

num_bins = 15
bin_edges = np.logspace(np.log10(Lx_mask.min()*0.9), np.log10(Lx_mask.max()*1.1), num_bins + 1)

z_fullrange = np.arange(0.02, 0.31, 0.01) # np.arange ignores upper point so put 0.31

co_mo_vol_range = cosmo.differential_comoving_volume(z_fullrange).value
co_mo_vol_range_interp = interp1d(z_fullrange, co_mo_vol_range)

DL_cm_range = cosmo.luminosity_distance(z_fullrange).to(u.cm).value
DL_cm_range_interp = interp1d(z_fullrange, DL_cm_range)

coverage_interp1d = interp1d(flux*1e-14, coverage, bounds_error = False, fill_value = ([0], [100]))


def pc_integrand(z, Luminosity, flux, coverage):
    # 
    co_mo_vol_step = co_mo_vol_range_interp(z)
    
    DL_cm_step = DL_cm_range_interp(z)
    flux_range = ( Luminosity / (4 * np.pi * DL_cm_step**2) ) * kcorr_interp(Luminosity, z)
    #print(flux_range)
    flim = 6.5e-14
    if flux_range >= flim:
        # coverage_interp = np.interp(flux_range, flux*1e-14, coverage, left = 0, right = 100)
        coverage_interp = coverage_interp1d(flux_range)
    else:
        coverage_interp = 0
    # print(coverage_interp* co_mo_vol_interp)
    # raise
    return (coverage_interp * co_mo_vol_step)
# pc_integrand(0.1, 1e43, flux, coverage)

pc_phis = []
pc_deltas = []
for i, lower_bin_edge in enumerate(tqdm.tqdm(bin_edges[:-1])):
    upper_bin_edge = bin_edges[i+1]
    indecis_in_bin = np.where((lower_bin_edge <= Lx_mask) & (Lx_mask < upper_bin_edge))[0]
    # print(indecis_in_bin)
    delta_L = upper_bin_edge - lower_bin_edge
    deltas.append(delta_L)
    # print(Vmax_good[indecis_in_bin])
    # print(lower_bin_edge, upper_bin_edge)
    pc_denoms, pc_denoms_err = dblquad(pc_integrand, lower_bin_edge, upper_bin_edge, 0.02, 0.3, args=(flux, coverage))
    #print(pc_denoms)
    clust_in_bin = len(indecis_in_bin)
    pc_phi = clust_in_bin / pc_denoms
    # new_phi = 1/delta_L * np.sum(1/Vmax_good[indecis_in_bin])
    # print(new_phi)
    pc_phis.append(pc_phi)
    #print(lower_bin_edge, upper_bin_edge)
    #print(Lx_mask[indecis_in_bin])
    
#pc_phis