import numpy as np
import tqdm
from scipy.integrate import dblquad
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import NearestNDInterpolator as NearND
import pandas as pd
from astropy import units as u

# Load Koens data; Lx_z contains Lx and z values for sample
# sf_flux contains flux and coverage for selection function
Lx_z = pd.read_csv('data/koens_2013/warps_xlf.dat', delim_whitespace=True, header=0)
sf_flux_coverage = pd.read_csv('data/koens_2013/sf_fluxobex_rc100kpc.dat', delim_whitespace=True)
sf_flux_coverage.to_csv('data/koens_2013/sf_fluxobex_rc100kpc.csv', index=False)
sf_coverage = sf_flux_coverage["coverage"]
sf_flux = sf_flux_coverage["flux"] 

# Load k-correction data, contains kcorr values for Lx and z
kcorr = pd.read_csv('data/koens_2013/kcorr_lz_large.dat', delim_whitespace=True, header=0)
k_z = kcorr["z"]
k_Lx = kcorr["Lx"]
k_kcorrect = kcorr["kcorr"]
# Create an interpolator for k-correction, use NearND as k_Lx and k_z are not ordered
kcorr_interp = NearND(list(zip(k_Lx, k_z)), k_kcorrect, rescale = False, tree_options = None)

# Filter Lx_z for z <= 0.3 as we're only interested in Local Universe
z_mask = Lx_z['z'] <= 0.3
Lx_mask = Lx_z["Lx"][z_mask] * 1e44

# Create bins for Lx, using logspace to cover the range of Lx values
num_bins = 15
bin_edges = np.logspace(np.log10(Lx_mask.min()*0.9), np.log10(Lx_mask.max()*1.1), num_bins + 1)

# Create an interpolator for the luminosity distance and comoving volume over full z range
z_fullrange = np.arange(0.02, 0.31, 0.01) # np.arange ignores upper point so put 0.31
co_mo_vol_range = cosmo.differential_comoving_volume(z_fullrange).value
co_mo_vol_range_interp = interp1d(z_fullrange, co_mo_vol_range)
DL_cm_range = cosmo.luminosity_distance(z_fullrange).to(u.cm).value
DL_cm_range_interp = interp1d(z_fullrange, DL_cm_range)

# Create an interpolator for the selection function
coverage_interp1d = interp1d(sf_flux*1e-14, sf_coverage, bounds_error = False, fill_value = ([0], [100]))

def pc_integrand(z, Luminosity):
    """
    Integrand for the Carrera method, calculates the comoving volume and flux for a given redshift and luminosity.
    """
    co_mo_vol_step = co_mo_vol_range_interp(z)
    
    DL_cm_step = DL_cm_range_interp(z)
    flux = ( Luminosity / (4 * np.pi * DL_cm_step**2) ) * kcorr_interp(Luminosity, z)
    # Koens uses flux limit for cluster sample, else use coverage interpolation for sel func
    flim = 6.5e-14
    if flux >= flim:
        coverage_interp = coverage_interp1d(flux)
    else:
        coverage_interp = 0
    return (coverage_interp * co_mo_vol_step)

# Calculate the Page & Carrera method's phi for each bin
pc_phis = []
pc_deltas = []
for i, lower_bin_edge in enumerate(tqdm.tqdm(bin_edges[:-1])):
    upper_bin_edge = bin_edges[i+1]
    indecis_in_bin = np.where((lower_bin_edge <= Lx_mask) & (Lx_mask < upper_bin_edge))[0]
    delta_L = upper_bin_edge - lower_bin_edge
    pc_deltas.append(delta_L)
    # Integrate over the bin edges and z range 0.02 to 0.3
    pc_denoms, pc_denoms_err = dblquad(pc_integrand, lower_bin_edge, upper_bin_edge, 0.02, 0.3)
    clust_in_bin = len(indecis_in_bin)
    pc_phi = clust_in_bin / pc_denoms
    pc_phis.append(pc_phi)
    
# Store the results in a DataFrame with bin centres and edges
pc_bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
pc_df = pd.DataFrame({
    'bin_centre': pc_bin_centres,
    'bin_edge_lower': bin_edges[:-1],
    'bin_edge_upper': bin_edges[1:],
    'phi': pc_phis,
    'delta_L': pc_deltas
})
pc_df.to_csv('results_carrera_xlf.csv', index=False)

# plotting the results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(pc_df['bin_centre'], pc_df['phi'], label='Page & Carrera Method', color='blue', marker='o')
plt.xscale('log')
plt.xlabel('Luminosity (erg/s)')
plt.ylabel('Phi (Mpc^-3 dex^-1)')
plt.title('X-ray Luminosity Function using Page & Carrera Method')
plt.legend()
plt.grid()
plt.savefig('xlf_page_carrera.png')
plt.show()