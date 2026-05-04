"""
Physical constants for QENS analysis.

"""

from __future__ import annotations


# Neutron mass in kg
NEUTRON_MASS_KG = 1.67493e-27


# Reduced Planck constant in J·s
# ħ (J·s)
HBAR_JS = 1.05457e-34


#  meV >> J conversion
# 1ev = 1.602176634 × e-19 J
# 1mev = e-3 ev  = 1.602176634 x e-22 J
MEV_TO_J = 1.60218e-22


# Reduced Planck constant in meV·ps — working constant for QENS
# ħ (meV·ps)
HBAR_MEV_PS = 0.65821


# FWHM = 2 * sqrt(2 * ln 2) * sigma  for a Gaussian
GAUSSIAN_FWHM_FACTOR = 2.3548200450309493
