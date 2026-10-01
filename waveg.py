## Compilo il codice per verificare le differenze tra i modelli di guide d'onda
## usiamo componenti delle librerie siepic e ideal

# %%  --- LIBRARIES ---

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax  ## dizionario matrice scattering e circuiti
from simphony.libraries import siepic, ideal

# %%  --- WAVEGUIDE SIEPIC ---

wl = 1.55

wg1 = siepic.waveguide(wl=wl, length = 900, width = 400, height = 210)


wg2 = siepic.waveguide(wl=wl, length= 900, width= 500, height= 230)

print(wg1["o0","o1"])

print(wg2["o0","o1"])
