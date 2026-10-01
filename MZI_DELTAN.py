## Compilo il modello per il MZI composto di uno splitter, due guide d'onda e un combiner
## uso componenti dalla libreria siepic e ideal
## le gude saranno solo ideal

# %%  --- LIBRARIES ---

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax  ## dizionario matrice scattering e circuiti
from simphony.libraries import ideal,siepic

# %%  --- MZI CUSTOM CIRCUIT ---

mzi, info = sax.circuit(
    netlist={
        "instances": {
            "splitter": "ybranch",
            "wg_l": "wg",
            "wg_s": "wg",
            "combiner": "ybranch",
        },
        "connections": {
            "splitter,port 2": "wg_l,o0",
            "splitter,port 3": "wg_s,o0",
            "wg_l,o1": "combiner,port 2",
            "wg_s,o1": "combiner,port 3",
        },
        "ports": {
            "in": "splitter,port 1",
            "out": "combiner,port 1",
        },
    },
    models={
        "ybranch": siepic.y_branch,
        "wg": ideal.waveguide,
    }
)

# %%  --- SET ---       neff = 2.34   ng = 3.4

wl = jnp.linspace(1.5, 1.6, 8000) ## tra 1.5 e 1.6 μm con 1000 campioni

S = mzi(wl=wl, wg_l={"neff": 2.7,"ng": 3.2, "length": 100.0}, wg_s={"length": 100.0})

mag = jnp.abs(S["out", "in"])**2 ## S_21

# %%  --- PLOT ---

fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(wl, mag)
axs[0].set_ylabel("Transmissione")
axs[1].plot(wl, 10*jnp.log10(mag))
axs[1].set_ylabel("Transmissione (dB)")
axs[1].set_xlabel("Lunghezza d'onda (µm)")
axs[0].grid()
axs[1].grid()
plt.suptitle("Risposta MZI")
plt.show()