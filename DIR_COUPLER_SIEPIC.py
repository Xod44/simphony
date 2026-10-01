## Compilo un accoppiatore con lunghezza per accoppiamento al 50% a lambda=1.55 µm
## uso una guida dalla libreria siepic 

# %%  --- LIBRARIES ---


import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax  ## dizionario matrice scattering e circuiti
from simphony.libraries import siepic

# %%  --- COUPLER CUSTOM CIRCUIT ---

tryon, info = sax.circuit(
    netlist={
        "instances": {
            "coupler": "dir_coupler",
            "wg_1": "wg_siepic",
            "wg_2": "wg_ideal",
        },
        "connections": {
            "wg_1,o1": "coupler,port_1",
            "wg_2,o1": "coupler,port_2",
        },
        "ports": {
            "in0": "wg_1,o0",
            "in1": "wg_2,o0",
            "out0": "coupler,port_3",
            "out1": "coupler,port_4",
        },
    },
    models={
        "dir_coupler": siepic.directional_coupler,
        "wg_siepic": siepic.waveguide,
        "wg_ideal": siepic.waveguide,
    }
)

## uso solo guide siepic per prime

# %%  --- WAVELENGHTS SET ---

wl = jnp.linspace(1.5, 1.6, 100) ## tra 1.5 e 1.6 μm con 1000 campioni

# %%  --- SCATTERING PARAMETERS SET ---

S = tryon(wl=wl,coupler={"coupling_length": 17.5}, wg_1={"length": 100.0}, wg_2={"length": 100.0})

# %%  --- TRANSMITTANCE SET o0-i0 ---

mag1 = jnp.abs(S["out0", "in0"])**2 ##

# %%  --- PLOT BAR ---

fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(wl, mag1)
axs[0].set_ylabel("Transmittanza")
axs[1].plot(wl, 10*jnp.log10(mag1))
axs[1].set_ylabel("Transmittanza (dB)")
axs[1].set_xlabel("Lunghezza d'onda (µm)")
axs[0].grid()
axs[1].grid()
plt.suptitle("Risposta accoppiatore direzionale BAR")
plt.show()


# %%  --- TRANSMITTANCE SET o1-i0 ---

mag2 = jnp.abs(S["out1", "in0"])**2 ##

# %%  --- PLOT CROSS ---

fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(wl, mag2)
axs[0].set_ylabel("Transmittanza")
axs[1].plot(wl, 10*jnp.log10(mag2))
axs[1].set_ylabel("Transmittanza (dB)")
axs[1].set_xlabel("Lunghezza d'onda (µm)")
axs[0].grid()
axs[1].grid()
plt.suptitle("Risposta accoppiatore direzionale linea CROSS")
plt.show()

