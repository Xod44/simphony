# %%  --- LIBRARIES ---


import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax  ## dizionario matrice scattering e circuiti
from simphony.libraries import siepic

# %%  --- MZI CUSTOM CIRCUIT ---

MZI_COUP, info = sax.circuit(
    netlist={
        "instances": {
            "ACCOP1": "coupler",
            #"wg_1": "wg_1",
            #"wg_2": "wg_2",
            "ACCOP2": "coupler",
        },
        "connections": {
            "ACCOP1,port_3": "ACCOP2,port_1",
            "ACCOP1,port_4": "ACCOP2,port_2",
            #"coupler_in,port_3": "wg_1,o0",
            #"coupler_in,port_4": "wg_2,o0",
            #"wg_1,o1": "coupler_out,port_1",
            #"wg_2,o1": "coupler_out,port_2",
        },
        "ports": {
            "in0": "ACCOP1,port_1",
            "in1": "ACCOP1, port_2",
            "out0": "ACCOP2,port_3",
            "out1": "ACCOP2, port_4",
        },
    },
    models={
        "coupler": siepic.directional_coupler,
        #"wg_1": ideal.waveguide,
        #"wg_2": ideal.waveguide,
    }
)

# %%  --- CHOOSING A LENGHT FOR DESTRUCTIVE INTERFERENCE ---


# %%  --- WAVELENGHTS SET ---

wl = jnp.linspace(1.5, 1.6, 1000)

# %%  --- SCATTERING PARAMETERS SET ---

S = MZI_COUP(wl = wl)

# %%  --- TRANSMITTANCE SET o0-i0 ---

mag1 = jnp.abs(S["out0", "in0"])**2 ##

# %%  --- PLOT ---

fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(wl, mag1)
axs[0].set_ylabel("Transmittanza")
axs[1].plot(wl, 10*jnp.log10(mag1))
axs[1].set_ylabel("Transmittanza (dB)")
axs[1].set_xlabel("Lunghezza d'onda (µm)")
plt.suptitle("Risposta MZI BAR")
plt.grid()
plt.show()

# %%  --- TRANSMITTANCE SET o1-i0 ---

mag2 = jnp.abs(S["out1", "in0"])**2 ##

# %%  --- PLOT ---

fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(wl, mag2)
axs[0].set_ylabel("Transmittanza")
axs[1].plot(wl, 10*jnp.log10(mag2))
axs[1].set_ylabel("Transmittanza (dB)")
axs[1].set_xlabel("Lunghezza d'onda (µm)")
plt.suptitle("Risposta MZI CROSS")
plt.show()
