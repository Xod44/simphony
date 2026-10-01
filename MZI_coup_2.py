## Compilo un MZI con accoppiatore
## uso una guida dalla libreria siepic ed una dalla libreria ideal per generare lo sfasamento

# %%  --- LIBRARIES ---


import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax  ## dizionario matrice scattering e circuiti
from simphony.libraries import siepic, ideal

# %%  --- PHASE SHIFTER ---

phase_shift=0*jnp.pi #sfasamento ideale

def phase_shifter(wl=1.55):
    phase = phase_shift
    transmission = jnp.exp(1j * phase)

    return sax.reciprocal({
        ("in0", "out0"): transmission,
    })

# %%  --- MZI CUSTOM CIRCUIT ---

MZI_COUP, info = sax.circuit(
    netlist={
        "instances": {
            "coupler_in": "coupler",
            "wg_1": "wg_1",
            "wg_2": "wg_2",
            "shifter": "shifter",
            "coupler_out": "coupler",
        },
        "connections": {
            "coupler_in,port_3": "wg_1,o0",
            "coupler_in,port_4": "wg_2,o0",
            "wg_1,o1": "shifter, in0",
            "shifter, out0": "coupler_out,port_1",
            "wg_2,o1": "coupler_out,port_2",
        },
        "ports": {
            "in0": "coupler_in,port_1",
            "in1": "coupler_in, port_2",
            "out0": "coupler_out,port_3",
            "out1": "coupler_out, port_4",
        },
    },
    models={
        "coupler": siepic.directional_coupler,
        "wg_1": siepic.waveguide,
        "wg_2": siepic.waveguide,
        "shifter": phase_shifter,
    }
)


# %%  --- WAVELENGHTS SET ---

wl = jnp.linspace(1.5, 1.6, 1000)

# %%  --- SCATTERING PARAMETERS SET ON STATE ---

S = MZI_COUP(wl = wl ,coupler={"length": 17.5}, wg_1={"length": 100.0}, wg_2={"length": 100.0})

# %%  --- TRANSMITTANCE SET o0-i0 ---

mag1 = jnp.abs(S["out0", "in0"])**2 ##

# %%  --- PLOT ---

fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(wl, mag1)
axs[0].set_ylabel("Transmittanza")
axs[1].plot(wl, 10*jnp.log10(mag1))
axs[1].set_ylabel("Transmittanza (dB)")
axs[1].set_xlabel("Lunghezza d'onda (µm)")
axs[0].grid()
axs[1].grid()
plt.suptitle("Risposta MZI BAR")
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
axs[0].grid()
axs[1].grid()
plt.suptitle("Risposta MZI CROSS")
plt.show()

# %%  --- INSERTION LOSS ---

IL1 = 10*jnp.log(1/mag1) 
plt.plot(wl, IL1)
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# %%  --- CROSS TALK ---

CR = 10*jnp.log(mag2/mag1)
plt.plot(wl, CR)
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# %%  --- SCATTERING PARAMETERS SET OFF STATE ---

S = MZI_COUP(wl = wl ,coupler={"length": 17.5}, wg_1={"neff": 2.652, "ng": 3.2,"length": 94.5}, wg_2={"length": 94.6})
