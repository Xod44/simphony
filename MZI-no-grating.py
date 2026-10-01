## Compilo il modello per il MZI composto di uno splitter, due guide d'onda e un combiner
## uso componenti dalla libreria siepic

# %%  --- LIBRARIES ---

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax  ## dizionario matrice scattering e circuiti
from simphony.libraries import siepic

# %%  --- CUSTOM SHIFTER ---

# def custom_shifter(param: float = 0.5) -> sax.SDict:
#     """This model will have one parameter, param, which defaults to 0.5.
    
#     Args:
#         param: Some float parameter.

#     Returns:
#         sdict: A dictionary of scattering matrices.
#     """
#     # a simple wavelength independent s-matrix
#     sdict = sax.reciprocal({
#         ("in0", "out0"): -1j * np.sqrt(param),

#     %%  jnp.conj()

#     })
#     return sdict
    
# # A model is simulated by "calling" it with appropriate paraeters.

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
        "wg": siepic.waveguide,
    }
)

# %%  --- COMMENTI ---

''' La netlist è un dizionario composto di 3 campi: 
    istanze (cosa vi legge l'operatore), 
    connessioni (interne tra le porte dei componenti), 
    porte esposte;
    si poggia sul dizionario che definisce i modelli (nomi istanze -> libreria.modello)
    
    le modifiche al dizionario le si può passare dizionari di argomento e valore in una funzione, __lo valuto successivamente__
    
    I parametri che vogliamo cambiare nl circuito sono esclusivamente le lunghezze delle guide dei due bracci del MZI
'''
# %%  --- INSERTION LOSS AT 1.55 um  ---

# wl = 1.55
# S = mzi(wl=wl, wg_l={"length": 150.0}, wg_s={"length": 159.5})
# IR = -20*jnp.log10(jnp.abs(S["out","in"]))
# print("Insertion Loss =",IR)


# %%  --- WAVELENGHTS SET ---

wl = jnp.linspace(1.5, 1.6, 1000) ## tra 1.5 e 1.6 μm con 1000 campioni

# %%  --- SCATTERING PARAMETERS SET ---

S = mzi(wl=wl, wg_l={"length": 290.0}, wg_s={"length": 293.5})

# %%  --- TRANSMITTANCE SET ---

'''Per leggere la potenza trasmessa dall'input all' output del dispositivo 
operiamo il quadrato dell'ampiezza del paramtero di interesse.'''

mag = jnp.abs(S["out", "in"])**2 ## S_21

# %%  --- PLOT ---

fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(wl, mag)
axs[0].set_ylabel("Transmissione")
axs[1].plot(wl, 10*jnp.log10(mag))
axs[1].set_ylabel("Transmissione (dB)")
axs[1].set_xlabel("Lunghezza d'onda (µm)")
plt.suptitle("Risposta MZI")
plt.show()

# %%  --- LASER SIMULATION ---

'''Un altro metodo di simulazione prevede l'uso di classi, 
ad esempio possiamo fare una simulazione di tipo sweep 
su un range di lunghezze d'onda impostando un laser ideale a monte 
del dispositivo e leggendone l'uscita'''

# %%  --- LOAD LIBRARY ---
from simphony.classical import ClassicalSim

# %%  --- SET SIMULATION PARAMETERS ---
sim = ClassicalSim(ckt=mzi, wl=wl, wg_l={"length":150.0}, wg_s={"length":159.5})
laser = sim.add_laser(ports=["in"], power=1.0)
detector = sim.add_detector(ports=["out"])

# %%  --- RUN SIMULATION ---

result = sim.run()
result.detectors["out"].plot()

# %% --- WAVEGUIDE FROM LIBRARY IDEAL ---

from simphony.libraries import ideal

# %%  --- NEW MZI CUSTOM CIRCUIT ---

mzi_NI, info = sax.circuit(
    netlist={
        "instances": {
            "splitter": "ybranch",
            "wg_1": "wg_siepic",
            "wg_2": "wg_ideal",
            "combiner": "ybranch",
        },
        "connections": {
            "splitter,port 2": "wg_1,o0",
            "splitter,port 3": "wg_2,o0",
            "wg_1,o1": "combiner,port 2",
            "wg_2,o1": "combiner,port 3",
        },
        "ports": {
            "in": "splitter,port 1",
            "out": "combiner,port 1",
        },
    },
    models={
        "ybranch": siepic.y_branch,
        "wg_siepic": siepic.waveguide,
        "wg_ideal": ideal.waveguide,
    }
)

# %%  --- WAVELENGHTS SET ---

wl = jnp.linspace(1.5, 1.6, 1000) ## tra 1.5 e 1.6 μm con 1000 campioni

# %%  --- SCATTERING PARAMETERS SET ---

S = mzi_NI(wl=wl, wg_1={"length": 100.0}, wg_2={"length": 106.50})

# %%  --- TRANSMITTANCE SET ---

mag = jnp.abs(S["out", "in"])**2 ## S_21

# %%  --- PLOT ---

fig, axs = plt.subplots(2,1, sharex=True)
axs[0].plot(wl, mag)
axs[0].set_ylabel("Transmissione")
axs[1].plot(wl, 10*jnp.log10(mag))
axs[1].set_ylabel("Transmissione (dB)")
axs[1].set_xlabel("Lunghezza d'onda (µm)")
plt.suptitle("Risposta MZI")
plt.show()

# %%  --- LASER SIMULATION ---

wl = jnp.linspace(1.5,1.6,800)
sim = ClassicalSim(ckt=mzi, wl=wl, wg_1={"length":100.0}, wg_2={"length":106.50})
laser = sim.add_laser(ports=["in"], power=1.0)
detector = sim.add_detector(ports=["out"])

result = sim.run()
result.detectors["out"].plot()

