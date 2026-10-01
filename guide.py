## Compilo un codice di esempio per ricavare i parametri di scattering di guide d'onda poste in cascata

# %% --- LIBRARIES ---

import numpy as np
import sax
from simphony.libraries import siepic

# %% --- DEFINE WAVEGUIDE ---

## consideriamo guide d'onda senza perdite

wg1 = siepic.waveguide(wl=1.55, length=550, width=480, height=210)
## genera un dizionario di parametri di scattering per il dispositivo appena definito
S21 = wg1['o1','o0'] ## isoliamo il paramtetro S_21
print(S21)

## possiamo cambiare la geometria per ottenere e verificare che cambia il parametro

wg2 = siepic.waveguide(wl=1.55, length=550, width=840, height=220)
S21 = wg2['o1','o0']
print(S21)

# %% --- WAVEGUIDE CASCADE CIRCUIT ---

## possiamo infine dichiarare un circuito composto di guide d'onda in cascata e ottenere i parametri di scattering del sistema.

cascade, info = sax.circuit(
    netlist={
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "wg3": "waveguide",
            },
        "connections": {
            "wg1, o1": "wg2, o0",
            "wg2, o1": "wg3, o0",
            },
        "ports": {
            "in": "wg1, o0",
            "out": "wg3, o1",
            },
        },
    models={
        "waveguide": siepic.waveguide,
        }
    )

# %% --- GETTING S-PARAM ---

S = cascade(wl=1.55, wg1={"loss": 1.3, "length": 720, "width": 420, "height": 210}, 
            wg2={"loss": 1.5, "length": 550, "width": 420, "height": 210}, 
            wg3={"loss": 1.3, "length": 200, "width": 420, "height": 210})
print("Parametro S_21 = ", S["out","in"])
mag = np.abs(S["out", "in"])**2
print("Trasmittanza = ", mag)