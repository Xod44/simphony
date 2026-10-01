## Compilo il codice per il risonatore ad anello
## usiamo componenti delle librerie siepic

# %%  --- LIBRARIES ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sax
from simphony.libraries import siepic
from simphony.classical import ClassicalSim

from functools import partial


# %%  --- RING CUSTOM CIRCUIT ---

'''
    ---------
     Crea un risonatore ad anello completo con terminator.
    le porte del circuito sono ('through', 'drop', 'in', 'add').

    ---------
    wl : ArrayLike
        le lunghezze a cui simulare in µm;
    radius : float
        il raggio del risonatore in µm.
    
'''

ring_factory, info = sax.circuit(
    netlist={
        "instances": {
            "ring1": "half_ring",
            "ring2": "half_ring",
        },
        "connections": {
            "ring1,port_1": "ring2,port_3",
            "ring2,port_1": "ring1,port_3",
        },
        "ports": {
            "in": "ring1,port_2",
            "through": "ring1,port_4",
            "drop": "ring2,port_2",
            "add": "ring2,port_4",
        }
    },
    models={
        "half_ring": partial(siepic.half_ring, gap=100, radius=10, width=500, thickness=220),
        # partial(sipann.half_ring, wl=wl, width=500, thickness=220, radius=radius, gap=100),
    }
)
    # Return the composite model.
   # return ring_factory(wl=wl, radius=radius)

# %%  --- WAVELENGHTS SET ---

wl = np.linspace(1.5, 1.6, 500)


# %%  --- SCATTERING PARAMETERS SET ---

ring1 = ring_factory(wl=wl, radius=10)

# %%  --- PLOT ---

plt.plot(wl, np.abs(ring1['in', 'through'])**2, label="through")
# plt.plot(wl, np.abs(ring1['add', 'drop'])**2, label="in")
plt.plot(wl, np.abs(ring1['in', 'drop'])**2, label="drop")
plt.plot(wl, np.abs(ring1['in', 'add'])**2, label="add")
plt.title("10-micron Ring Resonator")
plt.xlabel("Wavelength (microns)")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# %%  --- PROVA CON LASER ---

''' impostiamo un laser con potenza 1mW alla porta "in" e ricaviamone l'uscita sulla porta add e through'''

sim = ClassicalSim(ckt=ring_factory, wl=wl, radius=10)
laser = sim.add_laser(ports=["in"], power=1.0)
detector = sim.add_detector(ports=["add"])
detector = sim.add_detector(ports=["through"])
result = sim.run()
result.detectors["add"].plot()
plt.grid()
result.detectors["through"].plot()
plt.grid()


# %%  --- ADD-DROP FILTER WITH RING ---  NO NEED !

# %%  --- ADD-DROP CUSTOM CIRCUIT ---

filter_cir, info = sax.circuit(
    netlist={
        "instances": {
            "ring1": "ring",
            "ring2": "ring",
            "ring3": "ring",
        },
        "connections": {
            "ring1,through": "ring2,in",
            "ring2,through": "ring3,in",
        },
        "ports": {
            "in": "ring1,in",
            "out1": "ring1,drop",
            "out2": "ring2,drop",
            "out3": "ring3,drop",
        },
    },
    models={
        "ring": ring_factory,
    }
)

# %%  --- WAVELENGHTS SET ---

wl = np.linspace(1.52, 1.6, 1000)


# %%  --- SCATTERING PARAMETERS SET ---

S = filter_cir(wl=wl, ring1={"radius": 3}, ring2={"radius": 5}, ring3={"radius": 10})

# %%  --- PLOT ---

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 3)

# Plot the three outputs over the full simulated wavelength range.
ax = fig.add_subplot(gs[0, :2])
ax.plot(wl, np.abs(S['in', 'out1'])**2, label="out1", lw=0.7)
ax.plot(wl, np.abs(S['in', 'out2'])**2, label="out2", lw=0.7)
ax.plot(wl, np.abs(S['in', 'out3'])**2, label="out3", lw=0.7)
ax.set_ylabel("Fractional Optical Power")
ax.set_xlabel("Wavelength (um)")
ax.legend(loc="upper right")
ax.grid()

# Plot the three outputs over a restricted wavelength range.
ax = fig.add_subplot(gs[0, 2])
ax.plot(wl, np.abs(S['in', 'out1'])**2, label="out1", lw=0.7)
ax.plot(wl, np.abs(S['in', 'out2'])**2, label="out2", lw=0.7)
ax.plot(wl, np.abs(S['in', 'out3'])**2, label="out3", lw=0.7)
ax.set_xlim(1.557, 1.565)
ax.set_ylabel("Fractional Optical Power")
ax.set_xlabel("Wavelength (um)")
ax.grid()

plt.suptitle("Ring Filter")
fig.align_labels()
plt.show()

# %%  ---  WHY IT MATTERS  ---

# siepic.half_ring?