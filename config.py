import numpy as np

COMMUNICATION_RANGE = 200.0 # rc nel paper e vale 200m
MIN_DISTANCE = 5.0 # delta_min nel paper. Anche questo valore preso dal paper

# ALTITUDINE SU CUI LAVORA IL DRONE IN METRI
Y_MIN = 40.0
Y_MAX = 120.0

# PARAMETRI DEI DRONI IN m/s
MAX_VELOCITY = 15.0
MAX_ACCEL = 2.5

# PARAMETRI TEMPORALI 
DT = 0.05
HORIZON = 0.05

# PARAMETRI PER D-RHC
SAFETY_FACTOR = 1.6
N_NEIGHBORS = 7
BRAKE_GAIN = 1.0

# PARAMETRI DE
DE_POPSIZE = 12
DE_MAXITER = 25

# PESI INIZIALI DEI COSTI
INITIAL_WEIGHTS = { 
    'w_eta': 1.0,
    'w_y': 1.0,
    'w_g': 1.0
}

SEED = 44

# NUOVO DI CLAUDE
CBF_SLACK_QUAD = 1e-3   # piccolo termine quadratico, tiene P ben condizionata
CBF_SLACK_LIN = 50.0    # deve superare il moltiplicatore del vincolo (qui al massimo ~3)
CBF_ALPHA1 = 0.5           # parti da qui e calcola con la formula sopra
CBF_ALPHA2 = 0.5
_A_LIM = MAX_ACCEL / 3 ** 0.5
A_BRAKE = 0.6 * (MAX_ACCEL / 3 ** 0.5)   # decelerazione "di progetto": 60% di quella disponibile per asse
CBF_NEIGHBOR_RANGE = 1.2 * ((2 * MAX_VELOCITY) ** 2 / (4 * _A_LIM) + MIN_DISTANCE * SAFETY_FACTOR)   # ≈ 197 m con i tuoi valori
CBF_MAX_NEIGHBORS = 12     # con 10 droni li considera tutti
OBSTACLE_MARGIN = 1.0       # distanza extra dal bordo dell'ostacolo

_Y_MID = 0.5 * (Y_MIN + Y_MAX)
OBSTACLES = [
    (np.array([100.0, Y_MIN,  60.0]), 20.0),
    (np.array([180.0, Y_MIN, 200.0]), 25.0),
    (np.array([250.0, Y_MIN, 120.0]), 15.0),
    (np.array([ 80.0, Y_MIN, 240.0]), 18.0),
]

K_POS = 0.5                 # nominale: guadagno sulla posizione
K_VEL = 2.0                 # nominale: guadagno sulla velocità
V_CRUISE = 0.8 * MAX_VELOCITY

K_SWIRL = 0.3 * V_CRUISE      # velocità laterale massima aggiunta [m/s]
SWIRL_CONE = 0.5              # cos(60°): conta solo chi è entro 60° dalla mia direzione verso il bersaglio
SWIRL_TAU = 3.0               # [s] memoria dell'attivazione
SWIRL_UP = (0.0, 1.0, 0.0)    # asse "su": lo scarto avviene nel piano perpendicolare