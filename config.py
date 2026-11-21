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

### NUOVO DI CLAUDE
CBF_ALPHA1 = 2.0            # [1/s] più alto = meno conservativo, richiede più accelerazione
CBF_ALPHA2 = 2.0
CBF_SLACK_WEIGHT = 1e4      # penalità sulla violazione dei vincoli (slack)
CBF_NEIGHBOR_RANGE = 5.0 * MIN_DISTANCE
CBF_MAX_NEIGHBORS = 8       # considera solo i più vicini
OBSTACLE_MARGIN = 1.0       # distanza extra dal bordo dell'ostacolo
OBSTACLES = []              # lista di (centro np.array(3), raggio)

K_POS = 0.5                 # nominale: guadagno sulla posizione
K_VEL = 2.0                 # nominale: guadagno sulla velocità
V_CRUISE = 0.8 * MAX_VELOCITY