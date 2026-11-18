import matplotlib.pyplot as plt
from collections import defaultdict

# Nome del file di input
filename = "asi_results.txt"

# Struttura per raggruppare i dati: chiave = tris (x, y, z), valore = {'deltas': [], 'asis': []}
traiettorie = defaultdict(lambda: {'deltas': [], 'asis': []})

# Definiamo un finto oggetto numpy per permettere a eval di interpretare np.float64() come un normale float
class DummyNumpy:
    def float64(self, val):
        return float(val)

np = DummyNumpy()

with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # Il formato della riga è: ((... ), Delta): ASI
        # Dividiamo la chiave dal valore usando l'ultimo due punti ':'
        parts = line.rsplit(":", 1)
        if len(parts) != 2:
            continue
        
        chiave_str = parts[0].strip()
        valore_str = parts[1].strip()
        
        try:
            # Usiamo eval passando 'np' nel dizionario locale. 
            # In questo modo np.float64(val) verrà valutato correttamente come float(val).
            chiave_tupla = eval(chiave_str, {"__builtins__": {}}, {"np": np})
            
            # chiave_tupla sarà nella forma: ((x, y, z), Delta)
            tris_xyz = tuple(float(c) for c in chiave_tupla[0])
            delta = float(chiave_tupla[1])
            asi = float(valore_str)
            
            traiettorie[tris_xyz]['deltas'].append(delta)
            traiettorie[tris_xyz]['asis'].append(asi)
            
        except Exception as e:
            print(f"Errore nel parsing della riga: {line} -> {e}")

# Creazione del grafico
plt.figure(figsize=(10, 6))

# Tracciamo una linea per ogni tris (x, y, z) unico
for tris_xyz, dati in traiettorie.items():
    if not dati['deltas']:
        continue
        
    # Ordiniamo i dati in base a Delta per tracciare la linea correttamente da sinistra a destra
    coppie_ordinate = sorted(zip(dati['deltas'], dati['asis']))
    deltas_ordinati = [c[0] for c in coppie_ordinate]
    asis_ordinati = [c[1] for c in coppie_ordinate]
    
    # Etichetta formattata per la legenda
    label_str = f"x={tris_xyz[0]:.2f}, y={tris_xyz[1]:.2f}, z={tris_xyz[2]:.2f}"
    
    plt.plot(deltas_ordinati, asis_ordinati, marker='o', linestyle='-', label=label_str)

# Personalizzazione del grafico
plt.xlabel("Delta", fontsize=12)
plt.ylabel("ASI", fontsize=12)
plt.title("Andamento di ASI in funzione di Delta per ogni tris (x, y, z)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# Gestione della legenda se sono presenti linee
if traiettorie:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.savefig("grafico_asi.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()