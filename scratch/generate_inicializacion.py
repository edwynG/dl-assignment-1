"""
=============================================================================
Script: generate_figura_4.py
Descripción:
    Este script reproduce y grafica el experimento del impacto de la varianza
    de inicialización en redes neuronales profundas (K = 20 capas ocultas),
    tal como se analiza en la Sección II-C y en el Notebook práctico de la Tarea 1.

Fenómeno evaluado:
    1. Paso hacia adelante (Forward pass):
       Evolución de la varianza de las preactivaciones Var[f_k] capa por capa.
       Bajo ReLU, E[h^2] = 0.5 * Var[f], por lo que para preservar la varianza
       se requiere sigma_w^2 = 2 / D_h (criterio de He / Kaiming).
    2. Paso hacia atrás (Backward pass):
       Evolución de la desviación estándar del gradiente std[dL/dh_k] al retropropagar.
       Muestra el desvanecimiento o explosión del gradiente según la escala de los pesos.

Regímenes comparados:
    - Varianza reducida (0.2 * He): La señal y los gradientes colapsan a cero (~10^-8).
    - Varianza calibrada (He: 2/D_h): Señal y gradientes se mantienen estables (~10^0).
    - Varianza excesiva (3.0 * He): La señal y los gradientes explotan exponencialmente (>10^8).

Salida:
    report/images/figura_4_inicializacion.png (Gráfica en escala logarítmica, 300 DPI).
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Configuración de parámetros y reproducibilidad
# -----------------------------------------------------------------------------
# Fijamos la semilla pseudoaleatoria para garantizar resultados reproducibles
np.random.seed(30326271)

# Parámetros de la arquitectura de la red profunda:
K_deep = 20        # Número de capas ocultas (profundidad suficiente para ver desvanecimiento/explosión)
D_deep = 80        # Ancho de las capas ocultas (número de neuronas D_h en cada capa)
n_samples = 1000   # Número de muestras de entrada para estimar la varianza empírica de f_k

# Varianza teórica ideal de He para activaciones ReLU: sigma^2 = 2 / D_h
sigma_sq_ideal = 2.0 / D_deep

# Definición de los tres esquemas de inicialización a contrastar:
# Cada tupla contiene: (varianza sigma_Omega^2, color de línea, marcador, estilo de línea)
schemes = {
    r'$\sigma_\Omega^2$ reducida ($0.2 \times \mathrm{He}$)': (0.2 * sigma_sq_ideal, '#d9534f', 'o', '--'),
    r'$\sigma_\Omega^2$ calibrada ($\mathrm{He}: 2/D_h$)': (sigma_sq_ideal, '#0275d8', 's', '-'),
    r'$\sigma_\Omega^2$ excesiva ($3.0 \times \mathrm{He}$)': (3.0 * sigma_sq_ideal, '#f0ad4e', '^', '-.')
}

# -----------------------------------------------------------------------------
# 2. Simulación del paso hacia adelante (Forward Pass)
# -----------------------------------------------------------------------------
# Diccionario para almacenar la lista de varianzas Var[f_k] de cada esquema
var_f_results = {}

for name, (s_sq, color, marker, ls) in schemes.items():
    # Reiniciamos la semilla para usar exactamente las mismas realizaciones aleatorias
    np.random.seed(30326271)
    
    # Estructuras para almacenar matrices de pesos y vectores de sesgo de cada capa
    # Capa 0: entrada (1 dimensión) -> primera capa oculta (D_deep neuronas)
    # Capas 1 a K-1: capas intermedias (D_deep -> D_deep neuronas)
    # Capa K: capa oculta final -> salida (1 dimensión)
    weights = [None] * (K_deep + 1)
    biases = [None] * (K_deep + 1)
    
    # Capa de entrada
    weights[0] = np.random.normal(size=(D_deep, 1)) * np.sqrt(s_sq)
    biases[0] = np.zeros((D_deep, 1))
    
    # Capas ocultas intermedias
    for l in range(1, K_deep):
        weights[l] = np.random.normal(size=(D_deep, D_deep)) * np.sqrt(s_sq)
        biases[l] = np.zeros((D_deep, 1))
        
    # Capa final de salida
    weights[K_deep] = np.random.normal(size=(1, D_deep)) * np.sqrt(s_sq)
    biases[K_deep] = np.zeros((1, 1))
    
    # Generamos un lote de entradas sintéticas x ~ N(0, 1) con forma (1, n_samples)
    data_in = np.random.normal(size=(1, n_samples))
    h = data_in
    var_list = []
    
    # Propagación hacia adelante capa por capa
    for l in range(K_deep):
        # Preactivación afín: f = W * h + b
        f = biases[l] + np.matmul(weights[l], h)
        
        # Calculamos y registramos la varianza de las preactivaciones en este nivel
        var_list.append(np.var(f))
        
        # Función de activación no lineal ReLU: h = max(0, f)
        h = np.maximum(0, f)
        
    # Salida lineal final (capa K_deep)
    f_final = biases[K_deep] + np.matmul(weights[K_deep], h)
    var_list.append(np.var(f_final))
    
    var_f_results[name] = var_list

# -----------------------------------------------------------------------------
# 3. Función del paso hacia atrás (Backward Pass / Retropropagación)
# -----------------------------------------------------------------------------
def backward_pass(all_weights, all_biases, all_f, all_h, y):
    """
    Calcula los gradientes de la función de pérdida respecto a las activaciones
    dL/dh_k en cada capa oculta mediante la regla de la cadena.
    
    Parámetros:
        all_weights : lista de matrices de pesos de cada capa.
        all_biases  : lista de vectores de sesgo de cada capa.
        all_f       : lista de preactivaciones f_k guardadas durante el forward pass.
        all_h       : lista de activaciones post-ReLU h_k.
        y           : valor objetivo de salida (etiqueta escalar).
        
    Retorna:
        all_dl_dh   : lista con los gradientes dL/dh en cada capa oculta.
    """
    K = len(all_weights) - 1
    all_dl_df = [None] * (K + 1)
    all_dl_dh = [None] * (K + 1)
    
    # 1. Gradiente de la pérdida cuadrática L = (f_salida - y)^2 respecto a f_final:
    #    dL/df_K = 2 * (f_K - y)
    all_dl_df[K] = 2.0 * (all_f[K] - y)
    
    # 2. Retropropagación desde la penúltima capa hasta la primera capa oculta
    for layer in range(K - 1, -1, -1):
        # Gradiente respecto a la activación: dL/dh_{layer+1} = W_{layer+1}^T * dL/df_{layer+1}
        all_dl_dh[layer + 1] = np.matmul(all_weights[layer + 1].T, all_dl_df[layer + 1])
        
        # Gradiente a través de la no linealidad ReLU:
        # La derivada de ReLU(f) es 1 si f > 0, y 0 si f <= 0
        all_dl_df[layer] = np.copy(all_dl_dh[layer + 1])
        all_dl_df[layer][all_f[layer] < 0] = 0
        
    # Gradiente en la entrada
    all_dl_dh[0] = np.matmul(all_weights[0].T, all_dl_df[0])
    return all_dl_dh

# -----------------------------------------------------------------------------
# 4. Simulación del paso hacia atrás (Retropropagación de gradientes)
# -----------------------------------------------------------------------------
# Diccionario para almacenar la desviación estándar del gradiente en cada capa
grad_results = {}

for name, (s_sq, color, marker, ls) in schemes.items():
    # Reiniciamos la semilla para evaluar bajo la misma muestra
    np.random.seed(30326271)
    
    w_deep = [None] * (K_deep + 1)
    b_deep = [None] * (K_deep + 1)
    
    w_deep[0] = np.random.normal(size=(D_deep, 1)) * np.sqrt(s_sq)
    b_deep[0] = np.zeros((D_deep, 1))
    for l in range(1, K_deep):
        w_deep[l] = np.random.normal(size=(D_deep, D_deep)) * np.sqrt(s_sq)
        b_deep[l] = np.zeros((D_deep, 1))
    w_deep[K_deep] = np.random.normal(size=(1, D_deep)) * np.sqrt(s_sq)
    b_deep[K_deep] = np.zeros((1, 1))
    
    # Muestra de prueba para la pasada hacia adelante y cálculo del gradiente
    x_test = np.random.normal(size=(1, 1))
    all_f = [None] * (K_deep + 1)
    all_h = [None] * (K_deep + 1)
    all_h[0] = x_test
    
    for l in range(K_deep):
        all_f[l] = b_deep[l] + np.matmul(w_deep[l], all_h[l])
        all_h[l+1] = np.maximum(0, all_f[l])
    all_f[K_deep] = b_deep[K_deep] + np.matmul(w_deep[K_deep], all_h[K_deep])
    
    # Etiqueta objetivo arbitraria
    y_test = np.array([[1.0]])
    
    # Ejecución de la retropropagación
    all_dl_dh = backward_pass(w_deep, b_deep, all_f, all_h, y_test)
    
    # Registramos la desviación estándar del gradiente dL/dh en cada capa oculta (1 a K)
    grad_stds = [np.std(all_dl_dh[l]) for l in range(1, K_deep + 1)]
    grad_results[name] = grad_stds

# -----------------------------------------------------------------------------
# 5. Generación de la gráfica comparativa (Figura 4)
# -----------------------------------------------------------------------------
# Creamos una figura de 2 paneles (Paso adelante y Paso atrás)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.3), dpi=300)

# Panel 1: Evolución de la varianza en el paso adelante
layers = np.arange(K_deep + 1)
for name, (s_sq, color, marker, ls) in schemes.items():
    ax1.plot(layers, var_f_results[name], label=name, color=color, 
             marker=marker, markersize=3, linestyle=ls, linewidth=1.3)

ax1.set_yscale('log')
ax1.set_xlabel('Capa $k$', fontsize=8)
ax1.set_ylabel(r'$\mathrm{Var}[f_k]$ (escala log)', fontsize=8)
ax1.set_title(r'Paso adelante: varianza $\mathrm{Var}[f_k]$', fontsize=8.5, fontweight='bold')
ax1.grid(True, which='both', linestyle=':', alpha=0.5)
ax1.legend(fontsize=7, loc='best')

# Panel 2: Evolución de la magnitud del gradiente en el paso atrás
layers_back = np.arange(1, K_deep + 1)
for name, (s_sq, color, marker, ls) in schemes.items():
    ax2.plot(layers_back, grad_results[name], label=name, color=color, 
             marker=marker, markersize=3, linestyle=ls, linewidth=1.3)

ax2.set_yscale('log')
ax2.set_xlabel('Capa $k$', fontsize=8)
ax2.set_ylabel(r'$\mathrm{std}[\partial L / \partial h_k]$ (escala log)', fontsize=8)
ax2.set_title(r'Paso atrás: gradiente $\mathrm{std}[\partial L / \partial h_k]$', fontsize=8.5, fontweight='bold')
ax2.grid(True, which='both', linestyle=':', alpha=0.5)
ax2.legend(fontsize=7, loc='best')

# Ajuste fino de márgenes y exportación en alta resolución
plt.tight_layout()
output_path = 'report/images/figura_4_inicializacion.png'
plt.savefig(output_path, dpi=300)
print(f'Figura 4 generada exitosamente y guardada en: {output_path}')
