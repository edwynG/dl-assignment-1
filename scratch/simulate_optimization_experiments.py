import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Estilo para publicación limpia
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

os.makedirs('report/images', exist_ok=True)
os.makedirs('build/images', exist_ok=True)

# 1. Definición de la superficie objetivo
def f_true(x, y):
    u = x**2 + y**2
    u = np.maximum(u, 1e-6)
    return np.sin(0.8 * u) / (u**0.9)

# Generación del conjunto de datos sobre una cuadrícula regular
np.random.seed(42)
grid_x = np.linspace(-2.5, 2.5, 25)
grid_y = np.linspace(-2.5, 2.5, 25)
GX, GY = np.meshgrid(grid_x, grid_y)
X_train = np.stack([GX.ravel(), GY.ravel()], axis=1) # (625, 2)
Y_clean = f_true(X_train[:, 0], X_train[:, 1])[:, None] # (625, 1)

# 2. Implementación de MLP con 2 capas en NumPy
class MLP:
    def __init__(self, in_dim=2, hidden_dim=32, out_dim=1, seed=42):
        np.random.seed(seed)
        self.W0 = np.random.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b0 = np.zeros((1, hidden_dim))
        self.W1 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, out_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, out_dim))

    def forward(self, X):
        self.f0 = X @ self.W0 + self.b0
        self.h1 = np.maximum(0, self.f0) # ReLU
        self.f1 = self.h1 @ self.W1 + self.b1
        self.h2 = np.maximum(0, self.f1) # ReLU
        self.f2 = self.h2 @ self.W2 + self.b2
        return self.f2

    def backward(self, X, Y_pred, Y_true):
        N = X.shape[0]
        grad_f2 = 2.0 * (Y_pred - Y_true) / N
        grad_W2 = self.h2.T @ grad_f2
        grad_b2 = np.sum(grad_f2, axis=0, keepdims=True)

        grad_h2 = grad_f2 @ self.W2.T
        grad_f1 = grad_h2 * (self.f1 > 0)
        grad_W1 = self.h1.T @ grad_f1
        grad_b1 = np.sum(grad_f1, axis=0, keepdims=True)

        grad_h1 = grad_f1 @ self.W1.T
        grad_f0 = grad_h1 * (self.f0 > 0)
        grad_W0 = X.T @ grad_f0
        grad_b0 = np.sum(grad_f0, axis=0, keepdims=True)

        return [grad_W0, grad_b0, grad_W1, grad_b1, grad_W2, grad_b2]

    def get_params(self):
        return [self.W0, self.b0, self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params):
        self.W0, self.b0, self.W1, self.b1, self.W2, self.b2 = params

# 3. Función de entrenamiento con diferentes optimizadores
def train_model(optimizer_name='adam', lr=0.01, epochs=100, batch_size=32,
                noise_std=0.0, hidden_dim=32, seed=42):
    np.random.seed(seed)
    Y_noisy = Y_clean + np.random.normal(0, noise_std, Y_clean.shape) if noise_std > 0 else Y_clean
    N = X_train.shape[0]
    model = MLP(hidden_dim=hidden_dim, seed=seed)
    
    # Estados de optimizadores
    params = model.get_params()
    velocities = [np.zeros_like(p) for p in params]
    m_adam = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]
    
    history_loss = []
    history_time = []
    start_time = time.time()
    
    beta_mom = 0.9
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t_step = 0
    
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_noisy[indices]
        
        # Batch size handling
        bs = N if optimizer_name == 'batch_gd' else batch_size
        
        for i in range(0, N, bs):
            X_b = X_shuffled[i:i+bs]
            Y_b = Y_shuffled[i:i+bs]
            
            Y_pred = model.forward(X_b)
            grads = model.backward(X_b, Y_pred, Y_b)
            params = model.get_params()
            t_step += 1
            
            new_params = []
            for j in range(len(params)):
                p = params[j]
                g = grads[j]
                
                if optimizer_name in ['sgd', 'batch_gd']:
                    p_new = p - lr * g
                elif optimizer_name == 'momentum':
                    velocities[j] = beta_mom * velocities[j] + (1 - beta_mom) * g
                    p_new = p - lr * velocities[j]
                elif optimizer_name == 'adam':
                    m_adam[j] = beta1 * m_adam[j] + (1 - beta1) * g
                    v_adam[j] = beta2 * v_adam[j] + (1 - beta2) * (g**2)
                    m_hat = m_adam[j] / (1 - beta1**t_step)
                    v_hat = v_adam[j] / (1 - beta2**t_step)
                    p_new = p - lr * m_hat / (np.sqrt(v_hat) + eps)
                new_params.append(p_new)
                
            model.set_params(new_params)
            
        # Evaluar pérdida sobre el dataset completo al final de cada época
        pred_full = model.forward(X_train)
        loss = np.mean((pred_full - Y_clean)**2)
        history_loss.append(loss)
        history_time.append(time.time() - start_time)
        
    return np.array(history_loss), np.array(history_time)

# =========================================================================
# EXPERIMENTO 1: Impacto del Ruido (Pregunta 1)
# =========================================================================
epochs_exp = 120
loss_n0, _ = train_model('adam', lr=0.01, epochs=epochs_exp, noise_std=0.0)
loss_n1, _ = train_model('adam', lr=0.01, epochs=epochs_exp, noise_std=0.1)
loss_n2, _ = train_model('adam', lr=0.01, epochs=epochs_exp, noise_std=0.3)

# =========================================================================
# EXPERIMENTO 2: Comparativa de Optimizadores (Pregunta 2 y 4)
# =========================================================================
loss_gd, time_gd = train_model('batch_gd', lr=0.1, epochs=epochs_exp, batch_size=625)
loss_sgd, time_sgd = train_model('sgd', lr=0.05, epochs=epochs_exp, batch_size=32)
loss_mom, time_mom = train_model('momentum', lr=0.05, epochs=epochs_exp, batch_size=32)
loss_adam, time_adam = train_model('adam', lr=0.01, epochs=epochs_exp, batch_size=32)

# =========================================================================
# EXPERIMENTO 3: Impacto de la Dimensión de la Red (Pregunta 3)
# =========================================================================
loss_small, _ = train_model('adam', lr=0.01, epochs=epochs_exp, hidden_dim=4)
loss_med, _ = train_model('adam', lr=0.01, epochs=epochs_exp, hidden_dim=32)
loss_large, _ = train_model('adam', lr=0.01, epochs=epochs_exp, hidden_dim=128)

# =========================================================================
# GENERACIÓN DE FIGURAS PARA EL INFORME
# =========================================================================

# --- FIGURA 2: Superficie teórica y efecto del ruido ---
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)

# Panel A: Superficie teórica en 3D
ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
axes[0].remove() # Reemplazar por 3D
x_dense = np.linspace(-2.5, 2.5, 100)
y_dense = np.linspace(-2.5, 2.5, 100)
XD, YD = np.meshgrid(x_dense, y_dense)
ZD = f_true(XD, YD)
surf = ax_3d.plot_surface(XD, YD, ZD, cmap='viridis', edgecolor='none', alpha=0.9)
ax_3d.set_title('Superficie teórica $f(x, y)$', fontsize=11, fontweight='bold', pad=6)
ax_3d.set_xlabel('x', labelpad=2)
ax_3d.set_ylabel('y', labelpad=2)
ax_3d.set_zlabel('f(x,y)', labelpad=2)
ax_3d.view_init(elev=30, azim=45)

# Panel B: Curvas de pérdida con diferentes niveles de ruido
ax_noise = axes[1]
epochs_arr = np.arange(1, epochs_exp + 1)
ax_noise.plot(epochs_arr, loss_n0, 'b-', linewidth=2.0, label=r'Sin ruido ($\sigma = 0.0$)')
ax_noise.plot(epochs_arr, loss_n1, 'g--', linewidth=1.8, label=r'Ruido moderado ($\sigma = 0.1$)')
ax_noise.plot(epochs_arr, loss_n2, 'r-.', linewidth=1.8, label=r'Ruido alto ($\sigma = 0.3$)')
ax_noise.set_title('Impacto del ruido en el entrenamiento', fontsize=11, fontweight='bold')
ax_noise.set_xlabel('Épocas')
ax_noise.set_ylabel('Pérdida MSE')
ax_noise.set_yscale('log')
ax_noise.legend(loc='upper right', frameon=True, fontsize=9)
ax_noise.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('report/images/figura_2_superficie.png', bbox_inches='tight')
plt.close()
print("Saved figura_2_superficie.png")

# --- FIGURA 3: Comparativa de optimizadores y dimensión de la red ---
fig, (ax_opt, ax_dim) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)

# Panel A: Optimizadores (pérdida vs épocas)
ax_opt.plot(epochs_arr, loss_gd, 'r-', linewidth=2.0, label=r'Descenso del gradiente')
ax_opt.plot(epochs_arr, loss_sgd, color='darkorange', linestyle='--', linewidth=1.8, label=r'Descenso del gradiente estocástico')
ax_opt.plot(epochs_arr, loss_mom, 'b-.', linewidth=1.8, label=r'Momentum (acelerado)')
ax_opt.plot(epochs_arr, loss_adam, 'g-', linewidth=2.2, label=r'Adam (adaptativo, más rápido)')
ax_opt.set_title('Comparativa de optimizadores', fontsize=11, fontweight='bold')
ax_opt.set_xlabel('Épocas')
ax_opt.set_ylabel('Pérdida MSE')
ax_opt.set_yscale('log')
ax_opt.legend(loc='upper right', frameon=True, fontsize=8.5)
ax_opt.grid(True, linestyle='--', alpha=0.6)

# Panel B: Impacto de la dimensión de la red
ax_dim.plot(epochs_arr, loss_small, 'm--', linewidth=1.8, label=r'Red compacta (subparametrizada)')
ax_dim.plot(epochs_arr, loss_med, 'b-', linewidth=2.0, label=r'Red intermedia')
ax_dim.plot(epochs_arr, loss_large, 'g-', linewidth=2.2, label=r'Red sobreparametrizada')
ax_dim.set_title('Impacto de la dimensión de la red', fontsize=11, fontweight='bold')
ax_dim.set_xlabel('Épocas')
ax_dim.set_ylabel('Pérdida MSE')
ax_dim.set_yscale('log')
ax_dim.legend(loc='upper right', frameon=True, fontsize=8.5)
ax_dim.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('report/images/figura_3_optimizadores.png', bbox_inches='tight')
plt.close()
print("Saved figura_3_optimizadores.png")
