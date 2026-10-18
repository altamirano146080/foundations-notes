# Apuntes del Tema 6: Regularización en Redes Profundas

---

## 1. El Problema del Sobreajuste (*Overfitting*) en Redes Profundas

### 1.1 Naturaleza del Problema

* **Alta Capacidad Expresiva:** Las redes neuronales profundas poseen millones de parámetros libres. Esta gran capacidad les permite memorizar no solo los patrones subyacentes de los datos, sino también el ruido de entrenamiento.


* **Brecha de Generalización (*Generalization Gap*):** Se define como la diferencia entre la pérdida/error en el conjunto de validación (o prueba) y el conjunto de entrenamiento.


* *Baja varianza / Buen ajuste:* La pérdida de validación disminuye a la par con la de entrenamiento.


* *Alta varianza / Overfitting:* A partir de cierto punto, la pérdida de entrenamiento sigue bajando mientras la pérdida de validación empieza a crecer, ensanchando la brecha de generalización.




* **Objetivo de la Regularización:** Modificar o restringir el proceso de aprendizaje para reducir el error de prueba/generalización sin incrementar excesivamente el error de entrenamiento.



---

## 2. Decaimiento de Pesos (*Weight Decay* / Regularización $L_2$ vs. $L_1$)

La forma clásica de regularización consiste en añadir un término de penalización $\Omega(W)$ a la función de pérdida original $\mathcal{L}(W)$:

$$\mathcal{J}(W) = \mathcal{L}(W) + \lambda \Omega(W)$$

donde $\lambda > 0$ es el hiperparámetro que controla la intensidad de la regularización.

### 2.1 Decaimiento de Pesos ($L_2$ Regularization)

* **Penalización:** Se utiliza la norma $L_2$ al cuadrado de las matrices de pesos:



$$\Omega(W) = \frac{1}{2} \Vert{}W\Vert{}_2^2 = \frac{1}{2} \sum_{i} \sum_{j} w_{ij}^2$$


* **Efecto en el Gradiente y Actualización (SGD):**

$$\frac{\partial \mathcal{J}}{\partial W} = \frac{\partial \mathcal{L}}{\partial W} + \lambda W$$


$$W \leftarrow W - \eta \frac{\partial \mathcal{J}}{\partial W} = W(1 - \eta \lambda) - \eta \frac{\partial \mathcal{L}}{\partial W}$$


* **Interpretación:** Antes de aplicar el paso de gradiente habitual, los pesos se reducen por un factor multiplicativo $(1 - \eta \lambda) < 1$. Esto fuerza a la red a mantener pesos pequeños y distribuidos de forma suave, previniendo comportamientos extremos frente a pequeñas variaciones en la entrada.



### 2.2 Regularización $L_1$ (Lasso)

* **Penalización:** Suma de los valores absolutos de los pesos:



$$\Omega(W) = \Vert{}W\Vert{}_1 = \sum_{i} \sum_{j} \vert{}w_{ij}\vert{}$$


* **Efecto:** Mantiene una magnitud de penalización constante $\lambda \cdot \text{signo}(W)$ independientemente del tamaño del peso, lo que empuja muchos pesos **exactamente a cero**, generando matrices de pesos esparsas (*sparse models*) y actuando como selección implícita de características.

---

## 3. Dropout (Inyección de Ruido en Activaciones)

Proposed por Srivastava et al. (2014), es una de las técnicas de regularización más efectivas en redes profundas.

### 3.1 Mecanismo de Funcionamiento

Durante el entrenamiento, en cada iteración o minilote, cada neurona oculta se "apaga" (se fuerza a cero) con una probabilidad $p \in (0, 1)$ elegida de forma independiente.

* **Formulación Matemática (Inverted Dropout):**
Dada una activación intermedia $h$, se transforma en $h'$ mediante:

$$h' = \begin{cases} 0 & \text{con probabilidad } p \\ \frac{h}{1-p} & \text{con probabilidad } 1-p \end{cases}$$


* **Reescalado ($\frac{1}{1-p}$):** Escalar por $\frac{1}{1-p}$ durante el entrenamiento garantiza que el valor esperado de la activación sea constante ($E[h'] = h$), evitando modificar la magnitud de las entradas en la fase de test.



```text
  Antes de Dropout                Después de Dropout (con p = 0.4)
    [x1]  [x2]                      [x1]  [x2]
     │ ╲  ╱ │                        │      │
    [h1]  [h2]                     [ 0 ]  [h2]   <- Neurona h1 desactivada
     │ ╱  ╲ │                        │      │
    [o1]  [o2]                      [o1]  [o2]

```

### 3.2 ¿Por qué funciona Dropout?

1. **Prevención de la Co-adaptación:** Fuerza a cada neurona a aprender características útiles por sí misma, en lugar de depender de la presencia corregidora de otras neuronas específicas.


2. **Ensamble Implícito (*Implicit Ensembling*):** Entrenar con Dropout equivale a entrenar en cada paso un subconjunto distinto de la red (*thinned network*) que comparte pesos. La evaluación final sin Dropout equivale geométricamente a promediar las predicciones de un número masivo de subredes.


3. **Inferencia / Test:** Durante la fase de evaluación, el Dropout debe **desactivarse** llamando a `model.eval()` en PyTorch.



---

## 4. Parada Temprana (*Early Stopping*)

Es una técnica de regularización no paramétrica que aprovecha la dinámica temporal del entrenamiento.

### 4.1 Criterio de Funcionamiento

* **Fundamento:** A medida que aumentan las épocas de entrenamiento, el error de entrenamiento disminuye de manera monótona, pero el error de validación alcanza un mínimo y luego empieza a aumentar.


* **Mecanismo:**
1. Se evalúa el error sobre el conjunto de validación al final de cada época.


2. Se guarda una copia de los mejores pesos del modelo cuando la pérdida de validación alcanza un nuevo mínimo.
3. **Criterio de Paciencia (*Patience*):** Si la pérdida de validación no mejora durante un número $P$ determinado de épocas consecutivas, el entrenamiento se interrumpe y se restauran los pesos óptimos guardados.





```text
Pérdida
  │        \         /  <- Pérdida de Validación (Empieza a subir)
  │         \_______/
  │          ▲ Parada Temprana (Guardar mejor modelo)
  │           \
  │            \_______ <- Pérdida de Entrenamiento (Sigue bajando)
  └───────────────────────────── Épocas

```

---

## 5. Práctica y Aplicación en Código (PyTorch)

A continuación se presenta la implementación modular que integra **Weight Decay ($L_2$)**, **Dropout** y **Early Stopping** utilizando PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------
# 1. Definición del Modelo con Dropout
# --------------------------------------------------------------------------
class RegularizedMLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=2, dropout_p=0.5):
        super(RegularizedMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),  # Inyección de ruido en activaciones
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            nn.Linear(hidden_dim, num_classes)  # Logits de salida
        )

    def forward(self, x):
        return self.network(x)

# --------------------------------------------------------------------------
# 2. Clase para Gestionar Early Stopping
# --------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = model.state_dict().copy() # Guardar mejores pesos
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# --------------------------------------------------------------------------
# 3. Bucle de Entrenamiento y Validación
# --------------------------------------------------------------------------
def train_with_regularization():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Datos sintéticos
    X_train = torch.randn(800, 20)
    y_train = torch.randint(0, 2, (800,))
    X_val = torch.randn(200, 20)
    y_val = torch.randint(0, 2, (200,))
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    
    model = RegularizedMLP(input_dim=20, dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Regularización L2 (Weight Decay) aplicada directamente en el optimizador
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    early_stopper = EarlyStopping(patience=10)
    
    epochs = 100
    for epoch in range(1, epochs + 1):
        # FASE DE ENTRENAMIENTO (Dropout Activo)
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # FASE DE VALIDACIÓN (Dropout Desactivado)
        model.eval()  # Importante: desactiva Dropout
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Época [{epoch:3d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Comprobar criterio de Parada Temprana
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"\n[Early Stopping] Entrenamiento detenido en la época {epoch}.")
            # Restaurar mejores pesos
            model.load_state_dict(early_stopper.best_state_dict)
            break

if __name__ == '__main__':
    train_with_regularization()

```
