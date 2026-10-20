# Plan de generación de apuntes universitarios

## Alcance

Se generará un único documento académico en español a partir de las fuentes
locales del repositorio. Los archivos originales no se modifican:

- `slides/`: 21 barajas PDF, desde introducción y fundamentos matemáticos
  hasta sistemas de recomendación.
- `code/`: 29 notebooks Jupyter con prácticas, experimentos, gráficos y
  salidas ejecutadas.
- `apuntes/`: 9 notas Markdown personales para los temas 1, 2, 3–4, 5, 6,
  8, 9, 10 y 11.

## Índice propuesto

1. **Fundamentos del aprendizaje automático y modelos clásicos**
   - Paradigmas de aprendizaje, particiones de datos y evaluación.
   - Naive Bayes, árboles, vecinos próximos y clustering.
   - Prácticas de `Practice_01`.
2. **Ensembles y descomposición multiclase**
   - Bagging, boosting, voting, stacking y OVA/OVO.
   - Prácticas de ensembles y clasificación multiclase.
3. **Regresión lineal**
   - Modelo, función de coste, solución analítica y descenso de gradiente.
   - Superficie de pérdida, convergencia y validación.
4. **Regresión logística**
   - Sigmoide, entropía cruzada, frontera de decisión y regularización.
   - Features polinomiales y validación cruzada.
5. **Perceptrones multicapa**
   - Forward/backward pass, activaciones, logits y clasificación multiclase.
   - `nn.Linear`, `nn.Sequential` y autograd.
6. **Regularización en redes profundas**
   - L1/L2, weight decay, dropout, early stopping y data augmentation.
   - Lectura de curvas de entrenamiento y validación.
7. **Descenso de gradiente y optimizadores**
   - SGD, momentum, Nesterov, AdaGrad, RMSProp y Adam.
   - Schedulers y comparación de convergencia.
8. **Redes convolucionales**
   - Convolución, padding, stride, pooling, LeNet y clasificación de imágenes.
   - Convolución desde cero y `nn.Conv2d`.
9. **Datos secuenciales y redes recurrentes**
   - Modelado a nivel de carácter, estado oculto, RNN, LSTM y GRU.
   - Explosión/desaparición del gradiente y clipping.
10. **Atención y Transformers**
    - Atención escalada, multi-head attention, codificación posicional,
      encoder y máscaras causales.
11. **Sistemas de recomendación**
    - Tema cubierto por `slides/13_Recommender_systems.pdf` y
      `code/13_01_Recommender_systems/13_01_Recommender_systems.ipynb`.
12. **Apéndice de prácticas**
    - Tabla de correspondencia entre capítulos, diapositivas y notebooks.
    - Herramientas, reproducibilidad y errores habituales.

## Estrategia de generación

1. `generate_notes.py` lee las notas Markdown sin alterar sus archivos.
2. Reutiliza el conversor Markdown→LaTeX ya existente para conservar títulos,
   listas, fórmulas, tablas y bloques de código.
3. Añade a cada capítulo una sección de conexión teoría-práctica con:
   - notebooks relacionados;
   - encabezados detectados;
   - fragmentos de código representativos;
   - salidas textuales y número de gráficos realmente almacenados en el
     notebook.
4. Incluye referencias a las barajas PDF y a los notebooks originales mediante
   rutas relativas.
5. Compila `main.tex` con XeLaTeX dos veces para actualizar el índice.
6. El script elimina únicamente sus propios auxiliares de compilación.

## Herramientas comprobadas

- XeLaTeX y PDFLaTeX: disponibles mediante MiKTeX.
- Python 3.12 y Jupyter: disponibles.
- `nbformat` y `matplotlib`: disponibles.
- `pdftotext`/`pdfinfo`: disponibles para inspección de las diapositivas.
- Pandoc: no instalado y no necesario.

