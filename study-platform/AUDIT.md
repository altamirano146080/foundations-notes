# Auditoría de fuentes

## Inventario realizado

- **Slides:** 21 PDFs en `foundations-notes/slides/`, desde `01_01_Introduction1.pdf` hasta `13_Recommender_systems.pdf`.
- **Código:** 29 notebooks Jupyter en `foundations-notes/code/`, con prácticas de modelos clásicos, PyTorch, regresión, MLP, regularización, optimización, CNN, RNN, atención y recomendadores.
- **Apuntes:** 11 capítulos LaTeX (`chapter_01.tex` a `chapter_11.tex`) y `PLAN.md`. El capítulo 07 corresponde al Tema 8 y el capítulo 11 al Tema 13, por lo que la numeración de capítulos no se fuerza a coincidir con el número de la unidad.
- **Material complementario:** informes y figuras de las carpetas `02_04_ShortAssignment`, `assigment` y `assigment8`.

## Mapa de conocimiento

1. Fundamentos y modelos clásicos → `Practice_01`.
2. Ensembles y multiclase → `01_04_Ensembles.ipynb`.
3. Regresión lineal → `03_01_Linear_regression.ipynb`.
4. Regresión logística → `04_01_Logistic_regression.ipynb`.
5. Perceptrones multicapa → `05_MLP_1`, `05_02_MLP`, `05_04_MLP`.
6. Regularización → `06_Regularization_01.ipynb`.
7. Descenso de gradiente y optimizadores → tres notebooks de `08_01_GradientDescent`.
8. CNN → tres notebooks de `09_02_CNN`.
9. Secuencias y RNN → `10_01_SequentialData` y `10_02_RNNfromScratch`.
10. Atención y Transformers → notebooks `11_01`–`11_04`.
11. Sistemas de recomendación → `13_01_Recommender_systems.ipynb`.

## Trazabilidad y limitaciones

La interfaz enlaza a cada slide deck, capítulo y notebook real. Las referencias se expresan por archivo porque no se han inventado números de diapositiva ni líneas de notebook. La aplicación ofrece una primera capa pedagógica transversal; los diagramas, fórmulas completas, salidas ejecutadas y todos los ejercicios de las fuentes siguen consultándose en los originales.

La visualización de descenso de gradiente es un ejemplo didáctico creado para estudiar y no una ejecución del notebook. La web no modifica ni ejecuta las prácticas originales. El progreso es local al navegador y no contiene datos sensibles ni sincronización entre dispositivos.

## Verificación

- Se comprobó el inventario de extensiones, PDFs, notebooks, capítulos y encabezados.
- La aplicación no añade dependencias externas.
- Las rutas de los enlaces se construyen desde la carpeta `study-platform` hacia los materiales existentes.
- Se debe abrir `index.html` o servir la carpeta estáticamente para verificar navegación, búsqueda, quizzes, tema oscuro y persistencia.
