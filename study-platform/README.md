# ML Studio

Plataforma local de estudio para **Foundations of Machine Learning and Deep Learning**. El contenido se mantiene separado de las prácticas, slides y apuntes originales.

## Uso

Abre `index.html` en un navegador moderno. No requiere Node, backend ni conexión a Internet. Para una experiencia más fiable con enlaces a notebooks y PDFs, sirve esta carpeta con cualquier servidor estático:

```powershell
python -m http.server 8000 --directory ".\Foundations of machine learning and deep learning\study-platform"
```

Después visita `http://localhost:8000`.

La aplicación incluye 11 unidades, referencias a 21 barajas PDF, ejercicios de comprensión, búsqueda, modo oscuro, una visualización didáctica de descenso de gradiente y progreso persistido en `localStorage`. La ejecución real de notebooks permanece en Jupyter y no se simula dentro de la web.
