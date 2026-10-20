# Apuntes generados

Este directorio contiene un documento LaTeX unificado y su PDF para los
apuntes de *Foundations of Machine Learning and Deep Learning*.

## Regeneración

Desde PowerShell:

```powershell
Set-Location -LiteralPath "C:\Users\ruth_\Desktop\CLASE\MASTER\Foundations of machine learning and deep learning\foundations-notes\apuntes_generados"
python .\generate_notes.py
```

El script lee las fuentes desde `..\apuntes`, `..\slides` y `..\code`, genera
los capítulos y ejecuta XeLaTeX dos veces sobre `main.tex`. No requiere
Pandoc ni modifica las fuentes originales.

## Archivos principales

- `main.tex`: documento principal.
- `template.tex`: plantilla académica compartida.
- `generate_notes.py`: generador reproducible.
- `PLAN.md`: análisis, índice y estrategia de fuentes.
- `main.pdf`: PDF final generado por XeLaTeX.

Los archivos auxiliares de LaTeX se eliminan al finalizar la generación.
