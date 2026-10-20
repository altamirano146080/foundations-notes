"""Generate a unified Spanish study guide from the repository sources."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def latex_plain(value: str) -> str:
    value = value.replace("\\", r"\textbackslash{}")
    for char, replacement in {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        value = value.replace(char, replacement)
    return value


def inline(value: str) -> str:
    placeholders: list[str] = []

    def hold(text: str) -> str:
        placeholders.append(text)
        return f"\x00{len(placeholders) - 1}\x00"

    value = re.sub(r"(?<!\$)\$([^$\n]+)\$(?!\$)", lambda m: hold(r"\(" + m.group(1) + r"\)"), value)
    value = re.sub(r"\*\*([^*]+)\*\*", lambda m: hold(r"\textbf{" + latex_plain(m.group(1)) + "}"), value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: hold(r"\emph{" + latex_plain(m.group(1)) + "}"), value)
    value = re.sub(r"`([^`]+)`", lambda m: hold(r"\texttt{" + latex_plain(m.group(1)) + "}"), value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: hold(r"\href{" + m.group(2) + "}{" + latex_plain(m.group(1)) + "}"), value)
    value = latex_plain(value)
    token = re.compile("\x00(\\d+)\x00")

    def resolve(match: re.Match[str]) -> str:
        return token.sub(resolve, placeholders[int(match.group(1))])

    return token.sub(resolve, value)


def render_markdown(lines: list[str]) -> str:
    result: list[str] = []
    in_code = False
    code: list[str] = []
    list_kind: str | None = None

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            result.append(rf"\end{{{list_kind}}}")
            list_kind = None

    for raw in lines:
        line = raw.rstrip("\r")
        if line.startswith("```"):
            if in_code:
                result.extend([r"\begin{lstlisting}[language=Python]", *code, r"\end{lstlisting}"])
                code = []
                in_code = False
            else:
                close_list()
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        if not line.strip() or line.strip() == "---":
            close_list()
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            close_list()
            level = min(len(heading.group(1)), 3)
            command = ("section", "subsection", "subsubsection")[level - 1]
            result.append(rf"\{command}{{{inline(heading.group(2))}}}")
            continue
        item = re.match(r"^\s*[-*+]\s+(.*)$", line)
        numbered = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        if item or numbered:
            desired = "itemize" if item else "enumerate"
            if list_kind != desired:
                close_list()
                list_kind = desired
                result.append(rf"\begin{{{desired}}}")
            result.append(r"\item " + inline((item or numbered).group(1)))
            continue
        close_list()
        if line.startswith(">"):
            result.append(r"\begin{quote}" + inline(line[1:].strip()) + r"\end{quote}")
        else:
            result.append(inline(line))
    if in_code:
        result.extend([r"\begin{lstlisting}[language=Python]", *code, r"\end{lstlisting}"])
    close_list()
    return "\n\n".join(result)


TOPICS = [
    {
        "number": 1,
        "title": "Fundamentos del aprendizaje automático y modelos clásicos",
        "markdown": "Apuntes del Tema 1 - Foundations of Machine Learning — Classical Models & Clustering.md",
        "slides": ["01_01_Introduction1.pdf", "01_02_Introduction2.pdf", "01_03_Introduction3.pdf", "01_04_Introduction4.pdf", "01_05_Introduction5.pdf", "02_MathematicalFoundations.pdf"],
        "notebooks": ["code/Practice_01/01_01_warm-up_and_first_model.ipynb", "code/Practice_01/01_02_Naive_Bayes.ipynb", "code/Practice_01/01_03_Decision_trees.ipynb", "code/Practice_01/01_05_Clustering.ipynb"],
        "bridge": "Las prácticas de Practice_01 llevan los conceptos de representación, clasificación y clustering a datos ejecutables. Los gráficos y salidas se describen únicamente cuando están almacenados en los notebooks.",
        "questions": [("¿Por qué se separan entrenamiento, validación y prueba?", "Para estimar la generalización sin usar la prueba durante el ajuste ni durante la selección de hiperparámetros."), ("¿Qué diferencia esencial hay entre clasificación y clustering?", "La clasificación usa etiquetas objetivo; el clustering busca estructura en datos sin etiquetas.")],
    },
    {
        "number": 2,
        "title": "Ensembles y descomposición multiclase",
        "markdown": "Apuntes del Tema 2 - Ensemble Models & Multiclass Decomposition.md",
        "slides": [],
        "notebooks": ["code/Practice_01/01_04_Ensembles.ipynb"],
        "bridge": "El notebook de ensembles conecta la combinación de estimadores con la evaluación empírica y permite contrastar estrategias OVA y OVO cuando aparecen varias clases.",
        "questions": [("¿Qué aporta bagging?", "Reduce la varianza al entrenar modelos sobre muestras bootstrap y combinar sus predicciones."), ("¿Cuántos clasificadores binarios requiere OVO para K clases?", r"$K(K-1)/2$, uno por cada pareja de clases.")],
    },
    {
        "number": 3,
        "title": "Regresión lineal",
        "markdown": "Apuntes del Tema 3 y Tema 4 - Modelos Lineales — Regresión Lineal y Regresión Logística.md",
        "slides": ["03_Linear_regression.pdf"],
        "notebooks": ["code/03_01_Linear_regression/03_01_Linear_regression.ipynb"],
        "bridge": "La práctica recorre solución analítica, bucle de entrenamiento, convergencia, separación train/validation, superficie de pérdida y trayectoria del descenso de gradiente.",
        "questions": [("¿Qué indica una curva de pérdida que deja de descender?", "Que el paso puede ser demasiado grande, el entrenamiento puede haber convergido o existe un problema de escala/implementación."), ("¿Qué diferencia hay entre la solución analítica y SGD?", "La primera resuelve una ecuación global bajo supuestos adecuados; SGD actualiza parámetros mediante gradientes de lotes o ejemplos.")],
    },
    {
        "number": 4,
        "title": "Regresión logística",
        "markdown": None,
        "slides": ["04_Logistic_regression.pdf"],
        "notebooks": ["code/04_01_Logistic_regression/04_01_Logistic_regression.ipynb"],
        "bridge": "El notebook implementa la sigmoide y el entrenamiento, visualiza la frontera de decisión, añade features polinomiales y compara regularización L1/L2 mediante selección y validación cruzada.",
        "questions": [("¿Por qué la salida logística puede interpretarse como probabilidad?", "La sigmoide transforma el logit real en un valor del intervalo (0,1), que se usa como probabilidad modelada."), ("¿Qué riesgo introduce un polinomio de grado alto?", "Puede aumentar la flexibilidad y también el sobreajuste, por lo que se necesita regularización y validación.")],
    },
    {
        "number": 5,
        "title": "Perceptrones multicapa",
        "markdown": "Apuntes del Tema 5 - Multilayer Perceptron (MLP I y II).md",
        "slides": ["05_01_MLP.pdf", "05_03_MLP.pdf", "05_NotesPytorch1.pdf"],
        "notebooks": ["code/05_01_MLP/05_MLP_1.ipynb", "code/05_02_MLP/05_02_MLP.ipynb", "code/05_04_MLP/05_04_MLP.ipynb", "code/Practice_02/02_01_BasicPytorch.ipynb", "code/Practice_02/02_02_Autograd.ipynb"],
        "bridge": "Las prácticas muestran el paso de regresión logística multiclase a MLP y relacionan nn.Linear, nn.Sequential y autograd con el forward/backward pass.",
        "questions": [("¿Qué representa un logit en clasificación multiclase?", "La puntuación sin normalizar de una clase antes de aplicar la transformación usada por la pérdida o la probabilidad."), ("¿Qué calcula autograd?", "Construye y recorre el grafo de operaciones para obtener derivadas respecto de los tensores que requieren gradiente.")],
    },
    {
        "number": 6,
        "title": "Regularización en redes profundas",
        "markdown": "Apuntes del Tema 6 - Regularización en Redes Profundas.md",
        "slides": ["06_01.pdf"],
        "notebooks": ["code/06_02/06_Regularization_01.ipynb"],
        "bridge": "La práctica permite relacionar las curvas de entrenamiento/validación con weight decay, dropout y early stopping. La interpretación se limita a las métricas y figuras guardadas en el notebook.",
        "questions": [("¿Qué síntoma sugiere sobreajuste?", "La pérdida de entrenamiento sigue mejorando mientras la validación empeora o deja de mejorar."), ("¿Qué hace early stopping?", "Detiene el entrenamiento cuando la métrica de validación no mejora según un criterio y puede restaurar el mejor estado.")],
    },
    {
        "number": 7,
        "title": "Descenso de gradiente y optimizadores",
        "markdown": "Apuntes del Tema 8 - Descenso de Gradiente, Métodos Acelerados y Adaptativos.md",
        "slides": ["08_01_Optimizers.pdf", "08_02_Optimizers2.pdf"],
        "notebooks": ["code/08_01_GradientDescent/08_01_GradientDescent_Lab.ipynb", "code/08_01_GradientDescent/08_02_Optimizers_Lab.ipynb", "code/08_01_GradientDescent/08_03_ExploringTorchOptim.ipynb", "code/Practice_02/02_03_AutogradExercise.ipynb", "code/Practice_02/02_04_Autograd_Lab.ipynb"],
        "bridge": "Las prácticas comparan optimizadores y schedulers en experimentos con datos y pérdidas observables. Las curvas incluidas en las salidas permiten discutir velocidad, estabilidad y sensibilidad al learning rate.",
        "questions": [("¿Qué problema intenta aliviar momentum?", "La oscilación y el avance lento en direcciones con curvaturas o escalas muy diferentes."), ("¿Por qué un scheduler modifica el learning rate?", "Para adaptar el tamaño de las actualizaciones durante el entrenamiento y favorecer convergencia o refinamiento final.")],
    },
    {
        "number": 8,
        "title": "Redes neuronales convolucionales",
        "markdown": "Apuntes del Tema 9 - Redes Neuronales Convolucionales (CNN I y II).md",
        "slides": ["09_01_CNN.pdf", "09_03_CNN.pdf", "09_05_CNN.pdf"],
        "notebooks": ["code/09_02_CNN/09_02_CNN.ipynb", "code/09_02_CNN/09_04_CNN.ipynb", "code/09_02_CNN/09_06_CNN.ipynb"],
        "bridge": "Las prácticas implementan convolución desde cero, kernels aprendidos, nn.Conv2d, pooling, LeNet y clasificación de imágenes; las figuras guardadas muestran las representaciones y métricas producidas.",
        "questions": [("¿Qué controlan padding y stride?", "Padding controla el borde y el tamaño espacial disponible; stride controla cuánto se desplaza el kernel y, por tanto, la reducción espacial."), ("¿Por qué se comparten pesos en una CNN?", "Para detectar el mismo patrón en distintas posiciones con muchos menos parámetros que una capa totalmente conectada.")],
    },
    {
        "number": 9,
        "title": "Datos secuenciales y redes recurrentes",
        "markdown": "Apuntes del Tema 10 - Redes Neuronales Recurrentes.md",
        "slides": ["10_01_SequentialData.pdf", "10_02_RNN.pdf"],
        "notebooks": ["code/10_01_SequentialData/10_01_SequentialData.ipynb", "code/10_02_RNN/10_02_RNNfromScratch.ipynb"],
        "bridge": "Los notebooks pasan del modelado de secuencias con MLP a una RNN implementada desde cero para predicción del siguiente carácter y exponen la necesidad de controlar el gradiente.",
        "questions": [("¿Qué información conserva el estado oculto?", "Una representación del contexto procesado hasta el paso temporal actual."), ("¿Qué es gradient clipping?", "Una limitación de la norma del gradiente para evitar actualizaciones excesivas, especialmente útil frente a explosión del gradiente.")],
    },
    {
        "number": 10,
        "title": "Atención y Transformers",
        "markdown": "Apuntes del Tema 11 - Attention Mechanisms and Transformers.md",
        "slides": ["11_Attention_mechanisms_and_transformers.pdf"],
        "notebooks": ["code/11_01_Attention/11_01_Attention.ipynb", "code/11_02_Attention/11_02_Attention.ipynb", "code/11_03_Attention/11_03_Attention.ipynb", "code/11_04_Submission_task/11_04_Submission_task.ipynb"],
        "bridge": "Las prácticas implementan atención, multi-head attention, codificaciones posicionales, AddNorm, FFN y un encoder Transformer; los tensores de prueba y las máscaras muestran el flujo de dimensiones.",
        "questions": [("¿Por qué se escala el producto QK^T?", "Para controlar su magnitud antes de softmax y evitar saturación cuando crece la dimensión de las claves."), ("¿Qué aporta la codificación posicional?", "Añade información sobre el orden, que no estaría determinada por la atención sobre un conjunto de tokens sin posiciones.")],
    },
    {
        "number": 11,
        "title": "Sistemas de recomendación",
        "markdown": None,
        "slides": ["13_Recommender_systems.pdf"],
        "notebooks": ["code/13_01_Recommender_systems/13_01_Recommender_systems.ipynb"],
        "bridge": "Este tema no tiene una nota Markdown equivalente en el repositorio; se incorpora como capítulo documentado desde sus diapositivas y notebook, manteniendo la trazabilidad de la fuente.",
        "questions": [("¿Qué diferencia hay entre filtrado colaborativo y basado en contenido?", "El colaborativo usa patrones de interacción entre usuarios e ítems; el basado en contenido usa atributos de los ítems y preferencias observadas."), ("¿Por qué una métrica de recomendación debe definirse con cuidado?", "Porque precisión, ranking, cobertura y diversidad miden objetivos diferentes y pueden entrar en conflicto.")],
    },
]


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_evidence(path: Path) -> tuple[list[str], list[str], int, str]:
    nb = read_notebook(path)
    headings: list[str] = []
    snippets: list[str] = []
    outputs: list[str] = []
    images = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            headings.extend(line.lstrip("# ").strip() for line in "".join(cell.get("source", [])).splitlines() if line.startswith("#"))
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        data = "\n".join(source.splitlines()[:28])
        if len(data) > 80 and ("import " in data or "torch" in data or "sklearn" in data or "def " in data):
            snippets.append(data)
        for output in cell.get("outputs", []):
            if "image/png" in output.get("data", {}) or "image/jpeg" in output.get("data", {}):
                images += 1
            text = output.get("text", output.get("data", {}).get("text/plain", ""))
            if isinstance(text, list):
                text = "".join(text)
            if text:
                outputs.extend(line.strip() for line in str(text).splitlines() if line.strip())
    return headings[:5], snippets[:1], images, " | ".join(outputs[:4])


def practice_section(topic: dict) -> str:
    rows = [r"\section{Conexión con las prácticas}", latex_plain(topic["bridge"])]
    for relative in topic["notebooks"]:
        path = ROOT / relative
        headings, snippets, images, output = notebook_evidence(path)
        rows.append(rf"\subsection{{Fuente práctica: \texttt{{{latex_plain(path.name)}}}}}")
        rows.append(latex_plain(f"Ruta: {relative}"))
        if headings:
            rows.append(r"\textbf{Apartados detectados:} " + latex_plain("; ".join(headings)) + ".")
        rows.append(latex_plain(f"Salidas gráficas almacenadas: {images}."))
        if output:
            rows.append(r"\textbf{Salida textual conservada:} \begin{quote}" + latex_plain(output) + r"\end{quote}")
        if snippets:
            rows.append(r"\textbf{Fragmento representativo:}")
            rows.append(r"\begin{lstlisting}[language=Python]")
            rows.append(snippets[0])
            rows.append(r"\end{lstlisting}")
    return "\n\n".join(rows)


def source_section(topic: dict) -> str:
    rows = [r"\section{Fuentes y trazabilidad}"]
    if topic["slides"]:
        rows.append(r"\textbf{Diapositivas:}")
        rows.append(r"\begin{itemize}")
        rows.extend(rf"\item \path{{slides/{latex_plain(slide)}}}" for slide in topic["slides"])
        rows.append(r"\end{itemize}")
    if topic["markdown"]:
        rows.append(r"\textbf{Nota personal:} " + latex_plain(f"apuntes/{topic['markdown']}"))
    return "\n\n".join(rows)


def questions_section(topic: dict) -> str:
    rows = [r"\section{Autoevaluación razonada}"]
    for question, answer in topic["questions"]:
        rows.append(rf"\subsection*{{Pregunta}} {latex_plain(question)}")
        rows.append(r"\textbf{Solución.} " + latex_plain(answer))
    return "\n\n".join(rows)


def make_chapter(topic: dict) -> str:
    if topic["markdown"]:
        source = ROOT / "apuntes" / topic["markdown"]
        body = render_markdown(source.read_text(encoding="utf-8-sig").splitlines())
    else:
        body = r"\textit{Este capítulo se construye a partir de las fuentes de diapositivas y notebook indicadas al final.}"
    return "\n".join([
        rf"\chapter{{{latex_plain(topic['title'])}}}",
        body,
        practice_section(topic),
        questions_section(topic),
        source_section(topic),
    ])


def main() -> None:
    chapter_dir = HERE / "chapters"
    chapter_dir.mkdir(exist_ok=True)
    for old in chapter_dir.glob("chapter_*.tex"):
        old.unlink()
    includes = []
    for topic in TOPICS:
        chapter = chapter_dir / f"chapter_{topic['number']:02d}.tex"
        chapter.write_text(make_chapter(topic), encoding="utf-8", newline="\n")
        includes.append(rf"\input{{chapters/{chapter.stem}}}")
    main_tex = HERE / "main.tex"
    main_tex.write_text("\n".join([
        r"\documentclass[11pt,oneside]{book}",
        r"\input{template.tex}",
        r"\begin{document}",
        r"\coverpage",
        r"\tableofcontents",
        r"\clearpage",
        *includes,
        r"\end{document}",
    ]) + "\n", encoding="utf-8", newline="\n")
    xelatex = shutil.which("xelatex")
    if not xelatex:
        raise RuntimeError("No se encontró xelatex.")
    for _ in range(2):
        subprocess.run([xelatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=HERE, check=True)
    for pattern in ("*.aux", "*.log", "*.toc", "*.out", "*.synctex.gz"):
        for auxiliary in HERE.glob(pattern):
            auxiliary.unlink()
    print(f"Generado: {HERE / 'main.pdf'}")


if __name__ == "__main__":
    main()
