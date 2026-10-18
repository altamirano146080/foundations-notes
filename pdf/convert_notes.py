"""Convert the repository's Markdown-like notes to XeLaTeX documents.

This intentionally uses only the Python standard library so that regeneration
does not require Pandoc or any additional package.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent
NOTES = ROOT
NOTES_DIR = ROOT / "apuntes"
COURSE_TITLE = "Foundations of Machine Learning and Deep Learning"


def latex_text(value: str) -> str:
    """Escape text while retaining Markdown inline constructs."""
    placeholders: list[str] = []

    def hold(text: str) -> str:
        placeholders.append(text)
        return f"\x00{len(placeholders) - 1}\x00"

    value = html.unescape(value)
    value = re.sub(r"\$\$([\s\S]+?)\$\$", lambda m: hold(r"\[" + m.group(1) + r"\]"), value)
    value = re.sub(r"(?<!\$)\$([^$\n]+)\$(?!\$)", lambda m: hold(
        r"\(" + m.group(1) + r"\)"
    ), value)
    value = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: hold(
        rf"\includegraphics[width=\linewidth,keepaspectratio]{{{m.group(2)}}}"
        + (rf"\captionof{{figure}}{{{latex_plain(m.group(1))}}}" if m.group(1) else "")
    ), value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: hold(
        rf"\href{{{m.group(2)}}}{{{latex_plain(m.group(1))}}}"
    ), value)
    value = re.sub(r"`([^`]+)`", lambda m: hold(
        rf"\texttt{{{latex_plain(m.group(1))}}}"
    ), value)
    value = re.sub(r"\*\*([^*]+)\*\*", lambda m: hold(
        rf"\textbf{{{latex_plain(m.group(1))}}}"
    ), value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: hold(
        rf"\emph{{{latex_plain(m.group(1))}}}"
    ), value)
    value = value.replace("\\", r"\textbackslash{}")
    for char, replacement in {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", 
    }.items():
        value = value.replace(char, replacement)
    token = re.compile("\x00(\\d+)\x00")

    def resolve(match: re.Match[str]) -> str:
        return token.sub(resolve, placeholders[int(match.group(1))])

    return token.sub(resolve, value)


def latex_plain(value: str) -> str:
    value = html.unescape(value)
    value = value.replace("\\", r"\textbackslash{}")
    for char, replacement in {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        value = value.replace(char, replacement)
    return value


def table_row(line: str) -> list[str]:
    parts = line.strip().strip("|").split("|")
    return [part.strip() for part in parts]


def is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*", line))


def render_table(lines: list[str]) -> str:
    rows = [table_row(line) for line in lines]
    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]
    body = rows[2:] if len(rows) > 1 and is_table_separator(lines[1]) else rows[1:]
    column_spec = " ".join([r">{\raggedright\arraybackslash}X"] * columns)
    output = [
        r"\begin{center}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(latex_text(cell) for cell in rows[0]) + r" \\",
        r"\midrule",
    ]
    output.extend(" & ".join(latex_text(cell) for cell in row) + r" \\" for row in body)
    output.extend([r"\bottomrule", r"\end{tabularx}", r"\end{center}"])
    return "\n".join(output)


def language_for(fence: str) -> str:
    language = fence.strip().lower()
    return {"python": "Python", "py": "Python", "bash": "bash",
            "sh": "bash", "shell": "bash", "json": "json",
            "yaml": "yaml", "yml": "yaml", "latex": "TeX",
            "tex": "TeX", "sql": "SQL", "cpp": "C++",
            "javascript": "Java", "js": "Java"}.get(language, "")


def render_markdown(lines: list[str]) -> str:
    output: list[str] = []
    index = 0
    in_code = False
    code_language = ""
    code_lines: list[str] = []
    list_kind: str | None = None

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            output.append(rf"\end{{{list_kind}}}")
            list_kind = None

    while index < len(lines):
        line = lines[index].rstrip("\r")
        fence = re.match(r"^\s*```(.*)$", line)
        if fence:
            if in_code:
                language = language_for(code_language)
                option = rf"[language={language}]" if language else ""
                output.append(rf"\begin{{lstlisting}}{option}")
                output.extend(code_lines)
                output.append(r"\end{lstlisting}")
                in_code, code_lines = False, []
            else:
                close_list()
                in_code, code_language = True, fence.group(1)
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue
        if not line.strip():
            close_list()
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            close_list()
            level = min(len(heading.group(1)), 3)
            command = ["section", "subsection", "subsubsection"][level - 1]
            output.append(rf"\{command}{{{latex_text(heading.group(2).strip())}}}")
            index += 1
            continue
        if line.strip() == "---":
            index += 1
            continue
        if line.lstrip().startswith("|") and index + 1 < len(lines) and is_table_separator(lines[index + 1]):
            close_list()
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            output.append(render_table(table_lines))
            continue
        item = re.match(r"^\s*([-*+])\s+(.*)$", line)
        numbered = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        if item or numbered:
            desired = "itemize" if item else "enumerate"
            if list_kind != desired:
                close_list()
                list_kind = desired
                output.append(rf"\begin{{{list_kind}}}")
            text = item.group(2) if item else numbered.group(1)
            output.append(rf"\item {latex_text(text)}")
            index += 1
            continue
        close_list()
        if line.startswith("    ") or line.startswith("\t"):
            output.append(rf"\texttt{{{latex_plain(line.lstrip())}}}\\")
        elif line.startswith(">"):
            output.append(rf"\begin{{quote}}{latex_text(line[1:].strip())}\end{{quote}}")
        elif line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 4:
            output.append(r"\[" + line.strip()[2:-2].strip() + r"\]")
        else:
            output.append(latex_text(line))
        index += 1
    if in_code:
        language = language_for(code_language)
        option = rf"[language={language}]" if language else ""
        output.append(rf"\begin{{lstlisting}}{option}")
        output.extend(code_lines)
        output.append(r"\end{lstlisting}")
    close_list()
    return "\n\n".join(output)


def topic_files() -> list[Path]:
    files = [path for path in NOTES_DIR.iterdir() if path.is_file()]
    return sorted(files, key=lambda path: int(re.search(r"Tema (\d+)", path.name).group(1)))


def output_stem(path: Path) -> str:
    match = re.search(r"Tema (\d+)", path.name)
    if "Tema 3 y Tema 4" in path.name:
        return "temas_03_04"
    return f"tema_{int(match.group(1)):02d}"


def convert(path: Path) -> Path:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    title = next((re.sub(r"^#\s+", "", line).strip() for line in lines if line.startswith("# ")), path.stem)
    content = render_markdown(lines)
    destination = OUT / f"{output_stem(path)}.tex"
    document = rf"""\documentclass[11pt,oneside]{{article}}
\newcommand{{\TopicTitle}}{{{latex_plain(title)}}}
\input{{template.tex}}
\begin{{document}}
\topiccover{{{latex_plain(title)}}}
\tableofcontents
\clearpage
\markboth{{{latex_plain(title)}}}{{{latex_plain(title)}}}
{content}
\end{{document}}
"""
    destination.write_text(document, encoding="utf-8", newline="\n")
    return destination


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for pattern in ("tema_*.tex", "temas_*.tex", "tema_*.pdf", "temas_*.pdf",
                    "tema_*.aux", "temas_*.aux", "tema_*.log", "temas_*.log",
                    "tema_*.toc", "temas_*.toc", "tema_*.out", "temas_*.out",
                    "tema_*.synctex.gz", "temas_*.synctex.gz"):
        for old in OUT.glob(pattern):
            old.unlink()
    generated = [convert(path) for path in topic_files()]
    xelatex = shutil.which("xelatex")
    if not xelatex:
        print("Generated LaTeX; xelatex was not found, so compilation was skipped.")
        return
    for document in generated:
        for _ in range(2):
            subprocess.run(
                [xelatex, "-interaction=nonstopmode", "-halt-on-error", document.name],
                cwd=OUT, check=True,
            )
    for pattern in ("tema_*.aux", "temas_*.aux", "tema_*.log", "temas_*.log",
                    "tema_*.toc", "temas_*.toc", "tema_*.out", "temas_*.out",
                    "tema_*.synctex.gz", "temas_*.synctex.gz"):
        for auxiliary in OUT.glob(pattern):
            auxiliary.unlink()


if __name__ == "__main__":
    main()
