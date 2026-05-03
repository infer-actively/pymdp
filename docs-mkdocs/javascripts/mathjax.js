// Dollar-sign delimiters and `jp-Notebook` are needed for math in Jupyter
// notebooks: mkdocs-jupyter renders cells via nbconvert, which leaves raw
// `$...$` / `$$...$$` / `\begin{...}` in the HTML inside `.jp-Notebook`
// rather than wrapping them in `.arithmatex` spans the way pymdownx does.
// `ignoreHtmlClass` stays at the default so MathJax can descend through the
// generic divs nested between `.jp-Notebook` and the actual math text.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  svg: {
    fontCache: "global",
  },
  options: {
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "arithmatex|jp-Notebook",
  },
};

const renderMath = () => {
  if (!window.MathJax || !window.MathJax.typesetPromise) return;
  window.MathJax.typesetClear();
  window.MathJax.typesetPromise();
};

if (typeof document$ !== "undefined") {
  document$.subscribe(renderMath);
}

window.addEventListener("load", renderMath);
