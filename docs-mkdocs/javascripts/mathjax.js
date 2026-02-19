window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  svg: {
    fontCache: "global",
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
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
