(() => {
  const topics = window.STUDY_CONTENT;
  const app = document.querySelector("#app");
  const nav = document.querySelector("#topic-nav");
  const key = "ml-studio-progress";
  const getProgress = () => JSON.parse(localStorage.getItem(key) || "{}");
  const saveProgress = p => { localStorage.setItem(key, JSON.stringify(p)); updateProgress(); };
  const updateProgress = () => {
    const p = getProgress(); const done = topics.filter(t => p[t.id]).length;
    const percent = Math.round(done / topics.length * 100);
    document.querySelector("#global-progress").textContent = `${percent}%`;
    document.querySelector("#global-progress-bar").style.width = `${percent}%`;
    document.querySelectorAll("[data-card-progress]").forEach(el => { const value = p[el.dataset.cardProgress] ? 100 : 0; el.querySelector("span").style.width = `${value}%`; el.querySelector("b").textContent = `${value}%`; });
  };
  const link = (href, label, cls="") => `<a class="${cls}" href="${href}">${label}</a>`;
  const sourceLinks = t => `<div class="source-list">
    <a class="source" target="_blank" rel="noopener" href="../${t.slides}"><b>Slides del profesor</b>${t.slides}</a>
    <a class="source" target="_blank" rel="noopener" href="../${t.notes}"><b>Apuntes generados</b>${t.notes}</a>
    <a class="source" target="_blank" rel="noopener" href="../${t.code}"><b>Notebook original</b>${t.code}</a>
  </div>`;
  const renderNav = () => { nav.innerHTML = topics.map(t => link(`#/topic/${t.id}`, `<span class="topic-number">${t.order}</span>${t.title}`, "topic-link")).join(""); };
  const markActive = id => document.querySelectorAll(".topic-link").forEach(a => a.classList.toggle("active", a.href.endsWith(id)));
  const home = () => {
    const p = getProgress(); const done = topics.filter(t => p[t.id]).length;
    app.innerHTML = `<section class="hero"><div><div class="kicker">Foundations of machine learning</div><h1>Aprende conectando<br>teoría y práctica.</h1><p>Un mapa de estudio construido a partir de tus slides, apuntes y notebooks reales. Avanza desde los fundamentos hasta atención y recomendadores, siempre con trazabilidad.</p></div><div class="hero-stat"><strong>${done}/${topics.length}</strong><span>temas estudiados</span></div></section>
      <div class="callout"><strong>Ruta recomendada:</strong> completa una lección, intenta el ejercicio sin mirar la solución y vuelve a los conceptos marcados como difíciles. El progreso se guarda solo en este navegador.</div>
      <div class="section-heading"><h2>Mapa de conocimiento</h2><span class="muted">${topics.length} unidades · 29 notebooks relacionados</span></div>
      <section class="topic-grid">${topics.map(t => `<a class="topic-card" href="#/topic/${t.id}"><header><span class="kicker">${t.order}</span><span class="tag">${t.eyebrow}</span></header><h3>${t.title}</h3><p>${t.summary}</p><div class="card-progress" data-card-progress="${t.id}"><div class="progress-track"><span></span></div><b>0%</b></div></a>`).join("")}</section>`;
    updateProgress();
  };
  const topicView = id => {
    const t = topics.find(x => x.id === id); if (!t) return home();
    markActive(id);
    const l = t.lesson;
    app.innerHTML = `<article><header class="topic-header"><div class="kicker">${t.order} · ${t.eyebrow}</div><h1>${t.title}</h1><p>${t.summary}</p><div class="topic-meta"><span class="meta-pill">Prerequisito: ${t.prerequisites}</span><span class="meta-pill">${t.concepts.length} conceptos clave</span></div></header>
      <div class="lesson-grid"><div><section class="panel"><h2>Lección guiada</h2><h3>La intuición</h3><p>${l.intuition}</p><h3>Explicación conceptual</h3><p>${l.body}</p><h3>Cómo funciona</h3><ol class="mechanism">${l.mechanism.map(x=>`<li>${x}</li>`).join("")}</ol><h3>Conexión con tu material</h3><div class="callout">${l.example}</div><h3>Errores frecuentes</h3><div class="pitfalls"><ul>${l.pitfalls.map(x=>`<li>${x}</li>`).join("")}</ul></div><h3>Ideas que debes poder explicar</h3><div class="concept-list">${t.concepts.map(x=>`<span class="concept">${x}</span>`).join("")}</div></section>
      <section class="panel quiz-card"><h2>Comprueba tu comprensión</h2>${t.quiz.map((q,i)=>quizMarkup(q,i,t.id)).join("")}</section></div>
      <aside><section class="panel"><h2>Fuentes trazables</h2>${sourceLinks(t)}<div class="callout">Las explicaciones en lenguaje natural son una elaboración pedagógica basada en estas fuentes. Consulta el original para fórmulas, diagramas y contexto completo.</div></section>
      <section class="panel lab"><h2>Laboratorio</h2><p>La práctica original está disponible para abrirla en Jupyter. La visualización rápida de abajo es una simulación didáctica independiente.</p>${t.id==="linear" ? gradientLab() : `<a class="action secondary" href="../${t.code}" target="_blank" rel="noopener">Abrir notebook original</a>`}</section>
      <button class="action" id="complete-topic" type="button">${getProgress()[t.id] ? "Tema completado ✓" : "Marcar tema como estudiado"}</button></aside></div></article>`;
    document.querySelector("#complete-topic").addEventListener("click", () => { const p=getProgress(); p[t.id]=true; saveProgress(p); document.querySelector("#complete-topic").textContent="Tema completado ✓"; });
    t.quiz.forEach((q,i)=>document.querySelectorAll(`[data-quiz="${t.id}-${i}"]`).forEach(btn=>btn.addEventListener("click",()=>answerQuiz(btn,q,t.id,i))));
    if(t.id==="linear") initGradient();
  };
  const quizMarkup = (q,i,id) => `<div class="quiz-question"><strong>${i+1}. ${q.q}</strong>${q.options.map((o,n)=>`<button class="quiz-option" data-quiz="${id}-${i}" data-answer="${n}" type="button">${o}</button>`).join("")}<div class="quiz-feedback" id="feedback-${id}-${i}" hidden></div></div>`;
  const answerQuiz = (btn,q,id,i) => { const buttons=document.querySelectorAll(`[data-quiz="${id}-${i}"]`); buttons.forEach(b=>b.disabled=true); const correct=Number(btn.dataset.answer)===q.answer; btn.classList.add(correct?"correct":"wrong"); if(!correct) buttons[q.answer].classList.add("correct"); const f=document.querySelector(`#feedback-${id}-${i}`); f.hidden=false; f.innerHTML=`<strong>${correct?"Correcto":"Aún no"}.</strong> ${q.why}`; };
  const gradientLab = () => `<div class="slider-row"><label for="learning-rate">Learning rate</label><input id="learning-rate" type="range" min=".01" max=".5" step=".01" value=".12"><output id="lr-value">.12</output></div><canvas class="viz" id="gradient-canvas" width="600" height="250" aria-label="Visualización de descenso de gradiente"></canvas><p class="muted" id="gradient-caption">Ajusta el learning rate y observa el recorrido hacia el mínimo de una función cuadrática.</p>`;
  const initGradient = () => { const canvas=document.querySelector("#gradient-canvas"),ctx=canvas.getContext("2d"),range=document.querySelector("#learning-rate"),out=document.querySelector("#lr-value"); const draw=()=>{const lr=Number(range.value);out.value=lr.toFixed(2);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--line");ctx.lineWidth=1;for(let x=0;x<canvas.width;x+=30){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke()}for(let y=0;y<canvas.height;y+=30){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke()}ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--accent");ctx.lineWidth=3;ctx.beginPath();for(let x=0;x<canvas.width;x++){const y=35+Math.pow((x-300)/90,2)*.13;if(x===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}ctx.stroke();let x=70;ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--accent-2");ctx.beginPath();ctx.arc(x,35+Math.pow((x-300)/90,2)*.13,7,0,Math.PI*2);ctx.fill();for(let i=0;i<12;i++){const grad=2*(x-300)/90/90;x-=lr*grad*70;x=Math.max(10,Math.min(590,x));ctx.beginPath();ctx.arc(x,35+Math.pow((x-300)/90,2)*.13,4,0,Math.PI*2);ctx.fill()}};range.addEventListener("input",draw);draw();};
  const search = term => { const q=term.trim().toLowerCase(); if(!q) return home(); markActive(""); const results=topics.filter(t=>[t.title,t.summary,...t.concepts,t.slides,t.code].join(" ").toLowerCase().includes(q)); app.innerHTML=`<div class="kicker">Búsqueda</div><h1>Resultados para “${term.replace(/</g,"&lt;")}”</h1><p class="muted">${results.length} coincidencias en el mapa de conocimiento.</p><div class="search-results">${results.map(t=>`<a class="result" href="#/topic/${t.id}"><small>${t.order} · ${t.eyebrow}</small><br><strong>${t.title}</strong><br><span class="muted">${t.summary}</span></a>`).join("")||`<div class="empty">No hay coincidencias. Prueba con “gradiente”, “CNN”, “atención” o el nombre de un notebook.</div>`}</div>`; };
  const route = () => { const hash=location.hash.slice(1); const match=hash.match(/^\/topic\/([^/]+)/); if(match) topicView(match[1]); else home(); };
  document.querySelector("#search-input").addEventListener("input",e=>{if(e.target.value.length>1)search(e.target.value)}); document.querySelector("#theme-toggle").addEventListener("click",()=>{document.documentElement.classList.toggle("dark");localStorage.setItem("ml-studio-theme",document.documentElement.classList.contains("dark")?"dark":"light")}); document.querySelector("#reset-progress").addEventListener("click",()=>{if(confirm("¿Reiniciar el progreso de todos los temas?")){localStorage.removeItem(key);route()}}); if(localStorage.getItem("ml-studio-theme")==="dark")document.documentElement.classList.add("dark"); window.addEventListener("hashchange",route); renderNav(); route();
})();
