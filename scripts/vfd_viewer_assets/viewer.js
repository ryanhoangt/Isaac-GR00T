export default function ({ parentElement, data }) {
  const root = document.createElement('section');
  root.className = 'vfd';
  root.dataset.episode = `${data.episode.suite}/${data.episode.task_id}/${data.episode.initial_state_index}`;
  // Only static, trusted markup goes into innerHTML. Dataset text uses textContent.
  root.innerHTML = `
    <div class="hint episode-label"></div><div class="hint instruction"></div>
    <div class="toolbar">
      <button data-action="play">Play</button><button data-action="prev">← Frame</button><button data-action="next">Frame →</button>
      <label>Speed <select class="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>×</label>
      <span class="step-label"></span><input class="timeline" aria-label="Timestep" type="range" min="0" value="0">
      <input class="step-number" aria-label="Go to timestep" type="number" min="0" value="0" style="width:85px">
    </div>
    <div class="toolbar">
      <label><input class="show-source" type="checkbox" checked> <span class="source">Source threshold</span></label>
      <label><input class="show-target" type="checkbox" checked> <span class="target">Target threshold</span></label>
      <label><input class="smooth" type="checkbox"> Trailing mean (5 steps; display only)</label>
      <button data-action="zoom">Zoom ±20 steps</button><button data-action="reset">Reset zoom</button>
    </div>
    <div class="panels"></div>
    <div class="hint">Click/drag either chart to seek both views. Focus here, then use ←/→ for frames and Space to play/pause.
    Video time is recorded playback time, not inference wall time. Frame t is the observation used to generate action t; its consequence usually appears in frame t+1.</div>
    <div class="annotations"><strong>Mark a critical event</strong>
      <div class="toolbar"><input class="note" aria-label="Event note" type="text" placeholder="e.g. grasp, contact, slip, release">
      <button data-action="mark">Mark current step</button><button data-action="export">Download event notes</button><button data-action="clear">Clear notes</button></div>
      <div class="hint">Notes stay in this open view. Download before switching episode or reloading. These are manual observations, not inferred failure onset.</div>
      <div class="notes-list"></div>
    </div>`;
  parentElement.appendChild(root);
  root.tabIndex = 0;
  const $ = q => root.querySelector(q);
  const maxStep = Math.max(...data.panels.map(p => p.scores.length)) - 1;
  let step = 0, playing = false, frameId = 0, startClock = 0, startStep = 0;
  let windowStart = 0, windowEnd = maxStep, dragging = false;
  const notes = [], panels = [];
  $('.timeline').max = maxStep;
  $('.step-number').max = maxStep;
  $('.instruction').textContent = data.panels[0].metadata.language;
  $('.episode-label').textContent = `${data.episode.suite} · task ${data.episode.task_id} · initial state ${data.episode.initial_state_index}`;
  if (data.panels.length > 1) {
    const notice = document.createElement('div');
    notice.className = 'notice';
    notice.textContent = data.same_actions === true ? 'Executed actions match exactly across these two recordings.' :
      'These recordings share task/initial state. Their executed actions differ or are unavailable; compare trajectories carefully.';
    root.insertBefore(notice, $('.toolbar'));
  }
  const fmt = n => Number(n).toFixed(3);
  const first = (p, kind) => p.alarm_steps[kind].length ? p.alarm_steps[kind][0] : null;
  const download = (name, value) => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  function setStep(value) {
    step = Math.max(0, Math.min(maxStep, Math.round(Number(value) || 0)));
    $('.timeline').value = step; $('.step-number').value = step;
    $('.step-label').textContent = `Step ${step} / ${maxStep}`;
    for (const { p, video, image, caption, reading, action, canvas } of panels) {
      const index = Math.min(step, p.scores.length - 1), ended = step >= p.scores.length;
      if (video) {
        // Pause-and-seek gives a deterministic discrete frame for every score.
        const time = p.frame_times[index] + 0.00001;
        if (video.readyState >= 1 && Math.abs(video.currentTime - time) > 0.0001) video.currentTime = time;
        caption.textContent = `Frame ${index} · MP4 ${p.frame_times[index].toFixed(3)} s${ended ? ' · episode already ended' : ''}`;
      } else if (image && p.snapshots.length) {
        const snapshot = p.snapshots.reduce((a, b) => Math.abs(b.step-index) < Math.abs(a.step-index) ? b : a);
        image.src = snapshot.src;
        caption.textContent = `Nearest saved snapshot: step ${snapshot.step} (selected ${step}; not an exact frame)`;
      }
      const flags = ['source', 'target'].filter(k => p.thresholds[k] !== null && p.scores[index] > p.thresholds[k]);
      const calibrated = Object.values(p.thresholds).some(value => value !== null);
      reading.textContent = ended ? 'Episode ended; holding final score/frame.' : `U = ${fmt(p.scores[index])} · ${!calibrated ? 'no calibration selected' : flags.length ? 'above '+flags.join(' + ')+' threshold' : 'below available thresholds'}`;
      reading.className = 'reading' + (flags.length && !ended ? ' alarm' : '');
      action.textContent = p.actions && p.actions[index] ? `Executed action: [${p.actions[index].map(x => Number(x).toFixed(3)).join(', ')}]` : '';
      draw(canvas, p);
    }
  }
  function draw(canvas, p) {
    const box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(box.width * ratio)); canvas.height = Math.round(265 * ratio);
    const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
    const w = box.width, h = 265, pad = { l: 47, r: 12, t: 18, b: 32 };
    const lo = windowStart, hi = Math.max(lo + 1, windowEnd);
    const visible = p.scores.slice(lo, Math.min(hi+1, p.scores.length));
    const thresholds = ['source','target'].filter(k => $('.show-'+k).checked && p.thresholds[k] !== null);
    const ymax = Math.max(1e-9, ...visible, ...thresholds.map(k => p.thresholds[k])) * 1.1;
    const X = x => pad.l+(x-lo)/(hi-lo)*(w-pad.l-pad.r), Y = y => h-pad.b-y/ymax*(h-pad.b-pad.t);
    ctx.font = '11px system-ui';
    for(let i=0;i<=4;i++) {
      const y=ymax*i/4;ctx.strokeStyle='#344258';ctx.beginPath();ctx.moveTo(pad.l,Y(y));ctx.lineTo(w-pad.r,Y(y));ctx.stroke();
      ctx.fillStyle='#aebed5';ctx.textAlign='right';ctx.fillText(y.toFixed(ymax<10 ? 2 : 0),pad.l-6,Y(y)+4);
      const x=Math.round(lo+(hi-lo)*i/4);ctx.textAlign='center';ctx.fillText(String(x),X(x),h-10);
    }
    ctx.save();ctx.beginPath();ctx.rect(pad.l,pad.t,w-pad.l-pad.r,h-pad.b-pad.t);ctx.clip();
    function line(values,color,width) {
      ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash([]);ctx.beginPath();
      for(let i=lo;i<=hi && i<values.length;i++) { if(i===lo)ctx.moveTo(X(i),Y(values[i]));else ctx.lineTo(X(i),Y(values[i])); }
      ctx.stroke();
    }
    line(p.scores,'#d7e1f2',1.4);
    if($('.smooth').checked) line(p.scores.map((_,i) => p.scores.slice(Math.max(0,i-4),i+1).reduce((a,b)=>a+b,0)/Math.min(i+1,5)),'#c4a6ff',2);
    for(const kind of thresholds) {
      ctx.strokeStyle=kind==='source'?'#76b9ff':'#ffc36f';ctx.lineWidth=1.5;ctx.setLineDash([7,5]);
      ctx.beginPath();ctx.moveTo(pad.l,Y(p.thresholds[kind]));ctx.lineTo(w-pad.r,Y(p.thresholds[kind]));ctx.stroke();
      const alarm=first(p,kind);if(alarm!==null) {ctx.setLineDash([2,5]);ctx.beginPath();ctx.moveTo(X(alarm),pad.t);ctx.lineTo(X(alarm),h-pad.b);ctx.stroke();}
    }
    ctx.setLineDash([]);
    for(const note of notes) {ctx.fillStyle='#ff9ddc';ctx.fillRect(X(note.step)-2,pad.t,4,9);}
    ctx.strokeStyle='#ff8e9c';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(X(step),pad.t);ctx.lineTo(X(step),h-pad.b);ctx.stroke();
    if(step<p.scores.length) {ctx.fillStyle='#ff8e9c';ctx.beginPath();ctx.arc(X(step),Y(p.scores[step]),3.5,0,2*Math.PI);ctx.fill();}
    ctx.restore();
  }
  function seek(value) { pause();setStep(value); }
  function pause() { playing=false;cancelAnimationFrame(frameId);$('[data-action=play]').textContent='Play'; }
  function tick(now) {
    if(!playing)return;
    const fps=data.panels[0].video_fps || 20;
    const next=startStep+Math.floor((now-startClock)/1000*fps*Number($('.speed').value));
    if(next!==step)setStep(next);
    if(next>=maxStep) {pause();return;}frameId=requestAnimationFrame(tick);
  }
  function play() {
    if(playing) {pause();return;}
    if(step>=maxStep)setStep(0);
    playing=true;startClock=performance.now();startStep=step;$('[data-action=play]').textContent='Pause';frameId=requestAnimationFrame(tick);
  }
  for(const p of data.panels) {
    const panel=document.createElement('article');panel.className='panel';
    const heading=document.createElement('h3');heading.textContent=p.label;panel.appendChild(heading);
    const meta=document.createElement('div');meta.className='meta';meta.textContent=`${p.metadata.success?'SUCCESS':'FAILURE'} (observed outcome) · ${p.scores.length} action steps · peak ${fmt(Math.max(...p.scores))}`;panel.appendChild(meta);
    const jumps=document.createElement('div');jumps.className='jumps';
    for(const [label,target] of [['Peak',p.peak_step],['First source alarm',first(p,'source')],['First target alarm',first(p,'target')]]) {
      const button=document.createElement('button');button.textContent=target===null?label+': none':label+': '+target;button.disabled=target===null;button.onclick=()=>seek(target);jumps.appendChild(button);
    }
    panel.appendChild(jumps);
    let video=null,image=null;
    if(p.video_src) {
      video=document.createElement('video');video.className='frame';video.src=p.video_src;video.muted=true;video.preload='auto';video.playsInline=true;
      video.addEventListener('loadedmetadata',()=>setStep(step));
      video.addEventListener('error',()=>{caption.textContent='Browser cannot decode this MP4. Try Chrome/Firefox or export the score CSV.';caption.className='warning';});
      panel.appendChild(video);
    } else if(p.snapshots.length) {image=document.createElement('img');image.className='frame';panel.appendChild(image);}
    const caption=document.createElement('div');caption.className='hint';panel.appendChild(caption);
    if(p.warning) {const warning=document.createElement('p');warning.className='warning';warning.textContent=p.warning;panel.appendChild(warning);}
    const reading=document.createElement('div');panel.appendChild(reading);
    const legend=document.createElement('div');legend.className='legend';
    for(const kind of ['source','target']) {const span=document.createElement('span');span.className=kind;span.textContent=kind+': '+(p.thresholds[kind]===null?'unavailable':fmt(p.thresholds[kind]));legend.appendChild(span);}
    panel.appendChild(legend);
    const canvas=document.createElement('canvas');canvas.setAttribute('aria-label',p.label+' uncertainty chart');canvas.setAttribute('role','img');panel.appendChild(canvas);
    const action=document.createElement('div');action.className='action';panel.appendChild(action);
    const pick=event=> {const rect=canvas.getBoundingClientRect();seek(windowStart+(event.clientX-rect.left-47)/(rect.width-59)*Math.max(1,windowEnd-windowStart));};
    canvas.addEventListener('pointerdown',event=>{dragging=true;canvas.setPointerCapture(event.pointerId);pick(event);});
    canvas.addEventListener('pointermove',event=>{if(dragging)pick(event);});
    canvas.addEventListener('pointerup',()=>dragging=false);canvas.addEventListener('pointercancel',()=>dragging=false);
    $('.panels').appendChild(panel);panels.push({p,video,image,caption,reading,action,canvas});
  }
  $('[data-action=play]').onclick=play;
  $('[data-action=prev]').onclick=()=>seek(step-1);$('[data-action=next]').onclick=()=>seek(step+1);
  $('.timeline').oninput=e=>seek(e.target.value);$('.step-number').onchange=e=>seek(e.target.value);
  $('.speed').onchange=()=>{if(playing){startClock=performance.now();startStep=step;}};
  for(const selector of ['.show-source','.show-target','.smooth']) $(selector).onchange=()=>setStep(step);
  $('[data-action=zoom]').onclick=()=>{windowStart=Math.max(0,step-20);windowEnd=Math.min(maxStep,step+20);setStep(step);};
  $('[data-action=reset]').onclick=()=>{windowStart=0;windowEnd=maxStep;setStep(step);};
  root.addEventListener('keydown',event=>{
    if(['INPUT','TEXTAREA','SELECT','BUTTON'].includes(event.target.tagName))return;
    if(event.key==='ArrowLeft'){event.preventDefault();seek(step-1);}if(event.key==='ArrowRight'){event.preventDefault();seek(step+1);}
    if(event.code==='Space'){event.preventDefault();play();}
  });
  function refreshNotes() {
    $('.notes-list').replaceChildren();
    for(const note of notes) {const button=document.createElement('button');button.textContent=`Step ${note.step}: ${note.text}`;button.onclick=()=>seek(note.step);$('.notes-list').appendChild(button);}
    setStep(step);
  }
  $('[data-action=mark]').onclick=()=>{pause();const text=$('.note').value.trim();if(text){notes.push({step,text});$('.note').value='';refreshNotes();}};
  $('[data-action=export]').onclick=()=>download(`task${data.episode.task_id}_init${data.episode.initial_state_index}_events.json`,{episode:data.episode,runs:data.panels.map(p=>p.run),notes});
  $('[data-action=clear]').onclick=()=>{notes.length=0;refreshNotes();};
  const observer=new ResizeObserver(()=>setStep(step));observer.observe(root);
  setStep(0);
  return ()=>{pause();observer.disconnect();for(const {video} of panels)if(video){video.pause();video.removeAttribute('src');video.load();}root.remove();};
}
