/**
 * Pulse orb: a WebGL sphere that beats at the measured heart rate.
 *
 * The visual carries the same contract as the API. When the engine can measure,
 * the orb beats at the real rate and its surface detail tracks confidence. When
 * the engine declines, the orb *stops* and desaturates rather than drifting on
 * a stale value. A visualisation that keeps animating through an abstention
 * would quietly undo the thing the whole confidence system exists to do.
 *
 * Raw WebGL, no dependencies: an icosphere displaced in the vertex shader by
 * value noise, modulated by a cardiac envelope.
 */

const VERT = `
attribute vec3 aPos;
uniform mat4 uProj, uView;
uniform float uTime, uBeat, uDetail, uQuiet;
varying vec3 vNormal; varying float vDisp; varying float vNoise; varying vec3 vWorld;

float hash(vec3 p){ return fract(sin(dot(p, vec3(17.1,31.7,74.3))) * 43758.5453); }
float noise(vec3 p){
  vec3 i = floor(p), f = fract(p);
  f = f*f*(3.0-2.0*f);
  float n = mix(mix(mix(hash(i+vec3(0,0,0)), hash(i+vec3(1,0,0)), f.x),
                    mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
                mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x),
                    mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y), f.z);
  return n*2.0-1.0;
}

void main(){
  vec3 n = normalize(aPos);
  // Surface detail scales with confidence; a quiet orb is a smooth one.
  float turb = noise(n*4.5 + uTime*0.22)*0.055
             + noise(n*9.0 - uTime*0.13)*0.026
             + noise(n*17.0 + uTime*0.08)*0.011;
  // Cardiac envelope: a sharp systolic rise and a slower decay, not a sine.
  float beat = uBeat;
  float disp = 1.0 + beat*0.10 + turb*uDetail*(1.0-uQuiet*0.85);
  vDisp = beat; vNoise = turb; vNormal = n; vWorld = n*disp;
  gl_Position = uProj * uView * vec4(n*disp, 1.0);
}`;

const FRAG = `
precision highp float;
varying vec3 vNormal; varying float vDisp; varying float vNoise; varying vec3 vWorld;
uniform float uQuiet, uConf;
void main(){
  vec3 L = normalize(vec3(-0.35, 0.78, 0.62));
  float lambert = max(dot(vNormal, L), 0.0);
  float facing = max(dot(vNormal, vec3(0.0,0.0,1.0)), 0.0);
  float rim = pow(1.0-facing, 2.15);
  float bands = smoothstep(.22,.78,sin((vWorld.y+vNoise*.8)*20.0)*.5+.5);
  // Measuring: an electric cyan-violet material. Declining: drained to slate.
  vec3 deep = vec3(0.10,0.16,0.48);
  vec3 cyan = vec3(0.24,0.92,1.0);
  vec3 violet = vec3(0.46,0.34,0.98);
  vec3 live = mix(deep, cyan, .42 + lambert*.34 + vDisp*.45);
  live = mix(live, violet, bands*.18 + max(-vNoise,0.0)*.45);
  vec3 dead = vec3(0.26,0.30,0.38);
  vec3 base = mix(live, dead, uQuiet);
  vec3 edge = mix(vec3(0.42,0.94,1.0), vec3(0.34), uQuiet);
  vec3 col = base*(0.34 + 0.78*lambert) + rim*edge*(.72 + uConf*.42);
  col += pow(max(dot(vNormal,normalize(vec3(-.6,.7,.5))),0.0),28.0)*vec3(.8,1.0,1.0)*.7;
  gl_FragColor = vec4(col * (0.58 + 0.42*uConf), 1.0);
}`;

function icosphere(subdiv) {
  const t = (1 + Math.sqrt(5)) / 2;
  let verts = [[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],
               [0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]].map(norm);
  let faces = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],
               [11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],
               [3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]];
  function norm(v){ const l=Math.hypot(...v); return [v[0]/l,v[1]/l,v[2]/l]; }
  for (let s = 0; s < subdiv; s++) {
    const next = [], cache = new Map();
    const mid = (a,b) => {
      const key = a<b ? a+"_"+b : b+"_"+a;
      if (cache.has(key)) return cache.get(key);
      const m = norm([(verts[a][0]+verts[b][0])/2,(verts[a][1]+verts[b][1])/2,
                      (verts[a][2]+verts[b][2])/2]);
      verts.push(m); cache.set(key, verts.length-1); return verts.length-1;
    };
    for (const [a,b,c] of faces) {
      const ab=mid(a,b), bc=mid(b,c), ca=mid(c,a);
      next.push([a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]);
    }
    faces = next;
  }
  const out = [];
  for (const f of faces) for (const i of f) out.push(...verts[i]);
  return new Float32Array(out);
}

export function createOrb(canvas) {
  const gl = canvas.getContext("webgl", {antialias:true, alpha:true});
  if (!gl) return null;
  const compile = (type, src) => { const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog); gl.useProgram(prog);

  const data = icosphere(5);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "aPos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
  const U = n => gl.getUniformLocation(prog, n);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // State the orb animates toward, so changes ease rather than jump.
  let hr = null, quiet = 1, conf = 0, detail = 1, phase = 0, last = performance.now();
  let quietS = 1, confS = 0;

  function resize(){
    const d = Math.min(devicePixelRatio, 2);
    canvas.width = canvas.clientWidth*d; canvas.height = canvas.clientHeight*d;
    gl.viewport(0,0,canvas.width,canvas.height);
  }

  function frame(now){
    const dt = Math.min((now-last)/1000, 0.1); last = now;
    quietS += (quiet-quietS)*Math.min(dt*3,1);
    confS  += (conf-confS)*Math.min(dt*3,1);

    // Advance the cardiac phase only while measuring. A stale rate must not
    // keep the orb beating: stopping is the honest state.
    if (hr && quiet < 0.5) phase = (phase + dt*(hr/60)) % 1;
    const p = phase;
    // Sharp systolic upstroke, slower diastolic decay, plus a dicrotic bump.
    const beat = (1-quietS) * (Math.exp(-p*7.0)*Math.sin(p*Math.PI*2.2)*1.5
                               + Math.exp(-Math.max(p-0.34,0)*9.0)*0.35);

    const a = canvas.clientWidth/Math.max(canvas.clientHeight,1);
    const f = 1/Math.tan(0.62), zn = 0.1, zf = 20;
    const proj = new Float32Array([f/a,0,0,0, 0,f,0,0, 0,0,(zf+zn)/(zn-zf),-1,
                                   0,0,(2*zf*zn)/(zn-zf),0]);
    const ang = now*0.00016;
    const c = Math.cos(ang), s = Math.sin(ang);
    const view = new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,-2.55,1]);

    gl.uniformMatrix4fv(U("uProj"), false, proj);
    gl.uniformMatrix4fv(U("uView"), false, view);
    gl.uniform1f(U("uTime"), now*0.001);
    gl.uniform1f(U("uBeat"), beat);
    gl.uniform1f(U("uQuiet"), quietS);
    gl.uniform1f(U("uConf"), 0.25 + 0.75*confS);
    gl.uniform1f(U("uDetail"), detail);

    gl.clearColor(0,0,0,0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, data.length/3);
    requestAnimationFrame(frame);
  }

  resize(); addEventListener("resize", resize); requestAnimationFrame(frame);

  return {
    /** measuring: beat at `heartRate`. declining: pass null and quiet=1. */
    set(state){
      hr = state.heartRate ?? hr;
      quiet = state.measuring ? 0 : 1;
      conf = state.confidence ?? 0;
      detail = state.detail ?? 1;
    }
  };
}
