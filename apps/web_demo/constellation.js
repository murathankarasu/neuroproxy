/**
 * A small dependency-free 3D data scene for the researcher dashboard.
 * Each study session becomes a node on the globe. Node colour and height
 * encode usability and median heart rate; the scene remains an explicitly
 * empty wireframe when no sessions have arrived yet.
 */

export function createConstellation(canvas) {
  const ctx = canvas.getContext("2d");
  const pointer = {x: 0, y: 0, active: false};
  let width = 1, height = 1, dpr = 1, sessions = [];
  let usableRate = 0, answeredRate = 0, targetSpin = 0.00013;

  const seeds = Array.from({length: 86}, (_, i) => {
    const y = 1 - (i / 85) * 2;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const a = i * Math.PI * (3 - Math.sqrt(5));
    return {x: Math.cos(a) * radius, y, z: Math.sin(a) * radius};
  });

  function resize() {
    dpr = Math.min(devicePixelRatio || 1, 2);
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = Math.max(1, width * dpr);
    canvas.height = Math.max(1, height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function project(p, spin, tilt, scale) {
    const ca = Math.cos(spin), sa = Math.sin(spin);
    const x1 = p.x * ca - p.z * sa;
    const z1 = p.x * sa + p.z * ca;
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    const y1 = p.y * ct - z1 * st;
    const z2 = p.y * st + z1 * ct;
    const perspective = 2.8 / (3.2 - z2);
    return {
      x: width * 0.5 + x1 * scale * perspective,
      y: height * 0.48 + y1 * scale * perspective,
      z: z2,
      perspective
    };
  }

  function nodePoint(session, i) {
    const base = seeds[(i * 13 + 7) % seeds.length];
    const hrLift = session.median_hr == null ? 0 : Math.max(-.18, Math.min(.22, (session.median_hr - 74) / 110));
    const lift = session.usable ? .16 + hrLift : .04;
    return {x: base.x * (1 + lift), y: base.y * (1 + lift), z: base.z * (1 + lift)};
  }

  function frame(now) {
    ctx.clearRect(0, 0, width, height);
    const scale = Math.min(width, height) * .38;
    const spin = now * targetSpin + pointer.x * .24;
    const tilt = -.22 + pointer.y * .15;
    const projected = seeds.map(p => project(p, spin, tilt, scale));

    // Atmospheric bloom behind the data sphere.
    const glow = ctx.createRadialGradient(width*.5, height*.48, 4, width*.5, height*.48, scale*1.35);
    glow.addColorStop(0, `rgba(65,225,255,${.08 + answeredRate*.08})`);
    glow.addColorStop(.48, `rgba(91,111,255,${.045 + usableRate*.045})`);
    glow.addColorStop(1, "rgba(2,6,15,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, width, height);

    // Quiet latitude rings make the volume readable without implying values.
    ctx.lineWidth = 1;
    [-.55, 0, .55].forEach((lat, li) => {
      ctx.beginPath();
      for (let i=0; i<=80; i++) {
        const a = i / 80 * Math.PI * 2;
        const r = Math.sqrt(1-lat*lat);
        const q = project({x:Math.cos(a)*r,y:lat,z:Math.sin(a)*r}, spin, tilt, scale);
        i ? ctx.lineTo(q.x,q.y) : ctx.moveTo(q.x,q.y);
      }
      ctx.strokeStyle = `rgba(115,160,210,${li===1?.13:.075})`;
      ctx.stroke();
    });

    // Mesh edges between nearby Fibonacci points.
    ctx.lineWidth = .65;
    for (let i=0; i<projected.length; i++) {
      for (let j=i+1; j<projected.length; j++) {
        const a=seeds[i], b=seeds[j];
        const d=(a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2;
        if (d > .27) continue;
        const pa=projected[i], pb=projected[j];
        const alpha=Math.max(0,.08 + (pa.z+pb.z)*.035);
        ctx.strokeStyle=`rgba(103,145,190,${alpha})`;
        ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
      }
    }

    // Sessions sit above the neutral mesh. Pulsing is decorative only; colour
    // and the legend carry the actual meaning.
    sessions.slice(0, 36).map((s,i) => ({s, p:project(nodePoint(s,i),spin,tilt,scale)}))
      .sort((a,b)=>a.p.z-b.p.z)
      .forEach(({s,p},i) => {
        const pulse = 1 + Math.sin(now*.0024+i)*.13;
        const r = (s.usable ? 4.2 : 3.3) * p.perspective * pulse;
        const color = s.usable ? [72,231,184] : [255,119,136];
        const halo=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,r*3.8);
        halo.addColorStop(0,`rgba(${color.join(',')},.72)`);
        halo.addColorStop(.22,`rgba(${color.join(',')},.28)`);
        halo.addColorStop(1,`rgba(${color.join(',')},0)`);
        ctx.fillStyle=halo; ctx.beginPath(); ctx.arc(p.x,p.y,r*3.8,0,Math.PI*2); ctx.fill();
        ctx.fillStyle=`rgb(${color.join(',')})`;
        ctx.beginPath(); ctx.arc(p.x,p.y,Math.max(1.5,r),0,Math.PI*2); ctx.fill();
      });

    requestAnimationFrame(frame);
  }

  canvas.addEventListener("pointermove", e => {
    const r=canvas.getBoundingClientRect();
    pointer.x=(e.clientX-r.left)/Math.max(r.width,1)-.5;
    pointer.y=(e.clientY-r.top)/Math.max(r.height,1)-.5;
    pointer.active=true;
  });
  canvas.addEventListener("pointerleave", () => { pointer.x=0; pointer.y=0; pointer.active=false; });
  addEventListener("resize", resize);
  resize(); requestAnimationFrame(frame);

  return {
    set(data) {
      sessions = data.sessions || [];
      usableRate = data.usableRate || 0;
      answeredRate = data.answeredRate || 0;
      targetSpin = sessions.length ? .00011 + Math.min(sessions.length,20)*.000002 : .00008;
    }
  };
}
