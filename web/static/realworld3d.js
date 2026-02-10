//  REAL WORLD 3D TAB — Visual hull voxel raycaster (IIFE)
// ═══════════════════════════════════════════════════════
var rw_activate, rw_deactivate;
(function(){
'use strict';

const CHAR_H=14,CHAR_W=7,FOV=1.15,MAX_DIST=120,WSIZE=128,WM=WSIZE-1;
const GRAV=-30,JUMP_V=10.5,WALK_SPD=7.5,SPRINT_SPD=14,LOOK_SPD=2.2,EYE_H=2.5;
const STEP_HEIGHT=1.3,SLIDE_FACTOR=0.7;
const GRID_W=64,GRID_H=32,GRID_D=64;

// Voxel grid offset in world space (center the reconstruction)
const OX=32,OY=0,OZ=32;

const cvs=document.getElementById('realworldCanvas');
const ctx=cvs.getContext('2d',{alpha:false});
let W,H,COLS,ROWS;
let active=false,rafId=null,lastT=0;

function resize(){
  const rect=cvs.parentElement.getBoundingClientRect();
  W=cvs.width=Math.floor(rect.width);
  H=cvs.height=Math.floor(rect.height);
  COLS=Math.floor(W/CHAR_W);ROWS=Math.floor(H/CHAR_H);
}

// Height map and block map for raycaster
const hmap=new Float32Array(WSIZE*WSIZE);
const bmap=new Uint8Array(WSIZE*WSIZE);
// Per-voxel color storage (R,G,B per column — we store top voxel color)
const vR=new Uint8Array(WSIZE*WSIZE);
const vG=new Uint8Array(WSIZE*WSIZE);
const vB=new Uint8Array(WSIZE*WSIZE);

// WebSocket connection
let rwWs=null;
let wsStatus='Connecting...';
let voxelCount=0;
let frameNum=0;
let cam0ok=false,cam1ok=false;
let rwBackoff=1000;

function rwConnect(){
  if(rwWs&&rwWs.readyState<2)return;
  const proto=location.protocol==='https:'?'wss:':'ws:';
  rwWs=new WebSocket(proto+'//'+location.host+'/ws/realworld3d');
  rwWs.onopen=()=>{wsStatus='Connected';rwBackoff=1000;};
  rwWs.onclose=()=>{wsStatus='Disconnected';rwWs=null;if(active){setTimeout(rwConnect,rwBackoff);rwBackoff=Math.min(10000,rwBackoff*2);}};
  rwWs.onerror=()=>{wsStatus='Error';};
  rwWs.onmessage=(ev)=>{
    try{
      const d=JSON.parse(ev.data);
      if(d.type==='status'){wsStatus=d.message;return;}
      if(d.type==='voxels'){
        loadVoxels(d);
        return;
      }
    }catch(e){}
  };
}

function rwDisconnect(){
  if(rwWs){rwWs.close();rwWs=null;}
}

function loadVoxels(data){
  // Clear grid
  hmap.fill(0);bmap.fill(0);vR.fill(0);vG.fill(0);vB.fill(0);

  // Build floor
  for(let z=0;z<WSIZE;z++)for(let x=0;x<WSIZE;x++){
    const i=z*WSIZE+x;
    hmap[i]=-0.5;bmap[i]=1;// floor
    vR[i]=30;vG[i]=30;vB[i]=35;
  }

  const voxels=data.voxels;
  voxelCount=voxels.length;
  frameNum=data.frame||0;
  cam0ok=!!data.cam0;cam1ok=!!data.cam1;

  // For each voxel column, find the maximum height and store top color
  // Voxels are [x,y,z,r,g,b] in grid coords
  for(let i=0;i<voxels.length;i++){
    const v=voxels[i];
    const wx=(v[0]+OX)&WM, wz=(v[2]+OZ)&WM;
    const wy=v[1]*0.5; // scale Y to world units
    const idx=wz*WSIZE+wx;
    if(wy+0.5>hmap[idx]){
      hmap[idx]=wy+0.5;
      bmap[idx]=2;// voxel block
      vR[idx]=v[3];vG[idx]=v[4];vB[idx]=v[5];
    }
  }
}

// World access
function gH(wx,wz){const x=Math.floor(wx)&WM,z=Math.floor(wz)&WM;return hmap[z*WSIZE+x];}
function gB(wx,wz){const x=Math.floor(wx)&WM,z=Math.floor(wz)&WM;return bmap[z*WSIZE+x];}
function gC(wx,wz){const x=Math.floor(wx)&WM,z=Math.floor(wz)&WM,i=z*WSIZE+x;return[vR[i],vG[i],vB[i]];}

// Player
const pl={x:OX+GRID_W/2,y:25,z:OZ+GRID_D+15,yaw:-Math.PI/2,pitch:-0.3,vy:0,ground:false,sprint:false};
function initPlayer(){
  pl.x=OX+GRID_W/2;pl.z=OZ+GRID_D+15;pl.y=25;
  pl.yaw=-Math.PI/2;pl.pitch=-0.3;pl.vy=0;
}

// Input
const keys={};
function onKeyDown(e){
  if(!active)return;
  const k=e.key.toLowerCase();keys[k]=true;
  if([' ','arrowup','arrowdown','arrowleft','arrowright'].includes(k))e.preventDefault();
}
function onKeyUp(e){keys[e.key.toLowerCase()]=false;}

function updatePlayer(dt){
  if(keys.arrowleft)pl.yaw-=LOOK_SPD*dt;if(keys.arrowright)pl.yaw+=LOOK_SPD*dt;
  if(keys.arrowup)pl.pitch=Math.min(1.3,pl.pitch+LOOK_SPD*dt);if(keys.arrowdown)pl.pitch=Math.max(-1.3,pl.pitch-LOOK_SPD*dt);
  const spd=(pl.sprint?SPRINT_SPD:WALK_SPD)*dt;let mx=0,mz=0;
  if(keys.w){mx+=Math.cos(pl.yaw);mz+=Math.sin(pl.yaw)}if(keys.s){mx-=Math.cos(pl.yaw);mz-=Math.sin(pl.yaw)}
  if(keys.a){mx+=Math.cos(pl.yaw-1.5708);mz+=Math.sin(pl.yaw-1.5708)}if(keys.d){mx-=Math.cos(pl.yaw-1.5708);mz-=Math.sin(pl.yaw-1.5708)}
  const len=Math.sqrt(mx*mx+mz*mz);if(len>0){mx=mx/len*spd;mz=mz/len*spd;}
  const nx=pl.x+mx,nz=pl.z+mz,feet=pl.y-EYE_H;
  if(gH(nx,nz)<feet+STEP_HEIGHT){pl.x=nx;pl.z=nz;}else{
    if(gH(pl.x+mx*SLIDE_FACTOR,pl.z)<feet+STEP_HEIGHT)pl.x+=mx*SLIDE_FACTOR;
    if(gH(pl.x,pl.z+mz*SLIDE_FACTOR)<feet+STEP_HEIGHT)pl.z+=mz*SLIDE_FACTOR;
  }
  const gh=gH(pl.x,pl.z);pl.vy+=GRAV*dt;pl.y+=pl.vy*dt;
  if(pl.y-EYE_H<=gh){pl.y=gh+EYE_H;pl.vy=0;pl.ground=true;}else pl.ground=false;
  if(keys[' ']&&pl.ground){pl.vy=JUMP_V;pl.ground=false;}pl.sprint=!!keys.shift;
}

// Renderer
let cBuf,fR,fG,fB;
const colorBatch=new Map();

// Side face chars for walls
const SIDE_CHARS=['|','[',']','#','|','[',']','#'];
const TOP_CHARS=['.',':','+','=','.',':','+','='];

function render(time){
  const tot=COLS*ROWS;
  if(!cBuf||cBuf.length!==tot){cBuf=new Uint8Array(tot);fR=new Uint8Array(tot);fG=new Uint8Array(tot);fB=new Uint8Array(tot);}

  // Dark sky
  for(let r=0;r<ROWS;r++){const t=r/ROWS;
    for(let c=0;c<COLS;c++){const i=r*COLS+c;cBuf[i]=32;fR[i]=8+t*12|0;fG[i]=10+t*10|0;fB[i]=18+t*8|0;}}

  const maxD=MAX_DIST,pS=ROWS*0.6,pO=pl.pitch*ROWS*0.45,hF=FOV/2,iM=1/maxD;
  for(let col=0;col<COLS;col++){
    const sx=(col/COLS-0.5)*2,ang=pl.yaw+sx*hF,ca=Math.cos(ang),sa=Math.sin(ang),cc=Math.cos(sx*hF);
    let mR=ROWS,dist=0.8;
    while(dist<maxD&&mR>0){
      const wx=pl.x+ca*dist,wz=pl.z+sa*dist,pd=dist*cc;
      const tH=gH(wx,wz);const blk=gB(wx,wz);
      const pT=Math.floor((pl.y-tH)/pd*pS+ROWS/2-pO);
      if(pT<mR){
        const sR=Math.max(0,pT),eR=Math.min(ROWS,mR),fT=dist*iM,fT2=fT*fT;
        const dS=Math.min(1,3.5/pd)*0.5+0.5;
        const bx=Math.floor(wx)&WM,bz=Math.floor(wz)&WM;
        const ci=((bx*7+bz*13)&7);
        const [cr,cg,cb]=gC(wx,wz);
        // Simple point lighting from above center
        const ldx=wx-(OX+GRID_W/2),ldz=wz-(OZ+GRID_D/2),ld2=ldx*ldx+ldz*ldz;
        const ptLight=Math.max(0.3,1-ld2/(80*80));

        for(let row=sR;row<eR;row++){
          const isT=row<=sR+2;
          let ch,bRc,bGc,bBc;
          if(blk===0){continue;}
          else if(blk===1){// floor
            ch=TOP_CHARS[ci].charCodeAt(0);bRc=cr;bGc=cg;bBc=cb;
          }else{// voxel
            if(isT){ch=TOP_CHARS[ci].charCodeAt(0);}else{ch=SIDE_CHARS[ci].charCodeAt(0);}
            bRc=cr;bGc=cg;bBc=cb;
            if(!isT){bRc=bRc*0.7|0;bGc=bGc*0.7|0;bBc=bBc*0.7|0;}
          }
          const lightMul=0.4+ptLight*0.6;bRc=bRc*lightMul|0;bGc=bGc*lightMul|0;bBc=bBc*lightMul|0;
          const sh=dS*(1-fT2*0.5);const idx=row*COLS+col;cBuf[idx]=ch;
          fR[idx]=Math.max(0,Math.min(255,(bRc*sh+(8-bRc*sh)*fT2)))|0;
          fG[idx]=Math.max(0,Math.min(255,(bGc*sh+(10-bGc*sh)*fT2)))|0;
          fB[idx]=Math.max(0,Math.min(255,(bBc*sh+(18-bBc*sh)*fT2)))|0;
        }mR=sR;}
      dist+=Math.max(0.25,dist*0.007);
    }
  }

  // Crosshair
  const cx2=COLS>>1,cy2=ROWS>>1;
  const sC=(r,c,ch)=>{if(r>=0&&r<ROWS&&c>=0&&c<COLS){const i=r*COLS+c;cBuf[i]=ch;fR[i]=0;fG[i]=255;fB[i]=100;}};
  sC(cy2,cx2,43);sC(cy2,cx2-1,45);sC(cy2,cx2+1,45);sC(cy2-1,cx2,124);sC(cy2+1,cx2,124);

  // Batch draw
  ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);
  ctx.font=CHAR_H+'px "Courier New",monospace';ctx.textBaseline='top';
  colorBatch.clear();
  for(let r=0;r<ROWS;r++){const y=r*CHAR_H;for(let c=0;c<COLS;c++){const i=r*COLS+c;if(cBuf[i]===32)continue;
    const qr=fR[i]&0xFC,qg=fG[i]&0xFC,qb=fB[i]&0xFC,ck=(qr<<16)|(qg<<8)|qb;
    if(!colorBatch.has(ck))colorBatch.set(ck,[]);colorBatch.get(ck).push(c*CHAR_W,y,cBuf[i]);}}
  for(const [ck,entries] of colorBatch){const r2=(ck>>16)&0xFF,g2=(ck>>8)&0xFF,b2=ck&0xFF;
    ctx.fillStyle='rgb('+r2+','+g2+','+b2+')';for(let j=0;j<entries.length;j+=3)ctx.fillText(String.fromCharCode(entries[j+2]),entries[j],entries[j+1]);}

  drawHUD(time);
}

// HUD
let fps=0,fc=0,fpsT=0;
function drawHUD(t){
  fc++;if(t-fpsT>1000){fps=fc;fc=0;fpsT=t;}
  ctx.font='12px "Courier New",monospace';ctx.textBaseline='top';
  const cam0s=cam0ok?'✓':'✗',cam1s=cam1ok?'✓':'✗';
  const lines=[
    'REAL WORLD 3D — Visual Hull Reconstruction',
    'FPS '+fps+'  Voxels: '+voxelCount+'  Frame: '+frameNum,
    'CAM0(side):'+cam0s+'  CAM1(arm):'+cam1s+'  '+wsStatus,
    'XYZ '+pl.x.toFixed(1)+' '+(pl.y-EYE_H).toFixed(1)+' '+pl.z.toFixed(1),
    'R: Reset BG  |  WASD: Move  Arrows: Look  Space: Jump'
  ];
  ctx.fillStyle='rgba(0,0,0,0.7)';ctx.fillRect(8,8,420,lines.length*15+10);
  ctx.fillStyle='#00ff88';lines.forEach((l,i)=>ctx.fillText(l,14,13+i*15));
}

// Game loop
function loop(t){
  if(!active)return;
  const dt=Math.min(0.05,(t-lastT)/1000);lastT=t;
  // Check R key for background reset
  if(keys.r&&rwWs&&rwWs.readyState===1){
    rwWs.send(JSON.stringify({type:'capture_bg'}));
    keys.r=false;
  }
  updatePlayer(dt);render(t);
  rafId=requestAnimationFrame(loop);
}

rw_activate=function(){
  if(active)return;
  active=true;
  resize();
  initPlayer();
  // Initialize with empty floor
  loadVoxels({voxels:[],gridW:GRID_W,gridH:GRID_H,gridD:GRID_D,frame:0,cam0:false,cam1:false});
  cvs.addEventListener('keydown',onKeyDown);
  cvs.addEventListener('keyup',onKeyUp);
  window.addEventListener('resize',resize);
  cvs.focus();
  lastT=performance.now();
  rafId=requestAnimationFrame(loop);
  rwConnect();
};

rw_deactivate=function(){
  if(!active)return;
  active=false;
  if(rafId){cancelAnimationFrame(rafId);rafId=null;}
  rwDisconnect();
  cvs.removeEventListener('keydown',onKeyDown);
  cvs.removeEventListener('keyup',onKeyUp);
  window.removeEventListener('resize',resize);
  for(const k in keys)keys[k]=false;
};

})();
