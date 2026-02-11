// ═══════════════════════════════════════════════════════
//  FACTORY 3D TAB — Self-contained engine (IIFE)
// ═══════════════════════════════════════════════════════
var fw_activate, fw_deactivate;
(function(){
'use strict';

const CHAR_H=14,CHAR_W=7,FOV=1.15,MAX_DIST=120,WSIZE=512,WM=WSIZE-1;
const GRAV=-30,JUMP_V=10.5,WALK_SPD=7.5,SPRINT_SPD=14,LOOK_SPD=2.2,EYE_H=2.5;
const STEP_HEIGHT=1.3,SLIDE_FACTOR=0.7;
const FACTORY_LIGHT_RADIUS=25;
const FBM_PERSISTENCE=0.48,FBM_LACUNARITY=2.03;
const SUN_DX=0.57,SUN_DY=0.57,SUN_DZ=-0.57;

const GRASS=1,DIRT=2,STONE=3,SAND=4,SNOW=5,WATER=6,TRUNK=7,LEAVES=8;
const CONCRETE=9,METAL=10,MACHINE_BASE=11,MACHINE_ARM=12,ROBOT_BODY=13,ROBOT_HEAD=14,SAWDUST=15,CRATE=16,LIGHT=17;

const cvs=document.getElementById('factoryCanvas');
const ctx=cvs.getContext('2d',{alpha:false});
let W,H,COLS,ROWS,minimapImg=null;
let active=false, rafId=null, lastT=0;

// Joint data from D1 arm
let d1Joints=[0,0,0,0,0,0];
let jointFetchInterval=null;

// Arm 3D tracking via WebSocket
let arm3dWs=null, arm3dPositions=null, arm3dSource='none', arm3dConnected=false;
let arm3dBackoff=1000;
const ARM_LINK_LENGTHS={d0:0.1215,L1:0.2085,L2:0.2085,L3:0.113};
const VOXEL_SCALE=40; // 1 meter ≈ 40 voxels

function arm3dConnect(){
  if(arm3dWs&&arm3dWs.readyState<2)return;
  const proto=location.protocol==='https:'?'wss:':'ws:';
  arm3dWs=new WebSocket(proto+'//'+location.host+'/ws/arm3d');
  arm3dWs.onopen=()=>{arm3dConnected=true;arm3dBackoff=1000;};
  arm3dWs.onclose=()=>{arm3dConnected=false;arm3dPositions=null;arm3dSource='none';arm3dWs=null;if(active){setTimeout(arm3dConnect,arm3dBackoff);arm3dBackoff=Math.min(10000,arm3dBackoff*2);}};
  arm3dWs.onerror=()=>{arm3dConnected=false;};
  arm3dWs.onmessage=(ev)=>{
    try{
      const d=JSON.parse(ev.data);
      if(d.type==='arm3d'&&Array.isArray(d.positions)&&d.positions.length>=5){
        arm3dPositions=d.positions;
        arm3dSource=d.source||'unknown';
        updateArmVoxels(d.positions);
      }
    }catch(e){}
  };
}
function arm3dDisconnect(){if(arm3dWs){arm3dWs.close();arm3dWs=null;}}

// Store original static arm voxels for fallback
let staticArmCells=[];
let armVoxelsDirty=false;

function captureStaticArm(){
  staticArmCells=[];
  const FX=256,FZ=256,armX2=FX-30,armZ2=FZ-40;
  // Scan the arm area for MACHINE_ARM and nearby MACHINE_BASE
  for(let dz=-5;dz<=15;dz++)for(let dx=-5;dx<=15;dx++){
    const x=(armX2+dx)&WM,z=(armZ2+dz)&WM,i=z*WSIZE+x;
    if(bmap[i]===MACHINE_ARM){staticArmCells.push({x,z,h:hmap[i],b:bmap[i]});}
  }
}

function clearArmVoxels(){
  const FX=256,FZ=256,armX2=FX-30,armZ2=FZ-40,FLOOR=2;
  // Clear dynamic arm cells (wider scan)
  for(let dz=-10;dz<=25;dz++)for(let dx=-10;dx<=25;dx++){
    const x=(armX2+dx)&WM,z=(armZ2+dz)&WM,i=z*WSIZE+x;
    if(bmap[i]===MACHINE_ARM){hmap[i]=FLOOR;bmap[i]=CONCRETE;}
  }
}

function updateArmVoxels(positions){
  // positions: [[x,y,z],...] in meters (5 joints: base,shoulder,elbow,wrist,EE)
  // Map real arm coords to factory voxel grid
  // Arm base in factory: armX=FX-30, armZ=FZ-40, height=FLOOR+10
  const FX=256,FZ=256,armX2=FX-30,armZ2=FZ-40,FLOOR=2;

  clearArmVoxels();

  // Real arm base is at positions[0]. Map it to (armX2, armZ2).
  const baseReal=positions[0];
  for(let seg=0;seg<positions.length-1;seg++){
    const p0=positions[seg], p1=positions[seg+1];
    // Convert to voxel offsets relative to arm base
    const dx0=(p0[0]-baseReal[0])*VOXEL_SCALE, dz0=(p0[1]-baseReal[1])*VOXEL_SCALE, dy0=(p0[2]-baseReal[2])*VOXEL_SCALE;
    const dx1=(p1[0]-baseReal[0])*VOXEL_SCALE, dz1=(p1[1]-baseReal[1])*VOXEL_SCALE, dy1=(p1[2]-baseReal[2])*VOXEL_SCALE;

    // Bresenham-like line in 2D (x,z) with height interpolation
    const steps=Math.max(1,Math.ceil(Math.sqrt((dx1-dx0)**2+(dz1-dz0)**2)));
    for(let s=0;s<=steps;s++){
      const t=s/steps;
      const vx=Math.round(armX2+dx0+(dx1-dx0)*t);
      const vz=Math.round(armZ2+dz0+(dz1-dz0)*t);
      const vh=FLOOR+10+dy0+(dy1-dy0)*t; // base height + vertical offset
      const x=vx&WM,z=vz&WM,i=z*WSIZE+x;
      // Only raise, don't lower existing base
      if(vh>hmap[i]||bmap[i]===CONCRETE){hmap[i]=vh;bmap[i]=MACHINE_ARM;}
      // Widen arm by 1 voxel
      for(const[ox,oz] of [[1,0],[-1,0],[0,1],[0,-1]]){
        const nx=(vx+ox)&WM,nz=(vz+oz)&WM,ni=nz*WSIZE+nx;
        const nh=vh-0.5;
        if(nh>hmap[ni]&&bmap[ni]!==MACHINE_BASE){hmap[ni]=nh;bmap[ni]=MACHINE_ARM;}
      }
    }
  }
  armVoxelsDirty=true;
}

function restoreStaticArm(){
  clearArmVoxels();
  for(const c of staticArmCells){const i=c.z*WSIZE+c.x;hmap[i]=c.h;bmap[i]=c.b;}
}

function resize(){
  const rect=cvs.parentElement.getBoundingClientRect();
  W=cvs.width=Math.floor(rect.width);
  H=cvs.height=Math.floor(rect.height);
  COLS=Math.floor(W/CHAR_W);ROWS=Math.floor(H/CHAR_H);
  minimapImg=null;
}

// Noise
const P=new Uint8Array(512);
{const p=new Uint8Array(256);for(let i=0;i<256;i++)p[i]=i;
for(let i=255;i>0;i--){const j=(Math.random()*i)|0;[p[i],p[j]]=[p[j],p[i]];}
for(let i=0;i<512;i++)P[i]=p[i&255];}
function fade(t){return t*t*t*(t*(t*6-15)+10)}
function lrp(a,b,t){return a+t*(b-a)}
function grd(h,x,y){const v=h&3;return v===0?x+y:v===1?-x+y:v===2?x-y:-x-y}
function perlin(x,y){const X=Math.floor(x)&255,Y=Math.floor(y)&255,xf=x-Math.floor(x),yf=y-Math.floor(y),u=fade(xf),v=fade(yf);
return lrp(lrp(grd(P[P[P[X]+Y]],xf,yf),grd(P[P[P[X+1]+Y]],xf-1,yf),u),lrp(grd(P[P[P[X]+Y+1]],xf,yf-1),grd(P[P[P[X+1]+Y+1]],xf-1,yf-1),u),v);}
function fbm(x,y,o){let v=0,a=1,f=1,m=0;for(let i=0;i<o;i++){v+=perlin(x*f,y*f)*a;m+=a;a*=FBM_PERSISTENCE;f*=FBM_LACUNARITY;}return v/m;}
function hash2(x,y){let n=((x*374761393+y*668265263)^((x*1274126177)>>3))|0;n=((n^(n>>13))*1274126177)|0;return((n^(n>>16))&0x7fffffff)/0x7fffffff;}

// World data
const hmap=new Float32Array(WSIZE*WSIZE),bmap=new Uint8Array(WSIZE*WSIZE);
const mods=new Map();
let factoryLights=[];
function modKey(x,z){return (z&WM)*WSIZE+(x&WM);}

// Factory generation (copied from TextWorld)
function generateFactory(){
  hmap.fill(0);bmap.fill(0);mods.clear();factoryLights=[];
  const FX=256,FZ=256,FW=60,FD=80,FLOOR=2,WALL_H=18;
  for(let z=0;z<WSIZE;z++)for(let x=0;x<WSIZE;x++){const i=z*WSIZE+x;hmap[i]=FLOOR-2;bmap[i]=DIRT;}
  for(let z=FZ-FD;z<=FZ+FD;z++)for(let x=FX-FW;x<=FX+FW;x++){const i=(z&WM)*WSIZE+(x&WM);hmap[i]=FLOOR;bmap[i]=CONCRETE;}
  for(let z=FZ-FD;z<=FZ+FD;z++){for(let w=0;w<3;w++){let i=((z)&WM)*WSIZE+((FX-FW-w)&WM);hmap[i]=WALL_H;bmap[i]=METAL;i=((z)&WM)*WSIZE+((FX+FW+w)&WM);hmap[i]=WALL_H;bmap[i]=METAL;}}
  for(let x=FX-FW-2;x<=FX+FW+2;x++){for(let w=0;w<3;w++){let i=((FZ-FD-w)&WM)*WSIZE+((x)&WM);hmap[i]=WALL_H;bmap[i]=METAL;i=((FZ+FD+w)&WM)*WSIZE+((x)&WM);hmap[i]=WALL_H;bmap[i]=METAL;}}

  // Robotic arm
  const armX=FX-30,armZ=FZ-40;
  for(let dz=-2;dz<=2;dz++)for(let dx=-2;dx<=2;dx++){const i=((armZ+dz)&WM)*WSIZE+((armX+dx)&WM);hmap[i]=FLOOR+3;bmap[i]=MACHINE_BASE;}
  for(let dz=-1;dz<=1;dz++)for(let dx=-1;dx<=1;dx++){const i=((armZ+dz)&WM)*WSIZE+((armX+dx)&WM);hmap[i]=FLOOR+8;bmap[i]=MACHINE_BASE;}
  {const i=((armZ)&WM)*WSIZE+((armX)&WM);hmap[i]=FLOOR+10;bmap[i]=MACHINE_BASE;}
  for(let dx=2;dx<=12;dx++){const dropoff=dx>8?(dx-8)*0.6:0;const i=((armZ)&WM)*WSIZE+((armX+dx)&WM);hmap[i]=FLOOR+9-dropoff;bmap[i]=MACHINE_ARM;}
  for(let dz=1;dz<=8;dz++){const dropoff=dz>5?(dz-5)*0.8:0;const i=((armZ+dz)&WM)*WSIZE+((armX+12)&WM);hmap[i]=FLOOR+7-dropoff;bmap[i]=MACHINE_ARM;}
  for(let dz=-1;dz<=1;dz++)for(let dx=-1;dx<=1;dx++){const i=((armZ+8+dz)&WM)*WSIZE+((armX+12+dx)&WM);if(Math.abs(dz)+Math.abs(dx)<=1){hmap[i]=FLOOR+4;bmap[i]=MACHINE_ARM;}}
  for(let dx=-4;dx<=-2;dx++)for(let dz=-1;dz<=1;dz++){const i=((armZ+dz)&WM)*WSIZE+((armX+dx)&WM);hmap[i]=FLOOR+7;bmap[i]=MACHINE_BASE;}

  // Robotic dog
  const dogX=FX+20,dogZ=FZ+25;
  for(let dz=-1;dz<=1;dz++)for(let dx=-3;dx<=3;dx++){const i=((dogZ+dz)&WM)*WSIZE+((dogX+dx)&WM);hmap[i]=FLOOR+3.5;bmap[i]=ROBOT_BODY;}
  for(let dx=-2;dx<=2;dx++){const i=((dogZ)&WM)*WSIZE+((dogX+dx)&WM);hmap[i]=FLOOR+4.2;bmap[i]=ROBOT_BODY;}
  for(const [ldx,ldz] of [[-3,-2],[-3,2],[3,-2],[3,2]]){const i=((dogZ+ldz)&WM)*WSIZE+((dogX+ldx)&WM);hmap[i]=FLOOR+2;bmap[i]=ROBOT_BODY;}
  for(let dz=-1;dz<=1;dz++){const i=((dogZ+dz)&WM)*WSIZE+((dogX+4)&WM);hmap[i]=FLOOR+4.5;bmap[i]=ROBOT_HEAD;}
  {const i=((dogZ)&WM)*WSIZE+((dogX+5)&WM);hmap[i]=FLOOR+5;bmap[i]=ROBOT_HEAD;}
  {const i=((dogZ)&WM)*WSIZE+((dogX+6)&WM);hmap[i]=FLOOR+4;bmap[i]=LIGHT;factoryLights.push([dogX+6,dogZ]);}
  {const i=((dogZ)&WM)*WSIZE+((dogX-4)&WM);hmap[i]=FLOOR+4.8;bmap[i]=ROBOT_BODY;}
  {const i=((dogZ)&WM)*WSIZE+((dogX-5)&WM);hmap[i]=FLOOR+5.5;bmap[i]=ROBOT_BODY;}

  // Sawdust
  const sawdustSpots=[[FX-20,FZ-30],[FX-25,FZ-35],[FX-18,FZ-45],[FX-35,FZ-25],[FX-15,FZ-32],[FX-5,FZ-10],[FX+8,FZ+5],[FX-10,FZ+15],[FX+3,FZ-20],[FX-15,FZ+8],[FX+25,FZ+35],[FX+15,FZ+30],[FX+30,FZ+20],[FX+10,FZ+40],[FX+28,FZ+40],[FX+40,FZ-20],[FX-40,FZ+30],[FX+35,FZ-35],[FX-20,FZ+50],[FX+5,FZ+55],[FX-35,FZ-55],[FX+45,FZ+10],[FX-45,FZ-10]];
  for(const [sx,sz] of sawdustSpots){const ps=1+Math.floor(hash2(sx*7,sz*11)*3),ph=0.3+hash2(sx*3,sz*5)*0.8;
    for(let dz=-ps;dz<=ps;dz++)for(let dx=-ps;dx<=ps;dx++){const dist=Math.sqrt(dx*dx+dz*dz);if(dist<=ps+0.3){const px=(sx+dx)&WM,pz=(sz+dz)&WM,ii=pz*WSIZE+px;if(bmap[ii]===CONCRETE){hmap[ii]=FLOOR+ph*(1-dist/(ps+0.5));bmap[ii]=SAWDUST;}}}}

  // Crates
  for(const [cx2,cz2] of [[FX-50,FZ-65],[FX-48,FZ-65],[FX-50,FZ-63],[FX+45,FZ-60],[FX+47,FZ-60],[FX-50,FZ+60],[FX-48,FZ+60],[FX-50,FZ+58],[FX-48,FZ+58],[FX+40,FZ+65],[FX+42,FZ+65]]){
    for(let dz=0;dz<=1;dz++)for(let dx=0;dx<=1;dx++){const ii=((cz2+dz)&WM)*WSIZE+((cx2+dx)&WM);hmap[ii]=FLOOR+3+hash2(cx2*3+dx,cz2*5+dz)*3;bmap[ii]=CRATE;}}

  // Lights
  for(let lz=FZ-FD+12;lz<=FZ+FD-12;lz+=25){for(let lx=FX-FW+12;lx<=FX+FW-12;lx+=25){const ii=((lz)&WM)*WSIZE+((lx)&WM);if(bmap[ii]===CONCRETE){hmap[ii]=FLOOR+0.1;bmap[ii]=LIGHT;factoryLights.push([lx,lz]);}}}

  // Columns
  for(let cz=FZ-FD+10;cz<=FZ+FD-10;cz+=20){for(let cx=FX-FW+15;cx<=FX+FW-15;cx+=20){const ii=((cz)&WM)*WSIZE+((cx)&WM);hmap[ii]=WALL_H-2;bmap[ii]=METAL;}}
}

// World access
function gH(wx,wz){const x=Math.floor(wx)&WM,z=Math.floor(wz)&WM,k=z*WSIZE+x;return mods.has(k)?mods.get(k).h:hmap[k];}
function gB(wx,wz){const x=Math.floor(wx)&WM,z=Math.floor(wz)&WM,k=z*WSIZE+x;return mods.has(k)?mods.get(k).b:bmap[k];}
function gTop(wx,wz){return gH(wx,wz);}
function surfaceNormal(wx,wz){const hL=gH(wx-1,wz),hR=gH(wx+1,wz),hD=gH(wx,wz-1),hU=gH(wx,wz+1),nx=hL-hR,nz=hD-hU,ny=2,len=Math.sqrt(nx*nx+ny*ny+nz*nz);return[nx/len,ny/len,nz/len];}
function factoryLightAt(wx,wz){let total=0;for(let i=0;i<factoryLights.length;i++){const dx=wx-factoryLights[i][0],dz=wz-factoryLights[i][1],d2=dx*dx+dz*dz,r2=FACTORY_LIGHT_RADIUS*FACTORY_LIGHT_RADIUS;if(d2<r2)total+=1-d2/r2;}return Math.min(1,total);}

// Player
const pl={x:256,y:30,z:276,yaw:-Math.PI/2,pitch:0,vy:0,ground:false,sprint:false};
function initPlayer(){pl.x=256;pl.z=276;pl.y=hmap[(256&WM)*WSIZE+(256&WM)]+EYE_H;pl.yaw=-Math.PI/2;pl.pitch=0;pl.vy=0;}

// Input (only when active)
const keys={};
function onKeyDown(e){
  if(!active)return;
  const k=e.key.toLowerCase();keys[k]=true;
  if([' ','arrowup','arrowdown','arrowleft','arrowright'].includes(k))e.preventDefault();
}
function onKeyUp(e){const k=e.key.toLowerCase();keys[k]=false;}

function updatePlayer(dt){
  if(keys.arrowleft)pl.yaw-=LOOK_SPD*dt;if(keys.arrowright)pl.yaw+=LOOK_SPD*dt;
  if(keys.arrowup)pl.pitch=Math.min(1.3,pl.pitch+LOOK_SPD*dt);if(keys.arrowdown)pl.pitch=Math.max(-1.3,pl.pitch-LOOK_SPD*dt);
  const spd=(pl.sprint?SPRINT_SPD:WALK_SPD)*dt;let mx=0,mz=0;
  if(keys.w){mx+=Math.cos(pl.yaw);mz+=Math.sin(pl.yaw)}if(keys.s){mx-=Math.cos(pl.yaw);mz-=Math.sin(pl.yaw)}
  if(keys.a){mx+=Math.cos(pl.yaw-1.5708);mz+=Math.sin(pl.yaw-1.5708)}if(keys.d){mx-=Math.cos(pl.yaw-1.5708);mz-=Math.sin(pl.yaw-1.5708)}
  const len=Math.sqrt(mx*mx+mz*mz);if(len>0){mx=mx/len*spd;mz=mz/len*spd;}
  const nx=pl.x+mx,nz=pl.z+mz,feet=pl.y-EYE_H;
  if(gTop(nx,nz)<feet+STEP_HEIGHT){pl.x=nx;pl.z=nz;}else{
    const sx=pl.x+mx*SLIDE_FACTOR,sz=pl.z+mz*SLIDE_FACTOR;let movedX=false;
    if(gTop(sx,pl.z)<feet+STEP_HEIGHT){pl.x=sx;movedX=true;}if(gTop(movedX?pl.x:pl.x,sz)<feet+STEP_HEIGHT){pl.z=sz;}}
  const gh=gTop(pl.x,pl.z);pl.vy+=GRAV*dt;pl.y+=pl.vy*dt;
  if(pl.y-EYE_H<=gh){pl.y=gh+EYE_H;pl.vy=0;pl.ground=true;}else pl.ground=false;
  if(keys[' ']&&pl.ground){pl.vy=JUMP_V;pl.ground=false;}pl.sprint=!!keys.shift;
}

// Animation with real joint data
let dogBob=0;
function getAnimatedHeight(bx,bz,baseH,blk,time){
  const FX=256,FZ=256,armX=FX-30,armZ=FZ-40,dogX=FX+20,dogZ=FZ+25;
  if(blk===MACHINE_ARM){
    // If arm3d WebSocket is providing real positions, voxels are already updated — no animation needed
    if(arm3dPositions) return baseH;
    const dx=bx-armX,dz=bz-armZ,dist=Math.sqrt(dx*dx+dz*dz);
    if(dist>1){
      const j0=d1Joints[0]||0, j1=d1Joints[1]||0;
      return baseH+Math.sin(j0+dist*0.2)*0.6+Math.cos(j1+dist*0.15)*0.3;
    }}
  if(blk===ROBOT_BODY||blk===ROBOT_HEAD){const dx=bx-dogX,dz=bz-dogZ;if(Math.abs(dx)<=6&&Math.abs(dz)<=3)return baseH+dogBob;}
  return baseH;
}

// Block visuals (same tables as TextWorld)
const CHT={[GRASS]:[',','"','`','.',',','v','\'',','],[DIRT]:['.','.',':', ';','.','.','.',':'],[STONE]:['#','%','&','#','X','H','%','#'],[SAND]:['~','.',':','~','.','~','.','~'],[SNOW]:['*','.','+',' ','.','+','*','.'],[WATER]:['~','-','~','=','~','-','~','='],[TRUNK]:['O','0','@','O','0','@','O','0'],[LEAVES]:['@','#','&','$','%','@','#','&'],[CONCRETE]:['.',':','.','+','.',':','.','+'],
[METAL]:['#','=','#','|','#','=','#','H'],[MACHINE_BASE]:['@','#','@','O','@','#','@','O'],[MACHINE_ARM]:['-','=','-','<','-','=','-','>'],[ROBOT_BODY]:['[',']','#','=','[',']','#','='],[ROBOT_HEAD]:['@','O','@','o','@','O','@','*'],[SAWDUST]:['.',':', ',','`','.',':',',','.'],[CRATE]:['#','=','#','|','#','=','#','|'],[LIGHT]:['*','+','*','o','*','+','*','O']};
const CHS={[CONCRETE]:['|','.',':','|','.',':','|','.'],[METAL]:['|','[',']','|','[',']','|','#'],[MACHINE_BASE]:['|','#','|','[','|',']','|','#'],[MACHINE_ARM]:['|','-','|','=','|','-','|','='],[ROBOT_BODY]:['|','#','|','=','|','#','|','['],[ROBOT_HEAD]:['|','@','|','O','|','@','|','o'],[SAWDUST]:['.',',',':','.',',','.',':','.'],[CRATE]:['|','#','=','|','#','=','|','#'],[LIGHT]:['|','*','|','+','|','*','|','+'],
[GRASS]:['|','!','|','|','|',']','|','['],[DIRT]:['=','-','=',':','-','=',':','='],[STONE]:['[',']','#','|','[',']','#','|'],[SAND]:['.',':','.',':','.',':','.',':'],[SNOW]:['|','.',':','|','.',':','|','.'],[WATER]:['~','-','~','=','~','-','~','='],[TRUNK]:['|','|','!','|','|','|','|','|'],[LEAVES]:['$','%','&','#','$','%','&','#']};
const CT={[GRASS]:[55,185,55],[DIRT]:[140,100,50],[STONE]:[128,128,133],[SAND]:[215,195,120],[SNOW]:[232,237,248],[WATER]:[25,75,205],[TRUNK]:[125,82,38],[LEAVES]:[28,155,28],[CONCRETE]:[160,158,150],[METAL]:[90,95,110],[MACHINE_BASE]:[200,140,30],[MACHINE_ARM]:[220,170,40],[ROBOT_BODY]:[60,70,90],[ROBOT_HEAD]:[80,90,120],[SAWDUST]:[195,170,110],[CRATE]:[160,120,60],[LIGHT]:[255,220,100]};
const CS={[CONCRETE]:[120,118,112],[METAL]:[60,65,80],[MACHINE_BASE]:[160,110,20],[MACHINE_ARM]:[180,135,25],[ROBOT_BODY]:[40,48,65],[ROBOT_HEAD]:[55,65,90],[SAWDUST]:[155,135,85],[CRATE]:[120,85,35],[LIGHT]:[200,170,60],[GRASS]:[35,120,35],[DIRT]:[105,72,32],[STONE]:[95,95,102],[SAND]:[180,162,100],[SNOW]:[200,205,218],[WATER]:[18,55,175],[TRUNK]:[88,52,22],[LEAVES]:[18,108,18]};

// Renderer
let cBuf,fR,fG,fB;
const colorBatch=new Map();

function render(time){
  const tot=COLS*ROWS;
  if(!cBuf||cBuf.length!==tot){cBuf=new Uint8Array(tot);fR=new Uint8Array(tot);fG=new Uint8Array(tot);fB=new Uint8Array(tot);}
  const FC=[20,18,15];
  dogBob=Math.sin(time*0.003)*0.3;

  // Sky
  for(let r=0;r<ROWS;r++){const t=r/ROWS;const sr=18+(35-18)*t,sg=16+(30-16)*t,sb=14+(25-14)*t;
    for(let c=0;c<COLS;c++){const i=r*COLS+c;cBuf[i]=32;fR[i]=sr;fG[i]=sg;fB[i]=sb;}}

  // Raycasting
  const maxD=MAX_DIST,pS=ROWS*0.6,pO=pl.pitch*ROWS*0.45,hF=FOV/2,iM=1/maxD;
  for(let col=0;col<COLS;col++){
    const sx=(col/COLS-0.5)*2,ang=pl.yaw+sx*hF,ca=Math.cos(ang),sa=Math.sin(ang),cc=Math.cos(sx*hF);
    let mR=ROWS,dist=0.8;
    while(dist<maxD&&mR>0){
      const wx=pl.x+ca*dist,wz=pl.z+sa*dist,pd=dist*cc,bx=Math.floor(wx)&WM,bz=Math.floor(wz)&WM;
      let tH=gH(wx,wz);const blk=gB(wx,wz);
      tH=getAnimatedHeight(bx,bz,tH,blk,time);
      const topH=tH;
      const pT=Math.floor((pl.y-topH)/pd*pS+ROWS/2-pO);
      if(pT<mR){
        const sR=Math.max(0,pT),eR=Math.min(ROWS,mR),fT=dist*iM,fT2=fT*fT;
        const dS=Math.min(1,3.5/pd)*0.5+0.5,cS2=((bx*7+bz*13)&7);
        const ptLight=factoryLightAt(wx,wz);
        for(let row=sR;row<eR;row++){
          const wY=pl.y-(row-ROWS/2+pO)*pd/pS;
          let bt;const isT=row<=sR+2||wY>tH-0.3;
          if(wY>tH-0.5){bt=blk;}else if(wY>tH-2){bt=blk;}else{bt=CONCRETE;}
          const ci=(cS2+row)&7;let ch,bRc,bGc,bBc;
          if(isT&&CHT[bt]){ch=CHT[bt][ci].charCodeAt(0);bRc=CT[bt][0];bGc=CT[bt][1];bBc=CT[bt][2];}
          else if(CHS[bt]){ch=CHS[bt][ci].charCodeAt(0);bRc=CS[bt][0];bGc=CS[bt][1];bBc=CS[bt][2];}
          else{ch=35;bRc=100;bGc=100;bBc=100;}
          if(bt===LIGHT){const pulse=Math.sin(time*0.004)*0.4+0.6;bRc=255*pulse;bGc=80*pulse+100;bBc=40*pulse;}
          if(bt===MACHINE_ARM){const shimmer=Math.sin(time*0.002+dist*0.1)*0.15+1;bRc*=shimmer;bGc*=shimmer;}
          const lightMul=0.35+ptLight*0.65;bRc*=lightMul;bGc*=lightMul;bBc*=lightMul;
          const sh=dS*(1-fT2*0.5);const idx=row*COLS+col;cBuf[idx]=ch;
          fR[idx]=Math.max(0,Math.min(255,(bRc*sh+(FC[0]-bRc*sh)*fT2)))|0;
          fG[idx]=Math.max(0,Math.min(255,(bGc*sh+(FC[1]-bGc*sh)*fT2)))|0;
          fB[idx]=Math.max(0,Math.min(255,(bBc*sh+(FC[2]-bBc*sh)*fT2)))|0;
        }mR=sR;}
      dist+=Math.max(0.25,dist*0.007);
    }
  }

  // Crosshair
  const cx2=COLS>>1,cy2=ROWS>>1;
  const sC=(r,c,ch)=>{if(r>=0&&r<ROWS&&c>=0&&c<COLS){const i=r*COLS+c;cBuf[i]=ch;fR[i]=220;fG[i]=220;fB[i]=220;}};
  sC(cy2,cx2,43);sC(cy2,cx2-1,45);sC(cy2,cx2+1,45);sC(cy2-1,cx2,124);sC(cy2+1,cx2,124);

  // Batch draw
  ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);
  ctx.font=CHAR_H+'px "Courier New",monospace';ctx.textBaseline='top';
  colorBatch.clear();
  for(let r=0;r<ROWS;r++){const y=r*CHAR_H;for(let c=0;c<COLS;c++){const i=r*COLS+c;if(cBuf[i]===32)continue;
    const qr=fR[i]&0xFC,qg=fG[i]&0xFC,qb=fB[i]&0xFC,ck=(qr<<16)|(qg<<8)|qb;
    if(!colorBatch.has(ck))colorBatch.set(ck,[]);colorBatch.get(ck).push(c*CHAR_W,y,cBuf[i]);}}
  for(const [ck,entries] of colorBatch){const r=(ck>>16)&0xFF,g=(ck>>8)&0xFF,b=ck&0xFF;
    ctx.fillStyle='rgb('+r+','+g+','+b+')';for(let j=0;j<entries.length;j+=3)ctx.fillText(String.fromCharCode(entries[j+2]),entries[j],entries[j+1]);}

  drawHUD(time);
}

// HUD
let fps=0,fc=0,fpsT=0;
const MINIMAP_SIZE=72;
const BN={[CONCRETE]:'Factory Floor',[METAL]:'Wall',[MACHINE_BASE]:'Robotic Arm Base',[MACHINE_ARM]:'Robotic Arm',[ROBOT_BODY]:'Robotic Dog',[ROBOT_HEAD]:'Dog Sensor',[SAWDUST]:'Sawdust',[CRATE]:'Crate',[DIRT]:'Outside',[LIGHT]:'Light'};
function drawHUD(t){
  fc++;if(t-fpsT>1000){fps=fc;fc=0;fpsT=t;}
  const blk=gB(pl.x,pl.z),bm=BN[blk]||'???';
  ctx.font='12px "Courier New",monospace';ctx.textBaseline='top';
  const trackLabel=arm3dPositions?'3D:'+arm3dSource:'3D:off';
  const lines=['FACTORY EXPLORER (D1 Live)  FPS '+fps,'XYZ '+pl.x.toFixed(1)+' '+(pl.y-EYE_H).toFixed(1)+' '+pl.z.toFixed(1),bm,
    'J0:'+d1Joints[0].toFixed(2)+' J1:'+d1Joints[1].toFixed(2)+'  '+trackLabel];
  ctx.fillStyle='rgba(0,0,0,0.6)';ctx.fillRect(8,8,310,lines.length*15+10);
  ctx.fillStyle='#ffaa33';lines.forEach((l,i)=>ctx.fillText(l,14,13+i*15));

  // Minimap
  const ms=MINIMAP_SIZE,mx2=W-ms-14,my2=14;
  ctx.fillStyle='rgba(0,0,0,0.6)';ctx.fillRect(mx2-3,my2-3,ms+6,ms+6);
  if(!minimapImg||minimapImg.width!==ms)minimapImg=ctx.createImageData(ms,ms);
  const d=minimapImg.data;
  for(let my=0;my<ms;my++)for(let mmx=0;mmx<ms;mmx++){
    const wx2=Math.floor(pl.x+(mmx-ms/2)*0.8)&WM,wz2=Math.floor(pl.z+(my-ms/2)*0.8)&WM;
    const b=bmap[wz2*WSIZE+wx2],h2=hmap[wz2*WSIZE+wx2],br=Math.min(1,0.3+h2/20);
    const bc=CT[b]||[80,80,80];const off=(my*ms+mmx)*4;d[off]=bc[0]*br;d[off+1]=bc[1]*br;d[off+2]=bc[2]*br;d[off+3]=255;}
  ctx.putImageData(minimapImg,mx2,my2);
  ctx.fillStyle='#ff2222';ctx.fillRect(mx2+ms/2-1,my2+ms/2-1,3,3);
  ctx.strokeStyle='#ff2222';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(mx2+ms/2,my2+ms/2);
  ctx.lineTo(mx2+ms/2+Math.cos(pl.yaw)*7,my2+ms/2+Math.sin(pl.yaw)*7);ctx.stroke();

  // Controls hint
  ctx.fillStyle='rgba(0,0,0,0.45)';ctx.fillRect(8,H-42,400,34);
  ctx.fillStyle='rgba(255,170,51,0.4)';ctx.fillText('WASD:Move  Arrows:Look  Space:Jump  Shift:Run',14,H-34);
}

// Game loop
let worldGenerated=false;
function loop(t){
  if(!active)return;
  const dt=Math.min(0.05,(t-lastT)/1000);lastT=t;
  updatePlayer(dt);render(t);
  rafId=requestAnimationFrame(loop);
}

// Fetch D1 joint state
function fetchJoints(){
  fetch('/api/state').then(r=>r.json()).then(data=>{
    if(data&&data.joints&&Array.isArray(data.joints)){d1Joints=data.joints;}
  }).catch(()=>{});
}

// Activate/deactivate
fw_activate=function(){
  if(active)return;
  active=true;
  resize();
  if(!worldGenerated){generateFactory();captureStaticArm();initPlayer();worldGenerated=true;}
  cvs.addEventListener('keydown',onKeyDown);
  cvs.addEventListener('keyup',onKeyUp);
  window.addEventListener('resize',resize);
  cvs.focus();
  lastT=performance.now();
  rafId=requestAnimationFrame(loop);
  jointFetchInterval=setInterval(fetchJoints,500);
  fetchJoints();
  arm3dConnect();
};

fw_deactivate=function(){
  if(!active)return;
  active=false;
  if(rafId){cancelAnimationFrame(rafId);rafId=null;}
  if(jointFetchInterval){clearInterval(jointFetchInterval);jointFetchInterval=null;}
  arm3dDisconnect();
  if(arm3dPositions){restoreStaticArm();arm3dPositions=null;}
  cvs.removeEventListener('keydown',onKeyDown);
  cvs.removeEventListener('keyup',onKeyUp);
  window.removeEventListener('resize',resize);
  for(const k in keys)keys[k]=false;
};

})();

