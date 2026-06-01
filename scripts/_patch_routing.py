"""Patch app.py: replace _ROUTING_HTML_TEMPLATE with improved version."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP  = ROOT / "app.py"

NEW_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    html, body { width:100%; height:100%; overflow:hidden; font-family:'Segoe UI',sans-serif; }
    #map { width:100%; height:570px; }
    #aerial-bar {
      width:100%; height:54px;
      background:linear-gradient(90deg,#060b14,#0d1b2a,#060b14);
      border-top:2px solid rgba(100,116,139,0.4);
      display:flex; align-items:center; justify-content:center;
      font-size:13px; color:#94a3b8; gap:24px; padding:0 20px;
      transition:border-top-color 0.3s;
    }
    #aerial-bar.has-save { border-top-color:#22c55e; }
    #aerial-bar.no-save  { border-top-color:#475569; }
    .ab-seg { display:flex; flex-direction:column; align-items:center; gap:2px; }
    .ab-lbl { font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:.6px; }
    .ab-val { font-size:15px; font-weight:700; }
    .ab-val.ground { color:#ef4444; }
    .ab-val.air    { color:#22c55e; }
    .ab-val.dim    { color:#94a3b8; }
    .ab-div { width:1px; height:30px; background:#1e2a3a; }
    .ab-badge {
      font-size:14px; font-weight:700; color:#22c55e;
      background:rgba(34,197,94,.12); border:1px solid rgba(34,197,94,.3);
      border-radius:6px; padding:5px 14px;
    }
    .ab-neutral { font-size:13px; color:#64748b; }
    .ctrl-btn {
      background:#fff; border:2px solid rgba(0,0,0,.2); border-radius:6px;
      width:36px; height:36px; font-size:17px; cursor:pointer;
      display:flex; align-items:center; justify-content:center;
    }
    .ctrl-btn:hover { background:#f0f0f0; }
    #status-bar {
      display:none; position:absolute; top:10px; left:50%;
      transform:translateX(-50%);
      background:rgba(255,255,255,.95); border-radius:20px;
      padding:7px 18px; font-size:13px;
      z-index:999; box-shadow:0 2px 10px rgba(0,0,0,.2); white-space:nowrap;
    }
    #compare-panel {
      display:none; position:absolute; bottom:10px; left:50%;
      transform:translateX(-50%);
      background:rgba(10,12,28,.97); color:#e8e8f0; border-radius:10px;
      padding:12px 18px 10px; box-shadow:0 4px 24px rgba(0,0,0,.7);
      font-size:12px; z-index:1000; min-width:500px; max-width:720px;
    }
    #compare-panel h4 { margin:0; font-size:14px; color:#00d4ff; display:inline; }
    #compare-panel .close-x { float:right; background:none; border:none; color:#6b7280; cursor:pointer; font-size:16px; }
    #compare-panel .from-to { font-size:11px; color:#64748b; margin:4px 0 8px; }
    #compare-panel table { width:100%; border-collapse:collapse; }
    #compare-panel th { padding:4px 10px; color:#475569; border-bottom:1px solid #1e2a3a; font-size:11px; text-align:left; font-weight:500; }
    #compare-panel td { padding:5px 10px; border-bottom:1px solid #0f1623; }
    #compare-panel tr.gr-best td { background:rgba(239,68,68,.1); }
    #compare-panel tr.ai-best td { background:rgba(34,197,94,.1); }
    #compare-panel .tag { display:inline-block; border-radius:3px; padding:1px 5px; font-size:10px; margin-left:4px; }
    #compare-panel .tag-f { background:#22c55e; color:#fff; }
    #compare-panel .tag-a { background:#00d4ff; color:#0a0c1c; }
    #compare-panel .legend { margin-top:8px; font-size:11px; color:#475569; display:flex; gap:18px; flex-wrap:wrap; }
    #compare-panel .ld { display:inline-block; width:14px; height:3px; border-radius:2px; margin-right:4px; vertical-align:middle; }
    #route-info {
      display:none; position:absolute; bottom:10px; left:50%;
      transform:translateX(-50%);
      background:rgba(255,255,255,.96); border-radius:8px;
      padding:8px 16px; box-shadow:0 2px 12px rgba(0,0,0,.25);
      font-size:12px; z-index:1000; text-align:center;
    }
  </style>
</head>
<body>
<div style="position:relative">
  <div id="map"></div>
  <div id="status-bar"></div>
  <div id="compare-panel">
    <button class="close-x" onclick="clearMM()">&times;</button>
    <h4>&#x1F9ED; Multi-Modal Route Comparison</h4>
    <div class="from-to" id="cp-from-to"></div>
    <table>
      <thead><tr>
        <th>Mode</th><th>Distance</th><th>Time</th><th>Aerial Leg</th>
      </tr></thead>
      <tbody id="cp-tbody"></tbody>
    </table>
    <div class="legend">
      <span><span class="ld" style="background:#ef4444"></span>Ground route (red)</span>
      <span><span class="ld" style="background:#22c55e"></span>Drive to/from helipad (green)</span>
      <span><span class="ld" style="background:#00d4ff"></span>Helicopter flight (cyan dashed)</span>
    </div>
  </div>
  <div id="route-info">
    <strong id="ri-title"></strong>&nbsp;
    <span id="ri-dist"></span> &middot; <span id="ri-dur"></span>
    <button onclick="clearHeli()" style="margin-left:10px;background:#eee;border:none;border-radius:3px;padding:2px 8px;cursor:pointer;font-size:11px">&times; Clear</button>
  </div>
</div>
<div id="aerial-bar">
  <span style="color:#475569;font-size:12px">Click &#x1F9ED; then pick two points to compare ground vs aerial routes</span>
</div>
<script>
var RANGE_KM = 300 * 1.852;
var SPEED_HELI_KMH = 250;
var SPEED_WALK_KMH = 5;

// ── map ───────────────────────────────────────────────────────────────────────
var osmDay = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors', maxZoom: 19
});
var esriSat = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {attribution: 'Tiles &copy; Esri', maxZoom: 20, maxNativeZoom: 19}
);
var map = L.map('map', {center: [40.75, -73.5], zoom: 8, layers: [osmDay]});
L.control.scale({metric: true, imperial: true}).addTo(map);

// ── business POIs (Miles Urban destinations) ──────────────────────────────────
var poiData = [
  {lat:40.7589, lng:-73.9851, name:'Midtown Manhattan',              cat:'biz'},
  {lat:40.7127, lng:-74.0059, name:'Financial District (Wall St)',   cat:'biz'},
  {lat:40.7504, lng:-73.9967, name:'Hudson Yards',                   cat:'biz'},
  {lat:40.7531, lng:-73.9772, name:'Grand Central / Park Ave',       cat:'biz'},
  {lat:40.7580, lng:-73.9855, name:'Rockefeller Center',             cat:'biz'},
  {lat:40.7282, lng:-74.0776, name:'Jersey City Financial Center',   cat:'biz'},
  {lat:40.7357, lng:-74.1724, name:'Newark, NJ',                     cat:'biz'},
  {lat:40.7456, lng:-74.3204, name:'Short Hills, NJ (Corp. Park)',   cat:'biz'},
  {lat:41.0253, lng:-73.6282, name:'Greenwich, CT',                  cat:'biz'},
  {lat:41.0534, lng:-73.5387, name:'Stamford, CT',                   cat:'biz'},
  {lat:41.1220, lng:-73.7949, name:'White Plains, NY',               cat:'biz'},
  {lat:41.0662, lng:-73.8987, name:'Tarrytown, NY (Regeneron etc.)', cat:'biz'},
  {lat:40.7606, lng:-73.8296, name:'LaGuardia Airport (LGA)',        cat:'airport'},
  {lat:40.6413, lng:-73.7781, name:'JFK Airport',                    cat:'airport'},
  {lat:40.6895, lng:-74.1745, name:'Newark Liberty (EWR)',           cat:'airport'},
  {lat:41.0673, lng:-73.7076, name:'Westchester Airport (HPN)',      cat:'airport'},
  {lat:40.7060, lng:-74.0099, name:'Downtown Manhattan Heliport',    cat:'heliport'},
  {lat:40.7422, lng:-73.9750, name:'East 34th St Heliport',          cat:'heliport'},
];

function poiIcon(cat) {
  var bg  = cat==='airport' ? '#5b21b6' : cat==='heliport' ? '#0369a1' : '#1e3a8a';
  var sym = cat==='airport' ? '&#x2708;' : cat==='heliport' ? 'H' : '&#x1F3E2;';
  return L.divIcon({
    html: '<div style="background:'+bg+';color:#fff;font-size:11px;font-weight:600;'+
          'padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,.25);'+
          'white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,.4)">'+sym+' </div>',
    className: '', iconAnchor: [0, 0]
  });
}
var poiLayer = L.layerGroup();
poiData.forEach(function(p) {
  L.marker([p.lat, p.lng], {icon: poiIcon(p.cat), title: p.name})
   .bindTooltip(p.name, {sticky:true, direction:'top', offset:[0,-6]})
   .addTo(poiLayer);
});

// ── helipads (FAA) ─────────────────────────────────────────────────────────────
var helipadData = __GEOJSON__;
var cr = L.canvas({padding: 0.5});
var helipadLayer = L.geoJSON(helipadData, {
  renderer: cr,
  pointToLayer: function(f, ll) {
    return L.circleMarker(ll, {
      radius:5, color:'#1565C0', fillColor:'#42A5F5', fillOpacity:0.8, weight:1.5
    });
  },
  onEachFeature: function(f, l) {
    var p = f.properties, name = p.NAME || p.IDENT || 'Helipad';
    l.bindTooltip(name, {sticky:true});
    l.bindPopup(
      '<b>'+name+'</b><br>IDENT: '+(p.IDENT||'&mdash;')+'<br>'+
      (p.STATE ? 'State: '+p.STATE+(p.SERVCITY?' &middot; '+p.SERVCITY:'')+'<br>' : '')+
      (p.OPERSTATUS ? 'Status: '+p.OPERSTATUS+'<br>' : '')+
      (p.ELEVATION ? 'Elev: '+p.ELEVATION+' ft' : ''),
      {maxWidth: 220}
    );
  }
});
var allPointLayers = [helipadLayer];

L.control.layers(
  {'Street Map': osmDay, 'Satellite (ESRI)': esriSat},
  {'Helipads (FAA)': helipadLayer, 'Business POIs': poiLayer},
  {collapsed: false}
).addTo(map);
helipadLayer.addTo(map);
poiLayer.addTo(map);

// ── utils ─────────────────────────────────────────────────────────────────────
function haversine(lat1, lon1, lat2, lon2) {
  var R=6371, dLat=(lat2-lat1)*Math.PI/180, dLon=(lon2-lon1)*Math.PI/180;
  var a=Math.sin(dLat/2)*Math.sin(dLat/2)+
    Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)*Math.sin(dLon/2);
  return R*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
}
function fmtDur(m) {
  if (m<60) return Math.round(m)+' min';
  return Math.floor(m/60)+'h'+(Math.round(m%60)>0?' '+Math.round(m%60)+'m':'');
}
function fmtDist(km) { return km<1 ? Math.round(km*1000)+' m' : km.toFixed(1)+' km'; }
function getAllHelipads() {
  var pts=[];
  allPointLayers.forEach(function(ly){
    ly.eachLayer(function(l){
      if(!l.feature) return;
      var ll=l.getLatLng?l.getLatLng():l.getBounds().getCenter();
      var p=l.feature.properties;
      pts.push({lat:ll.lat, lon:ll.lng, name:p.NAME||p.name||'Helipad'});
    });
  });
  return pts;
}
function nearestHelipad(lat,lng,pts) {
  var best=null,bd=Infinity;
  pts.forEach(function(p){var d=haversine(lat,lng,p.lat,p.lon);if(d<bd){bd=d;best=p;}});
  return best?{pad:best,dist:bd}:null;
}

// ── custom markers ────────────────────────────────────────────────────────────
function flagIcon(color, letter) {
  return L.divIcon({
    html:'<div style="position:relative;width:30px;height:40px">'+
      '<div style="position:absolute;bottom:0;left:4px;width:3px;height:28px;background:'+color+';border-radius:2px"></div>'+
      '<div style="position:absolute;top:0;left:4px;background:'+color+';color:#fff;font-weight:800;font-size:12px;'+
        'padding:2px 6px;border-radius:4px 4px 4px 0;box-shadow:0 2px 6px rgba(0,0,0,.4);'+
        'min-width:18px;text-align:center;border:1px solid rgba(255,255,255,.5)">'+letter+'</div>'+
    '</div>',
    className:'', iconSize:[30,40], iconAnchor:[6,40]
  });
}
function hIcon(color) {
  return L.divIcon({
    html:'<div style="background:'+color+';color:#fff;font-weight:900;font-size:13px;'+
      'width:28px;height:28px;border-radius:50%;border:3px solid #fff;'+
      'text-align:center;line-height:22px;'+
      'box-shadow:0 2px 8px rgba(0,0,0,.5)">H</div>',
    className:'', iconSize:[28,28], iconAnchor:[14,14]
  });
}

// ── state ─────────────────────────────────────────────────────────────────────
var routeLayers=[], mkA=null, mkB=null, ptA=null, ptB=null, clickState=0;

function clearHeli() {
  routeLayers.forEach(function(l){map.removeLayer(l);}); routeLayers=[];
  document.getElementById('route-info').style.display='none';
}
function clearMM() {
  routeLayers.forEach(function(l){map.removeLayer(l);}); routeLayers=[];
  if(mkA){map.removeLayer(mkA);mkA=null;} if(mkB){map.removeLayer(mkB);mkB=null;}
  ptA=null; ptB=null;
  document.getElementById('compare-panel').style.display='none';
  resetBar();
}
function clearAll() { clearMM(); clearHeli(); }

function resetBar() {
  var b=document.getElementById('aerial-bar'); b.className='';
  b.innerHTML='<span style="color:#475569;font-size:12px">Click &#x1F9ED; then pick two points to compare ground vs aerial routes</span>';
}
function setBar(groundMin, airMin) {
  var b=document.getElementById('aerial-bar');
  if (airMin===null) {
    b.className='no-save';
    b.innerHTML=
      '<div class="ab-seg"><div class="ab-lbl">Best ground</div><div class="ab-val ground">'+fmtDur(groundMin)+'</div></div>'+
      '<div class="ab-div"></div>'+
      '<div class="ab-neutral">No helipad route available for these points</div>';
    return;
  }
  var saving=groundMin-airMin;
  if (saving>0.5) {
    b.className='has-save';
    b.innerHTML=
      '<div class="ab-seg"><div class="ab-lbl">Best ground</div><div class="ab-val ground">'+fmtDur(groundMin)+'</div></div>'+
      '<div class="ab-div"></div>'+
      '<div class="ab-seg"><div class="ab-lbl">Aerial route</div><div class="ab-val air">'+fmtDur(airMin)+'</div></div>'+
      '<div class="ab-div"></div>'+
      '<div class="ab-badge">&#x2708; Aerial advantage: saves '+fmtDur(saving)+'</div>';
  } else {
    b.className='no-save';
    b.innerHTML=
      '<div class="ab-seg"><div class="ab-lbl">Best ground</div><div class="ab-val ground">'+fmtDur(groundMin)+'</div></div>'+
      '<div class="ab-div"></div>'+
      '<div class="ab-seg"><div class="ab-lbl">Aerial route</div><div class="ab-val dim">'+fmtDur(airMin)+'</div></div>'+
      '<div class="ab-div"></div>'+
      '<div class="ab-neutral">Ground is faster for this trip</div>';
  }
}

// ── OSRM ──────────────────────────────────────────────────────────────────────
async function osrmRoute(lat1,lng1,lat2,lng2,profile) {
  var url='https://router.project-osrm.org/route/v1/'+profile+'/'+
    lng1.toFixed(6)+','+lat1.toFixed(6)+';'+lng2.toFixed(6)+','+lat2.toFixed(6)+
    '?overview=full&geometries=geojson';
  try {
    var r=await (await fetch(url)).json();
    if(r.code==='Ok'&&r.routes&&r.routes.length)
      return {dist:r.routes[0].distance/1000, duration:r.routes[0].duration/60, geom:r.routes[0].geometry};
  } catch(e){}
  var d=haversine(lat1,lng1,lat2,lng2);
  return {dist:d, duration:d/30*60, geom:null};
}

// ── Multi-modal computation ────────────────────────────────────────────────────
async function computeMultiModal(pA, pB) {
  var sb=document.getElementById('status-bar');
  sb.innerHTML='&#x23F3; Computing routes&hellip;'; sb.style.display='block';

  var driveR=await osrmRoute(pA.lat,pA.lng,pB.lat,pB.lng,'driving');
  var driveDur=driveR.duration, taxiDur=driveDur*1.1, transitDur=driveDur*1.5;
  var walkDist=haversine(pA.lat,pA.lng,pB.lat,pB.lng), walkDur=walkDist/SPEED_WALK_KMH*60;

  var allPts=getAllHelipads(), nearA=nearestHelipad(pA.lat,pA.lng,allPts), nearB=nearestHelipad(pB.lat,pB.lng,allPts);
  var heli=null;
  if(nearA&&nearB) {
    var hd=haversine(nearA.pad.lat,nearA.pad.lon,nearB.pad.lat,nearB.pad.lon);
    if(hd>0.1&&hd<=RANGE_KM) {
      var d2a=await osrmRoute(pA.lat,pA.lng,nearA.pad.lat,nearA.pad.lon,'driving');
      var d2b=await osrmRoute(nearB.pad.lat,nearB.pad.lon,pB.lat,pB.lng,'driving');
      var hDur=hd/SPEED_HELI_KMH*60;
      heli={dur:d2a.duration+hDur+d2b.duration, dist:d2a.dist+hd+d2b.dist,
            hd:hd, hDur:hDur, padA:nearA.pad, padB:nearB.pad, gA:d2a.geom, gB:d2b.geom};
    }
  }

  // best ground time
  var groundTimes=[driveDur,taxiDur]; if(walkDur<90) groundTimes.push(walkDur);
  var bestGround=Math.min.apply(null,groundTimes);

  // ── draw RED: driving route ───────────────────────────────────────────────
  if(driveR.geom)
    routeLayers.push(L.geoJSON(driveR.geom,{style:{color:'#ef4444',weight:5,opacity:0.85}}).addTo(map));
  else
    routeLayers.push(L.polyline([[pA.lat,pA.lng],[pB.lat,pB.lng]],{color:'#ef4444',weight:5,opacity:0.7,dashArray:'6 4'}).addTo(map));

  // ── draw GREEN+CYAN: aerial route ─────────────────────────────────────────
  if(heli) {
    if(heli.gA) routeLayers.push(L.geoJSON(heli.gA,{style:{color:'#22c55e',weight:4,opacity:0.9}}).addTo(map));
    routeLayers.push(L.polyline([[heli.padA.lat,heli.padA.lon],[heli.padB.lat,heli.padB.lon]],
      {color:'#00d4ff',weight:3,dashArray:'10 6',opacity:0.95}).addTo(map));
    if(heli.gB) routeLayers.push(L.geoJSON(heli.gB,{style:{color:'#22c55e',weight:4,opacity:0.9}}).addTo(map));
    routeLayers.push(L.marker([heli.padA.lat,heli.padA.lon],{icon:hIcon('#00b8d9'),zIndexOffset:500})
      .bindTooltip('Take-off: '+heli.padA.name,{direction:'top'}).addTo(map));
    routeLayers.push(L.marker([heli.padB.lat,heli.padB.lon],{icon:hIcon('#22c55e'),zIndexOffset:500})
      .bindTooltip('Landing: '+heli.padB.name,{direction:'top'}).addTo(map));
  }

  // ── fit bounds ────────────────────────────────────────────────────────────
  var pts=[pA,pB];
  if(heli){pts.push(L.latLng(heli.padA.lat,heli.padA.lon));pts.push(L.latLng(heli.padB.lat,heli.padB.lon));}
  map.fitBounds(L.latLngBounds(pts),{padding:[50,50]});

  // ── table ─────────────────────────────────────────────────────────────────
  var rows=[
    {name:'&#x1F6B6; Walking',         dur:walkDur,    dist:walkDist,    air:'&mdash;', ground:true},
    {name:'&#x1F697; Car',             dur:driveDur,   dist:driveR.dist, air:'&mdash;', ground:true},
    {name:'&#x1F695; Taxi',            dur:taxiDur,    dist:driveR.dist, air:'&mdash;', ground:true, note:'est.'},
    {name:'&#x1F68C; Transit/Subway',  dur:transitDur, dist:driveR.dist, air:'&mdash;', ground:true, note:'est.'},
  ];
  if(heli) rows.push({
    name:'&#x1F697;&#x2708; Car + Heli + Car',
    dur:heli.dur, dist:heli.dist,
    air:fmtDist(heli.hd)+' &middot; '+fmtDur(heli.hDur), ground:false
  });

  var fastestDur=rows.reduce(function(a,b){return a.dur<b.dur?a:b;}).dur;
  var fastestGround=rows.filter(function(r){return r.ground;}).reduce(function(a,b){return a.dur<b.dur?a:b;}).dur;

  var tbody=document.getElementById('cp-tbody'); tbody.innerHTML='';
  rows.forEach(function(r){
    var tr=document.createElement('tr');
    tr.className=r.ground?(Math.abs(r.dur-fastestGround)<0.1?'gr-best':''):'ai-best';
    var tags='';
    if(Math.abs(r.dur-fastestDur)<0.1) tags+='<span class="tag tag-f">Fastest</span>';
    if(!r.ground) tags+='<span class="tag tag-a">Aerial</span>';
    tr.innerHTML='<td>'+r.name+tags+(r.note?'<span style="color:#475569;font-size:10px"> '+r.note+'</span>':'')+'</td>'+
      '<td>'+fmtDist(r.dist)+'</td><td>'+fmtDur(r.dur)+'</td><td>'+r.air+'</td>';
    tbody.appendChild(tr);
  });
  document.getElementById('cp-from-to').textContent='A ('+pA.lat.toFixed(4)+', '+pA.lng.toFixed(4)+')  →  B ('+pB.lat.toFixed(4)+', '+pB.lng.toFixed(4)+')';
  document.getElementById('compare-panel').style.display='block';
  sb.style.display='none';
  setBar(fastestGround, heli?heli.dur:null);
}

// ── Helipad-to-helipad Dijkstra ───────────────────────────────────────────────
function findRoute(start,end,allPts) {
  var nodes=[{lat:start.lat,lon:start.lng,name:'Origin'}].concat(allPts).concat([{lat:end.lat,lon:end.lng,name:'Destination'}]);
  var N=nodes.length, dst=N-1, dist=[], prev=[];
  for(var i=0;i<N;i++){dist.push(Infinity);prev.push(-1);}
  dist[0]=0; var heap=[[0,0]];
  while(heap.length){
    heap.sort(function(a,b){return a[0]-b[0];});
    var top=heap.shift(),cost=top[0],u=top[1];
    if(cost>dist[u])continue; if(u===dst)break;
    for(var v=0;v<N;v++){
      if(v===u)continue;
      var d=haversine(nodes[u].lat,nodes[u].lon,nodes[v].lat,nodes[v].lon);
      if(d>RANGE_KM)continue;
      var nc=dist[u]+d; if(nc<dist[v]){dist[v]=nc;prev[v]=u;heap.push([nc,v]);}
    }
  }
  if(dist[dst]===Infinity)return null;
  var path=[],cur=dst; while(cur!==-1){path.unshift(nodes[cur]);cur=prev[cur];}
  return {path:path, dist:dist[dst], dur:dist[dst]/SPEED_HELI_KMH*60};
}
function drawHeliRoute(result) {
  clearHeli();
  if(!result||result.path.length<2){alert('No helipad route found within range.');return;}
  var ll=result.path.map(function(p){return[p.lat,p.lon];});
  routeLayers.push(L.polyline(ll,{color:'#1565C0',weight:3,dashArray:'8 4',opacity:0.9}).addTo(map));
  for(var i=1;i<result.path.length-1;i++)
    routeLayers.push(L.marker([result.path[i].lat,result.path[i].lon],
      {icon:hIcon('#1565C0'),zIndexOffset:500}).bindTooltip(result.path[i].name,{direction:'top'}).addTo(map));
  map.fitBounds(L.latLngBounds(ll),{padding:[40,40]});
  document.getElementById('ri-title').textContent='Helipad Route ('+result.path.length+' waypoints)';
  document.getElementById('ri-dist').textContent=fmtDist(result.dist);
  document.getElementById('ri-dur').textContent=fmtDur(result.dur);
  document.getElementById('route-info').style.display='block';
}

// ── controls ──────────────────────────────────────────────────────────────────
var HeliControl=L.Control.extend({
  options:{position:'topright'},
  onAdd:function(){
    var btn=L.DomUtil.create('button','ctrl-btn');
    btn.innerHTML='&#x1F681;'; btn.title='Helipad-to-helipad routing (click 2 points)';
    L.DomEvent.on(btn,'click',function(e){
      L.DomEvent.stopPropagation(e); clearAll(); clickState=1;
      var sb=document.getElementById('status-bar');
      sb.innerHTML='&#x1F4CD; Click <b>Origin</b> on the map'; sb.style.display='block';
    }); return btn;
  }
});
var MMControl=L.Control.extend({
  options:{position:'topright'},
  onAdd:function(){
    var btn=L.DomUtil.create('button','ctrl-btn');
    btn.innerHTML='&#x1F9ED;'; btn.title='Multi-modal comparison (click 2 points)';
    L.DomEvent.on(btn,'click',function(e){
      L.DomEvent.stopPropagation(e); clearAll(); clickState=10;
      var sb=document.getElementById('status-bar');
      sb.innerHTML='&#x1F4CD; Click <b>Origin (A)</b> on the map'; sb.style.display='block';
    }); return btn;
  }
});
new HeliControl().addTo(map);
new MMControl().addTo(map);

// ── click handler ─────────────────────────────────────────────────────────────
map.on('click', async function(e){
  var sb=document.getElementById('status-bar');
  if(clickState===1){
    ptA=e.latlng; if(mkA)map.removeLayer(mkA);
    mkA=L.marker(ptA,{icon:flagIcon('#ef4444','A'),zIndexOffset:1000}).addTo(map);
    sb.innerHTML='&#x1F4CD; Click <b>Destination</b> on the map'; clickState=2;
  } else if(clickState===2){
    ptB=e.latlng; if(mkB)map.removeLayer(mkB);
    mkB=L.marker(ptB,{icon:flagIcon('#1565C0','B'),zIndexOffset:1000}).addTo(map);
    sb.style.display='none'; clickState=0;
    drawHeliRoute(findRoute(ptA,ptB,getAllHelipads()));
  } else if(clickState===10){
    ptA=e.latlng; if(mkA)map.removeLayer(mkA);
    mkA=L.marker(ptA,{icon:flagIcon('#ef4444','A'),zIndexOffset:1000}).addTo(map);
    sb.innerHTML='&#x1F4CD; Click <b>Destination (B)</b> on the map'; clickState=11;
  } else if(clickState===11){
    ptB=e.latlng; if(mkB)map.removeLayer(mkB);
    mkB=L.marker(ptB,{icon:flagIcon('#1565C0','B'),zIndexOffset:1000}).addTo(map);
    clickState=0; await computeMultiModal(ptA,ptB);
  }
});
</script>
</body>
</html>"""

src = APP.read_text(encoding='utf-8')

START_MARKER = '_ROUTING_HTML_TEMPLATE = """'
# Find where template ends: the """ on its own followed by blank line + def
import re
m = re.search(r'_ROUTING_HTML_TEMPLATE = """.*?"""', src, re.DOTALL)
if not m:
    raise RuntimeError("Could not find _ROUTING_HTML_TEMPLATE block")

OLD_BLOCK = m.group(0)
NEW_BLOCK = START_MARKER + NEW_TEMPLATE + '"""'

src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)

# Also update height
src = src.replace(
    'components.html(build_routing_html(faa_raw), height=620, scrolling=False)',
    'components.html(build_routing_html(faa_raw), height=650, scrolling=False)',
)

APP.write_text(src, encoding='utf-8')
print(f"Written: {len(src.splitlines())} lines")

import ast
ast.parse(src)
print("Syntax OK")
