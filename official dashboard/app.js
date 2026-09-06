const villages = [
  { village:"Sialsuk",       slope:0.82, rainfall:0.75, population:152, access_score:0.40, land_change:0.146 },
  { village:"Zohmun",        slope:0.55, rainfall:0.60, population:210, access_score:0.60, land_change:0.0 },
  { village:"Thenzawl",      slope:0.48, rainfall:0.65, population:95,  access_score:0.50, land_change:0.0 },
  { village:"Reiek",         slope:0.20, rainfall:0.30, population:180, access_score:0.80, land_change:0.0 },
  { village:"Aizawl_Rural1", slope:0.70, rainfall:0.55, population:300, access_score:0.30, land_change:0.0 },
  { village:"Aizawl_Rural2", slope:0.35, rainfall:0.40, population:120, access_score:0.70, land_change:0.0 },
  { village:"Lungdai",       slope:0.65, rainfall:0.70, population:88,  access_score:0.45, land_change:0.0 },
];

function scoreAll(rows){
  const maxPop = Math.max(...rows.map(r=>r.population));
  const scored = rows.map(r=>{
    const hazard_score = 0.40*r.slope + 0.35*r.rainfall + 0.25*r.land_change;
    const population_norm = r.population / maxPop;
    const access_factor = 1 - r.access_score;
    const relocation_score = 0.85*hazard_score + 0.10*population_norm + 0.05*access_factor;
    let recommended_action;
    if (hazard_score > 0.75) recommended_action = "Immediate Verification";
    else if (hazard_score > 0.55) recommended_action = "Relocation Assessment";
    else if (hazard_score > 0.35) recommended_action = "Enhanced Monitoring";
    else recommended_action = "Routine Monitoring";
    if (r.population > 250 && recommended_action === "Routine Monitoring") recommended_action = "Enhanced Monitoring";
    return {...r, hazard_score, population_norm, access_factor, relocation_score, recommended_action,
      hazard_pct:{
        hazard: 0.85*hazard_score,
        pop: 0.10*population_norm,
        access: 0.05*access_factor
      }
    };
  });
  const idx = 0.75*(rows.length-1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  const popSortedAsc = [...rows.map(r=>r.population)].sort((a,b)=>a-b);
  const threshold = popSortedAsc[lo] + (popSortedAsc[hi]-popSortedAsc[lo])*(idx-lo);
  scored.forEach(r => r.high_population_flag = r.population > threshold);
  scored.sort((a,b)=>b.relocation_score - a.relocation_score);
  return scored;
}

function riskTier(hazard_score){
  if (hazard_score > 0.55) return "High";
  if (hazard_score > 0.35) return "Medium";
  return "Low";
}

function riskIcon(tier){ return tier==="High" ? "!" : tier==="Medium" ? "•" : "✓"; }

const data = scoreAll(villages);
let activeVillage = data[0].village;

function renderStats(){
  const highRisk = data.filter(d=>riskTier(d.hazard_score)==="High").length;
  const needsAction = data.filter(d=>["Immediate Verification","Relocation Assessment"].includes(d.recommended_action)).length;
  const highPop = data.filter(d=>d.high_population_flag).length;
  const landChange = data.filter(d=>d.land_change > 0).length;

  const cards = [
    {label:"High-risk villages", value:highRisk, icon:"⚠️", cls:"red"},
    {label:"Need verification / assessment", value:needsAction, icon:"🔍", cls:"amber"},
    {label:"High population pressure", value:highPop, icon:"👥", cls:"teal"},
    {label:"Land-change alerts", value:landChange, icon:"🏞️", cls:"red"},
  ];
  document.getElementById("statsGrid").innerHTML = cards.map(c=>`
    <div class="stat-card">
      <div class="stat-icon ${c.cls}">${c.icon}</div>
      <div>
        <div class="stat-label">${c.label}</div>
        <div class="stat-value">${c.value}</div>
      </div>
    </div>
  `).join("");
}

function renderVillageGrid(){
  document.getElementById("villageGrid").innerHTML = data.map(d=>{
    const tier = riskTier(d.hazard_score);
    const active = d.village === activeVillage ? "active" : "";
    return `
      <div class="village-node ${active}" onclick="selectVillage('${d.village}')">
        <div class="dot ${tier}">${riskIcon(tier)}</div>
        <div class="vname">${d.village.replace("_"," ")}</div>
      </div>
    `;
  }).join("");
}

function renderWorklist(){
  document.getElementById("worklistBody").innerHTML = data.map(d=>{
    const tier = riskTier(d.hazard_score);
    const active = d.village === activeVillage ? "active" : "";
    return `
      <tr class="row ${active}" onclick="selectVillage('${d.village}')">
        <td>${d.village.replace("_"," ")}</td>
        <td><span class="badge ${tier}">${tier}</span></td>
        <td>${d.recommended_action}</td>
        <td class="pop-flag">${d.high_population_flag ? "👥" : "—"}</td>
      </tr>
    `;
  }).join("");
}

function renderDetail(){
  const d = data.find(x=>x.village===activeVillage);
  const tier = riskTier(d.hazard_score);
  document.getElementById("detailPanel").innerHTML = `
    <div class="kicker">Village Detail</div>
    <div class="detail-header">
      <div class="detail-village">${d.village.replace("_"," ")}</div>
      <span class="badge ${tier}">${tier} risk</span>
    </div>
    <p style="color:var(--muted);font-size:12.5px;margin:0 0 6px;">Why this score</p>
    <div class="why-bar">
      <div class="why-seg hazard" style="width:${(d.hazard_pct.hazard*100).toFixed(1)}%"></div>
      <div class="why-seg pop" style="width:${(d.hazard_pct.pop*100).toFixed(1)}%"></div>
      <div class="why-seg access" style="width:${(d.hazard_pct.access*100).toFixed(1)}%"></div>
    </div>
    <div class="why-legend">
      <span><i style="background:var(--navy-soft)"></i> Hazard (slope, rainfall, land-change)</span>
      <span><i style="background:#4C9BB5"></i> Population</span>
      <span><i style="background:var(--amber)"></i> Access difficulty</span>
    </div>

    <div class="action-chip ${tier}">📌 ${d.recommended_action}</div>

    <div class="detail-row"><span class="k">Hazard score</span><span class="v">${d.hazard_score.toFixed(3)}</span></div>
    <div class="detail-row"><span class="k">Relocation score</span><span class="v">${d.relocation_score.toFixed(3)}</span></div>
    <div class="detail-row"><span class="k">Population</span><span class="v">${d.population}${d.high_population_flag ? " · high pressure 👥" : ""}</span></div>
    <div class="detail-row"><span class="k">Access to safe zone</span><span class="v">${(d.access_score*100).toFixed(0)}% easy</span></div>
    <div class="detail-row"><span class="k">Land change detected</span><span class="v">${d.land_change > 0 ? (d.land_change*100).toFixed(1)+"%" : "None"}</span></div>
  `;
}

function selectVillage(name){
  activeVillage = name;
  renderVillageGrid();
  renderWorklist();
  renderDetail();
}

function setUpdatedTime(){
  const now = new Date();
  document.getElementById("updatedText").textContent =
    "Updated " + now.toLocaleDateString(undefined,{day:"2-digit",month:"short",year:"numeric"});
}

renderStats();
renderVillageGrid();
renderWorklist();
renderDetail();
setUpdatedTime();