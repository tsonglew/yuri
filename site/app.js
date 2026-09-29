import {decide,labels,scenarios,parseReplay} from './policy.mjs';
const $=id=>document.getElementById(id);
const fields={army:'army_supply',enemy:'enemy_army_supply',health:'army_health',threat:'base_threat',intel:'intel_age_seconds'};
let records=[],frame=0,current;
function readState(){const o={schema_version:1,minerals:550,gas:250,supply_left:18,game_time_seconds:420,enemy_visible:true};for(const [id,key] of Object.entries(fields))o[key]=Number($(id).value)/(['health','threat'].includes(id)?100:1);return o;}
function setState(o){for(const [id,key] of Object.entries(fields)){const v=o[key]*(['health','threat'].includes(id)?100:1);$(id).max=Math.max(id==='intel'?120:100,Math.ceil(v));$(id).value=v;}}
function dots(id,count,cx,cy,color){const group=$(id);group.replaceChildren();for(let i=0;i<Math.min(40,Math.ceil(count/2));i++){const circle=document.createElementNS('http://www.w3.org/2000/svg','circle');circle.setAttribute('cx',cx+(i%6)*12-(Math.min(Math.ceil(count/2),6)-1)*6);circle.setAttribute('cy',cy+Math.floor(i/6)*12);circle.setAttribute('r',4);circle.setAttribute('fill',color);group.append(circle);}}
function render(record){current=record;const o=record.observation,d=record.decision,replay=records.length>0;
  for(const [id,key] of Object.entries(fields)){const ratio=['health','threat'].includes(id);$(id+'-value').textContent=`${Math.round(o[key]*(ratio?100:1)*10)/10}${ratio?'%':id==='intel'?' 秒':''}`;$(id).disabled=replay;}
  $('scenario').disabled=replay;$('reset').disabled=replay;
  $('decision-action').textContent=labels[d.action];$('decision-reason').textContent=d.reason;
  $('requested').textContent=d.requested_policy==='rules'?'规则基线':d.requested_policy;
  $('executed').textContent=d.executed_policy==='rules'?'规则基线':d.executed_policy;
  $('confidence').textContent=d.confidence==null?'不适用':`${(d.confidence*100).toFixed(1)}%`;
  $('latency').textContent=d.latency_ms==null?'未记录':`${d.latency_ms.toFixed(2)} ms`;
  $('proposed').textContent=d.proposed_action?labels[d.proposed_action]:'与执行动作一致';
  $('revision').textContent=d.model_revision|| (d.model?'未记录':'不适用');
  $('source-badge').textContent=replay?`REPLAY / ${d.executed_policy.toUpperCase()}`:'RULES';
  $('decision-kicker').textContent=replay?'导入记录 · 原始结果':'当前建议';
  $('flow-policy').textContent=d.executed_policy==='laya'?'Laya 策略':'规则策略';$('flow-detail').textContent=replay?'记录回放':'优先级判断';
  $('fallback').hidden=!d.fallback_reason;$('fallback').textContent=d.fallback_reason?`规则回退：${d.fallback_reason}`:'';
  const index=replay?null:d.ruleIndex;document.querySelectorAll('#rule-list li').forEach((li,i)=>li.classList.toggle('selected',i===index));
  const routes={attack:'M210 275Q330 130 490 115',defend:'M210 275Q145 220 114 300',retreat:'M210 275Q190 340 130 340',scout:'M210 275Q400 340 530 150',hold:'M210 275Q170 245 155 300'};
  $('route').setAttribute('d',routes[d.action]);$('route').style.opacity=o.army_supply>0?'1':'0';
  $('threat-ring').style.opacity=String(o.base_threat);dots('allies',o.army_supply,210,270,'#55c9f2');dots('enemies',o.enemy_army_supply,o.base_threat>=.6?160:450,o.base_threat>=.6?335:145,'#dc8c3c');
  $('map-action').textContent={attack:'推进至敌方区域',defend:'回防己方基地',retreat:'撤回安全区域',scout:'更新敌方情报',hold:'基地附近集结'}[d.action];
  $('map-ratio').textContent=o.enemy_army_supply>0?`对最后可见兵力 ${(o.army_supply/o.enemy_army_supply).toFixed(2)} : 1`:'无已知敌军兵力';
  $('exit-replay').hidden=!replay;$('timeline').hidden=!replay;
  if(replay){$('frame').max=records.length-1;$('frame').value=frame;$('frame-label').textContent=`${frame+1} / ${records.length}`;$('previous').disabled=frame===0;$('next').disabled=frame===records.length-1;}
}
function live(){const o=readState(),start=performance.now(),d=decide(o);d.latency_ms=performance.now()-start;render({schema_version:1,source:'browser-rules',observation:o,decision:d});}
function scenario(){setState(scenarios[$('scenario').value]);live();}
for(const id of Object.keys(fields))$(id).addEventListener('input',live);
$('scenario').addEventListener('change',scenario);$('reset').addEventListener('click',()=>{$('scenario').value='advantage';scenario();});
$('export').addEventListener('click',()=>{const a=document.createElement('a');const url=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:'application/json'}));a.href=url;a.download='yuri-decision.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('replay-file').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>10*1024*1024)throw Error('文件超过 10 MB，请截取部分记录。');const parsed=parseReplay(await file.text());records=parsed;frame=0;showFrame();$('file-status').textContent=`已载入 ${records.length} 条记录。结果来自导入文件，未由本站重新推理或验证真实性。${records.some(r=>r.inference_observation)?' 异步推理使用先前局面；地图显示执行时局面。':''}`;}catch(error){$('file-status').textContent=`导入失败：${error.message}`;}finally{e.target.value='';}});
function showFrame(){setState(records[frame].observation);render(records[frame]);}
$('frame').addEventListener('input',()=>{frame=Number($('frame').value);showFrame();});$('previous').addEventListener('click',()=>{frame=Math.max(0,frame-1);showFrame();});$('next').addEventListener('click',()=>{frame=Math.min(records.length-1,frame+1);showFrame();});
$('exit-replay').addEventListener('click',()=>{records=[];$('file-status').textContent='';scenario();});
scenario();
