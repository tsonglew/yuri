export const actions = ['hold', 'defend', 'attack', 'retreat', 'scout'];
export const labels = {hold:'集结', defend:'防守', attack:'进攻', retreat:'撤退', scout:'侦察'};
export const scenarios = {
  advantage: {army_supply:32, enemy_army_supply:18, base_threat:.15, army_health:.9, intel_age_seconds:12},
  defense: {army_supply:24, enemy_army_supply:22, base_threat:.8, army_health:.8, intel_age_seconds:5},
  retreat: {army_supply:18, enemy_army_supply:32, base_threat:.4, army_health:.25, intel_age_seconds:8},
  scout: {army_supply:28, enemy_army_supply:14, base_threat:.1, army_health:.95, intel_age_seconds:70},
  opening: {army_supply:0, enemy_army_supply:0, base_threat:0, army_health:1, intel_age_seconds:15},
};
export function decide(o) {
  let action, reason, ruleIndex;
  if(o.army_supply<=0) [action,reason,ruleIndex]=['hold','没有可调动的战斗部队，优先完成运营与集结。',0];
  else if(o.army_health<.35) [action,reason,ruleIndex]=['retreat','部队健康低于 35%，优先撤回基地保存兵力。',1];
  else if(o.base_threat>=.6) [action,reason,ruleIndex]=['defend','基地威胁达到 60%，优先回防。',2];
  else if(o.intel_age_seconds>=45) [action,reason,ruleIndex]=['scout','情报超过 45 秒未更新，先确认敌方状态。',3];
  else if(o.army_supply>=12 && o.army_supply>=1.3*Math.max(o.enemy_army_supply,1)) [action,reason,ruleIndex]=['attack','兵力达到 12 人口，且至少为已知敌军的 1.3 倍，选择推进。',4];
  else [action,reason,ruleIndex]=['hold','暂未满足进攻条件，在基地附近集结并继续运营。',5];
  return {schema_version:1, action, requested_policy:'rules', executed_policy:'rules', reason, confidence:null, fallback_reason:null, ruleIndex};
}
export function validateRecord(record) {
  if(!record || record.schema_version!==1 || !record.observation || !record.decision) throw Error('需要 schema_version = 1 的 Yuri 决策记录。');
  const o=record.observation,d=record.decision;
  if(o.schema_version!==1 || d.schema_version!==1) throw Error('局面或决策的协议版本不受支持。');
  for(const key of ['army_supply','enemy_army_supply','base_threat','army_health','intel_age_seconds']) {
    if(typeof o[key]!=='number'||!Number.isFinite(o[key])||o[key]<0) throw Error(`局面字段 ${key} 必须是有限的非负数。`);
  }
  if(o.base_threat>1||o.army_health>1) throw Error('威胁与健康必须处于 0 到 1 之间。');
  if(!actions.includes(d.action)) throw Error('未知动作。');
  if(typeof d.requested_policy!=='string'||typeof d.executed_policy!=='string'||typeof d.reason!=='string') throw Error('缺少策略来源或原因。');
  if(d.confidence!=null && (typeof d.confidence!=='number'||!Number.isFinite(d.confidence)||d.confidence<0||d.confidence>1)) throw Error('置信度必须处于 0 到 1 之间。');
  if(d.latency_ms!=null && (typeof d.latency_ms!=='number'||!Number.isFinite(d.latency_ms)||d.latency_ms<0)) throw Error('耗时必须是有限非负数。');
  if(d.executed_policy==='laya' && d.confidence==null) throw Error('Laya 结果缺少置信度。');
  return record;
}
export function parseReplay(text) {
  let data;
  try {data=JSON.parse(text);} catch {data=text.trim().split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));}
  const records=Array.isArray(data)?data:[data];
  if(!records.length||records.length>10000) throw Error('请导入 1 至 10000 条记录。');
  return records.map(validateRecord);
}
