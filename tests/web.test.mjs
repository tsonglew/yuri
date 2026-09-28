import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {decide,parseReplay,validateRecord,scenarios} from '../site/policy.mjs';
const cases=JSON.parse(readFileSync(new URL('./fixtures/policy-cases.json',import.meta.url),'utf8'));
for(const entry of cases)test(`rule contract: ${entry.name}`,()=>assert.equal(decide(entry.observation).action,entry.action));
const record={schema_version:1,source:'test-fixture',observation:{schema_version:1,...scenarios.advantage},decision:{...decide(scenarios.advantage),latency_ms:1}};
test('single, array and JSONL replay',()=>{
  assert.equal(parseReplay(JSON.stringify(record)).length,1);
  assert.equal(parseReplay(JSON.stringify([record,record])).length,2);
  assert.equal(parseReplay(JSON.stringify(record)+'\n'+JSON.stringify(record)).length,2);
});
test('reject malformed and misleading model records',()=>{
  assert.throws(()=>parseReplay('[]'));
  assert.throws(()=>parseReplay('{broken'));
  assert.throws(()=>validateRecord({...record,schema_version:2}));
  assert.throws(()=>validateRecord({...record,observation:{...record.observation,army_health:2}}));
  assert.throws(()=>validateRecord({...record,decision:{...record.decision,action:'cheat'}}));
  assert.throws(()=>validateRecord({...record,decision:{...record.decision,executed_policy:'laya',confidence:null}}));
  assert.throws(()=>validateRecord({...record,decision:{...record.decision,confidence:NaN}}));
});
