#!/usr/bin/env bash
# End-to-end test of bizen-ctl against the mock llama-server. Usage: test/run-tests.sh
D="$(cd "$(dirname "$0")/.." && pwd)"
T=$(mktemp -d /tmp/bizentest.XXXX)
B=http://127.0.0.1:18990
touch "$T/model.gguf"
python3 - "$D/config.example.json" "$T/config.json" "$T" "$D" <<'EOF'
import json,sys
c=json.load(open(sys.argv[1])); T=sys.argv[3]; D=sys.argv[4]
c.update(api_keys=["k1"], admin_password="pw", llama_server=D+"/test/mock-llama-server.py", model_dirs=[T],
         listen_port=18990, internal_port=18991, idle_ttl_min=0.4)
for p in c["profiles"]: p["model"]=T+"/model.gguf"
c["kev"]={"managed":True,"dir":T,"model":"jaredpalmer/kev-0.8b","command":"python3 "+D+"/test/mock-kev.py {port} {model}","port":18992,"idle_ttl_min":0.3,"hf_home":"","gpus":"auto:1"}
c["router"]={"enabled":True,"alias":"auto","default_profile":"qwen36-1gpu","threshold":0.6,
  "routes":[{"profile":"qwen36-2gpu","description":"Writing, fixing or reviewing programming code"},
            {"profile":"qwen36-1gpu","description":"Friendly general chat and email writing"}]}
json.dump(c,open(sys.argv[2],"w"))
EOF
python3 -u "$D/bizenctl.py" --config "$T/config.json" > "$T/ctl.log" 2>&1 &
CTL=$!; sleep 1
j() { python3 -c "import sys,json;d=json.load(sys.stdin);$1"; }

echo "--- auth: no key / no login / bad login"
curl -s -o /dev/null -w "%{http_code} " $B/v1/models
curl -s -o /dev/null -w "%{http_code} " $B/api/status
curl -s -o /dev/null -w "%{http_code}\n" -X POST -d '{"password":"x"}' $B/api/login
curl -s -c "$T/jar" -X POST -d '{"password":"pw"}' $B/api/login; echo
echo "--- status (stopped)"
curl -s -b "$T/jar" $B/api/status | j "print(d['server']['state'],'pin',d['auto_pin'],'node',d['auto_pin_node'])"
echo "--- cold proxied request"
time curl -s -H "Authorization: Bearer k1" $B/v1/chat/completions -d '{"messages":[]}'; echo
echo "--- streaming (timestamps should be ~0.3s apart)"
curl -sN -H "Authorization: Bearer k1" $B/v1/chat/completions -d '{"stream":true}' |
  while IFS= read -r l; do [ -n "$l" ] && echo "$(date +%S.%N | cut -c1-5) $l"; done
echo "--- test prompt"
curl -s -b "$T/jar" -X POST -d '{"prompt":"hi"}' $B/api/test; echo
echo "--- status (running)"
curl -s -b "$T/jar" $B/api/status | j "s=d['server'];print(s['state'],s['profile'],'gpus',s['gpus'],'tps',s['last_tps'],s['last_prompt_tps'],'unload_in',s['unload_in_s'])"
echo "--- preview"
curl -s -b "$T/jar" -X POST -d '{"name":"x","model":"/m.gguf","gpus":"auto:1","fit":false,"n_cpu_moe":32,"kv_quant":"q8_0"}' $B/api/preview | j "print(d['cmd'])"
echo "--- preview: per-profile llama_server override"
curl -s -b "$T/jar" -X POST -d '{"name":"x","model":"/m.gguf","gpus":"auto:1","llama_server":"/opt/other/llama-server"}' $B/api/preview | j "print('override ok' if '/opt/other/llama-server' in d['cmd'] else 'override MISSING')"
echo "--- invalid config rejected"
curl -s -b "$T/jar" -X PUT -d '{"profiles":[],"api_keys":["a"],"admin_password":"p","active_profile":"x"}' $B/api/config; echo
echo "--- valid config change flags restart_needed"
curl -s -b "$T/jar" $B/api/config | j "d['profiles'][0]['ctx']=8192;print(json.dumps(d))" > "$T/new.json"
curl -s -b "$T/jar" -X PUT --data @"$T/new.json" $B/api/config | j "print(d['ok'],'restart_needed=',d['status']['restart_needed'])"
echo "--- router: /v1/models lists profiles + alias"
curl -s -H "Authorization: Bearer k1" $B/v1/models | j "print([m['id'] for m in d['data']])"
echo "--- router: /v1/route (cold Kev start)"
curl -s -H "Authorization: Bearer k1" $B/v1/route -d '{"prompt":"Fix this programming code that crashes"}' | j "print(d['profile'],round(d['probability'],2),'would_switch=',d['would_switch'])"
echo "--- router: auto -> code goes to qwen36-2gpu"
curl -s -D "$T/h1" -o /dev/null -H "Authorization: Bearer k1" $B/v1/chat/completions -d '{"model":"auto","messages":[{"role":"user","content":"Review my programming code please"}]}'
grep -i "x-bizen-route" "$T/h1" | tr -d '\r'
curl -s -b "$T/jar" $B/api/status | j "print('serving',d['server']['profile'],'routed',d['server']['routed'],'restart_needed',d['server']['restart_needed'])"
echo "--- router: auto -> email goes to qwen36-1gpu (switch)"
curl -s -D "$T/h2" -o /dev/null -H "Authorization: Bearer k1" $B/v1/chat/completions -d '{"model":"auto","messages":[{"role":"user","content":"Write a friendly email to my advisor"}]}'
grep -i "x-bizen-route" "$T/h2" | tr -d '\r'
curl -s -b "$T/jar" $B/api/status | j "print('serving',d['server']['profile'],'routed',d['server']['routed'])"
echo "--- router: unsure -> default + fallback header"
curl -s -D "$T/h3" -o /dev/null -H "Authorization: Bearer k1" $B/v1/chat/completions -d '{"model":"auto","messages":[{"role":"user","content":"UNSURE what this is"}]}'
grep -i "x-bizen-route" "$T/h3" | tr -d '\r'
echo "--- router: plain model name is not routed"
curl -s -D "$T/h4" -o /dev/null -H "Authorization: Bearer k1" $B/v1/chat/completions -d '{"model":"whatever","messages":[{"role":"user","content":"Review my programming code"}]}'
grep -ic "x-bizen-route" "$T/h4"
echo "--- router: admin test run=true"
curl -s -b "$T/jar" -X POST -d '{"prompt":"Fix my programming code","run":true}' $B/api/route/test | j "print(d['profile'],d['decision']['profile'],round(d['decision']['probability'],2),d['text'])"
echo "--- router: kev status"
curl -s -b "$T/jar" $B/api/status | j "k=d['kev'];print(k['state'],'gpus',k['gpus'],'requests',k['requests'])"
echo "--- router: invalid route rejected"
curl -s -b "$T/jar" $B/api/config | j "d['router']['routes'][0]['profile']='nope';print(json.dumps(d))" > "$T/bad.json"
curl -s -b "$T/jar" -X PUT --data @"$T/bad.json" $B/api/config; echo
echo "--- kev model picker: presets, change model, restart flagged, restart applies"
curl -s -b "$T/jar" $B/api/kev/presets | j "print(len(d['presets']), d['presets'][0]['id'])"
curl -s -b "$T/jar" $B/api/status | j "k=d['kev'];print('before:',k['state'],k['running_model'],'restart_needed=',k['restart_needed'])"
curl -s -b "$T/jar" $B/api/config | j "d['kev']['model']='jaredpalmer/kev-4b';d['kev']['dtype']='fp16';d['kev']['cuda_graphs']='off';print(json.dumps(d))" > "$T/kevcfg.json"
curl -s -b "$T/jar" -X PUT --data @"$T/kevcfg.json" $B/api/config | j "print('saved',d['ok'])"
curl -s -b "$T/jar" $B/api/status | j "k=d['kev'];print('after save:',k['state'],k['running_model'],'restart_needed=',k['restart_needed'])"
curl -s -b "$T/jar" -X POST $B/api/kev/restart >/dev/null; sleep 5
curl -s -b "$T/jar" $B/api/status | j "k=d['kev'];print('after restart:',k['state'],k['running_model'],'restart_needed=',k['restart_needed'])"
curl -s -b "$T/jar" $B/api/kev/logs | j "print([l for l in d['lines'] if 'mock kev starting' in l][-1])"
curl -s -H "Authorization: Bearer k1" $B/v1/route -d '{"prompt":"Fix this programming code"}' | j "print('route still works:',d['profile'])"
echo "--- kev empty model rejected (no command override)"
curl -s -b "$T/jar" $B/api/config | j "d['kev']['model']='';d['kev']['command']='';print(json.dumps(d))" > "$T/kevbad.json"
curl -s -b "$T/jar" -X PUT --data @"$T/kevbad.json" $B/api/config; echo
FAILS=0
ok() { if [ "$2" = "1" ]; then echo "PASS $1"; else echo "FAIL $1"; FAILS=$((FAILS+1)); fi; }
echo "--- unit: meminfo / proc reads tolerate missing files, sampler ring buffer"
python3 "$D/test/test_metrics.py" > "$T/unit.log" 2>&1; UR=$?
grep -E "^(PASS|FAIL)" "$T/unit.log" | grep -c PASS | sed 's/^/unit checks passed: /'
grep FAIL "$T/unit.log" | head -5; tail -3 "$T/unit.log" | grep -i "traceback\|error"
ok "unit tests exit 0" $([ $UR = 0 ] && echo 1 || echo 0)
for U in test_loadmap test_gen test_placement; do
  python3 "$D/test/$U.py" > "$T/$U.log" 2>&1; UR=$?
  grep -c "^PASS" "$T/$U.log" | sed "s/^/$U checks passed: /"
  grep "^FAIL" "$T/$U.log" | head -5
  ok "$U exits 0" $([ $UR = 0 ] && echo 1 || echo 0)
done
echo "--- status.mem has the extended fields"
curl -s -b "$T/jar" $B/api/status | j "m=d['mem'];print('mem ok' if all(k in m for k in ('total_mib','available_mib','used_mib','free_mib','buffers_mib','cached_mib','swap_used_mib','nodes','child')) and m['used_mib']==m['total_mib']-m['available_mib'] else 'mem MISSING: '+str(sorted(m)))" | tee "$T/o"
ok "status.mem extended" $(grep -q "^mem ok" "$T/o" && echo 1 || echo 0)
echo "--- /api/metrics: auth, shape, since filter"
ok "metrics needs login" $([ "$(curl -s -o /dev/null -w %{http_code} $B/api/metrics)" = 401 ] && echo 1 || echo 0)
sleep 3
curl -s -b "$T/jar" "$B/api/metrics" > "$T/m1.json"
python3 - "$T/m1.json" > "$T/o" <<'EOF2'
import json,sys
d=json.load(open(sys.argv[1])); s=d["samples"]
need={"t","ram_used","nodes","gpus","cpu","rss"}
print("shape", all(need<=set(x) for x in s) and bool(s) and "cpu_ids" in d and "cpus_pinned" in d and "now" in d)
print("ordered", all(a["t"]<b["t"] for a,b in zip(s,s[1:])))
print("cpu len", len(set(len(x["cpu"]) for x in s))==1 and len(s[-1]["cpu"])==len(d["cpu_ids"]))
print("last", s[-1]["t"])
EOF2
cat "$T/o" | head -4
ok "metrics shape" $(grep -q "^shape True" "$T/o" && echo 1 || echo 0)
ok "metrics ordered" $(grep -q "^ordered True" "$T/o" && echo 1 || echo 0)
ok "metrics per-core length matches cpu_ids" $(grep -q "^cpu len True" "$T/o" && echo 1 || echo 0)
LAST=$(grep "^last" "$T/o" | cut -d' ' -f2)
sleep 2.5
curl -s -b "$T/jar" "$B/api/metrics?since=$LAST" | j "s=d['samples'];print('since',len(s)>=1 and all(x['t']>$LAST for x in s))" | tee "$T/o"
ok "metrics since returns only newer samples" $(grep -q "since True" "$T/o" && echo 1 || echo 0)
curl -s -b "$T/jar" "$B/api/metrics?since=9999999999" | j "print('empty',d['samples']==[])" | tee "$T/o"
ok "metrics since=future is empty" $(grep -q "empty True" "$T/o" && echo 1 || echo 0)
echo "--- /api/chat: auth, validation, streaming"
ok "chat needs login" $([ "$(curl -s -o /dev/null -w %{http_code} -X POST -d '{"messages":[{"role":"user","content":"hi"}]}' $B/api/chat)" = 401 ] && echo 1 || echo 0)
ok "chat rejects empty messages (400)" $([ "$(curl -s -b "$T/jar" -o /dev/null -w %{http_code} -X POST -d '{"messages":[]}' $B/api/chat)" = 400 ] && echo 1 || echo 0)
curl -sN -b "$T/jar" -X POST -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":0.2}' $B/api/chat |
  while IFS= read -r l; do case "$l" in *Hello*|*from*|*mock*|*DONE*) echo "$(date +%s.%N) $l";; esac; done > "$T/chat.out"
python3 - "$T/chat.out" > "$T/o" <<'EOF2'
import sys,json
rows=[l.rstrip("\n").split(" ",1) for l in open(sys.argv[1]) if l.strip()]
txt="".join(json.loads(r[1][5:])["choices"][0]["delta"]["content"] for r in rows if r[1].startswith("data: {"))
print("text", repr(txt))
print("done", any(r[1].strip()=="data: [DONE]" for r in rows))
ts=[float(r[0]) for r in rows if r[1].startswith("data: {")]
print("spread", round(ts[-1]-ts[0],2) if len(ts)>1 else -1)
EOF2
cat "$T/o" | head -3
ok "chat streams the mock text" $(grep -q "^text 'Hello from mock'" "$T/o" && echo 1 || echo 0)
ok "chat ends with [DONE]" $(grep -q "^done True" "$T/o" && echo 1 || echo 0)
ok "chat chunks arrive incrementally (not buffered)" $(python3 -c "print(1 if float(open('$T/o').read().split('spread ')[1].split()[0])>=0.4 else 0)")
curl -sN -b "$T/jar" -X POST -d '{"messages":[{"role":"user","content":"hi"}]}' $B/api/chat | head -c 4000 > "$T/chat.raw"
ok "chat sends ready event before tokens" $(grep -q '"bizen": "ready"' "$T/chat.raw" && echo 1 || echo 0)
echo "--- server.gen after the streamed chats above"
curl -s -b "$T/jar" $B/api/status | j "g=d['server']['gen'];print('genend',g['active'],g['tokens'],g['tps'],g['last_tps'] is not None and g['last_tps']>0)" | tee "$T/o"
ok "gen after streams: inactive, last_tps recorded" $(grep -q "^genend False None None True" "$T/o" && echo 1 || echo 0)
echo "--- chat cold start shows loading events"
curl -s -b "$T/jar" -X POST $B/api/stop >/dev/null; sleep 3
curl -sN -b "$T/jar" -X POST -d '{"messages":[{"role":"user","content":"hi"}]}' $B/api/chat | head -c 4000 > "$T/chat.cold"
ok "chat cold start: loading event then tokens" $(grep -q '"bizen": "loading"' "$T/chat.cold" && grep -q 'Hello' "$T/chat.cold" && echo 1 || echo 0)
echo "--- models / UI"
curl -s -b "$T/jar" $B/api/models; echo
curl -s $B/ | grep -c "<title>bizen-ctl"
echo "--- idle unload (ttl 24s, loop 15s)"
sleep 42
curl -s -b "$T/jar" $B/api/status | j "print(d['server']['state'])"
curl -s -b "$T/jar" "$B/api/logs?n=3" | j "print('\n'.join(d['lines']))"
echo "--- kev idle unload"
curl -s -b "$T/jar" $B/api/status | j "print(d['kev']['state'])"
curl -s -b "$T/jar" -X POST $B/api/kev/start >/dev/null; sleep 4
KCHILD=$(curl -s -b "$T/jar" $B/api/status | j "print(d['kev']['pid'])")
echo "--- start, then controller SIGTERM must stop child"
curl -s -b "$T/jar" -X POST $B/api/start >/dev/null; sleep 3
CHILD=$(curl -s -b "$T/jar" $B/api/status | j "print(d['server']['pid'])")
kill $CTL; sleep 2
if kill -0 "$CHILD" 2>/dev/null; then echo "LEAKED CHILD $CHILD"; kill "$CHILD"; else echo "child $CHILD stopped"; fi
if [ "$KCHILD" != "None" ] && kill -0 "$KCHILD" 2>/dev/null; then echo "LEAKED KEV $KCHILD"; kill "$KCHILD"; else echo "kev child stopped"; fi
echo "--- second controller: load progress and /api/loadmap (slow mock load with the fixture log)"
head -c 20000000 /dev/zero > "$T/model-big.gguf"
python3 - "$T/config.json" "$T/config2.json" <<'EOF3'
import json,sys
c=json.load(open(sys.argv[1])); c.update(listen_port=18993, internal_port=18994, idle_ttl_min=0)
c["kev"]=dict(c["kev"], managed=False); c["router"]=dict(c["router"], enabled=False)
for p in c["profiles"]: p["model"]=sys.argv[1].rsplit("/",1)[0]+"/model-big.gguf"
json.dump(c,open(sys.argv[2],"w"))
EOF3
B2=http://127.0.0.1:18993
MOCK_TOKENS=6 MOCK_LOAD_S=8 MOCK_LOG="$D/test/fixtures/startup-q3.log" python3 -u "$D/bizenctl.py" --config "$T/config2.json" > "$T/ctl2.log" 2>&1 &
CTL2=$!; sleep 1
curl -s -c "$T/jar2" -X POST -d '{"password":"pw"}' $B2/api/login >/dev/null
curl -s -b "$T/jar2" -X POST $B2/api/start >/dev/null; sleep 3
curl -s -b "$T/jar2" $B2/api/status | j "l=d['server']['load'];print('loadmid',d['server']['state'],sorted(l)==sorted(['phase','bytes_read','bytes_total','pct','eta_s','mbps','source']),l['pct'] is not None and 0<=l['pct']<=99,l['phase'])" | tee "$T/o"
ok "server.load while loading: fields present, pct in 0..99" $(grep -q "^loadmid starting True True" "$T/o" && echo 1 || echo 0)
ok "/api/loadmap needs login" $([ "$(curl -s -o /dev/null -w %{http_code} $B2/api/loadmap)" = 401 ] && echo 1 || echo 0)
sleep 7
curl -s -b "$T/jar2" $B2/api/status | j "l=d['server']['load'];print('loadend',d['server']['state'],l['phase'],l['pct'])" | tee "$T/o"
ok "server.load when running: ready, 100" $(grep -q "^loadend running ready 100" "$T/o" && echo 1 || echo 0)
curl -s -b "$T/jar2" $B2/api/loadmap | j "print('loadmap',d['state'],d['layers_total'],d['layers_offloaded'],len(d['devices']),len(d['per_layer']),d['have_data'],'gpu' in d['devices'][0] if d['devices'] else False)" | tee "$T/o"
ok "/api/loadmap shape on the fixture log" $(grep -q "^loadmap .* 44 24 4 43 True True" "$T/o" && echo 1 || echo 0)
echo "--- server.gen during and after a streamed /api/chat (6 extra mock tokens)"
( curl -sN -b "$T/jar2" -X POST -d '{"messages":[{"role":"user","content":"hi"}]}' $B2/api/chat > /dev/null ) &
CP=$!
sleep 1.2
curl -s -b "$T/jar2" $B2/api/status | j "g=d['server']['gen'];print('genmid',g['active'],g['tokens'] is not None and g['tokens']>=1,g['ttft_s'] is not None,sorted(g)==sorted(['active','requests','tokens','ttft_s','tps','last_tps']))" | tee "$T/o"
ok "gen mid-stream: active, tokens counted, ttft set, fields present" $(grep -q "^genmid True True True True" "$T/o" && echo 1 || echo 0)
wait $CP; sleep 1.5
curl -s -b "$T/jar2" $B2/api/status | j "g=d['server']['gen'];print('genend',g['active'],g['tokens'],g['tps'],g['last_tps'] is not None and g['last_tps']>0)" | tee "$T/o"
ok "gen after stream: inactive, last_tps recorded" $(grep -q "^genend False None None True" "$T/o" && echo 1 || echo 0)
curl -s -b "$T/jar2" "$B2/api/metrics" | j "print('genmetrics',all('gen' in x for x in d['samples']) and any(x['gen'] and x['gen']['active'] for x in d['samples']))" | tee "$T/o"
ok "metrics samples carry gen, with an active span" $(grep -q "genmetrics True" "$T/o" && echo 1 || echo 0)
kill $CTL2; sleep 2
CH2=$(pgrep -f "config2.json" | head -1); [ -n "$CH2" ] && kill $CH2
echo "--- controller log"; cat "$T/ctl.log"
echo "--- result"; if [ "$FAILS" = 0 ]; then echo "ALL CHECKS PASSED"; else echo "$FAILS CHECK(S) FAILED"; fi
exit $([ "$FAILS" = 0 ] && echo 0 || echo 1)
