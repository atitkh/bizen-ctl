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
echo "--- controller log"; cat "$T/ctl.log"
