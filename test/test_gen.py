#!/usr/bin/env python3
"""Unit checks for SSECount (token chunks counted across arbitrary splits) and the Supervisor generation stats."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

def ev(content=None, **kw):
    d = {"choices": [{"delta": ({"content": content} if content is not None else {})}]}
    d.update(kw)
    return b"data: " + json.dumps(d).encode() + b"\n\n"

stream = ev(None) + ev("Hel") + ev("lo") + b'data: {"bizen":"ready"}\n\n' + ev("") + \
    b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n' + b"data: [DONE]\n\n"
c = b.SSECount()
check("counts only token events", c.feed(stream) == 2)
c = b.SSECount()
check("same count when split byte by byte", sum(c.feed(stream[i:i + 1]) for i in range(len(stream))) == 2)
c = b.SSECount()
check("CRLF framing", c.feed(stream.replace(b"\n", b"\r\n")) == 2)
c = b.SSECount()
check("garbage and error events never raise / count", c.feed(b"data: {not json\n\n\x00\xff\n\ndata: {\"error\":\"x\"}\n\n: comment\n\n") == 1)
check("reasoning_content counts", b.SSECount().feed(b'data: {"choices":[{"delta":{"reasoning_content":"hm"}}]}\n\n') == 1)
check("text-completion chunk counts", b.SSECount().feed(b'data: {"choices":[{"text":"a"}]}\n\n') == 1)

class C:
    def get(self, k): return None
s = b.Supervisor(C())
g = s.gen_info()
check("idle: inactive, nothing known", g["active"] is False and g["tokens"] is None and g["tps"] is None and g["last_tps"] is None)
gid = s.gen_begin()
g = s.gen_info()
check("begun, no token yet: active, tps None", g["active"] and g["tokens"] == 0 and g["tps"] is None and g["ttft_s"] is None)
for _ in range(6):
    s.gen_chunk(gid, 1)
    time.sleep(0.1)
g = s.gen_info()
check("running: tokens, ttft and a plausible live tps", g["tokens"] == 6 and g["ttft_s"] is not None and g["tps"] and 5 < g["tps"] < 15)
s.gen_end(gid)
g = s.gen_info()
check("finished: inactive, last_tps kept", not g["active"] and g["tokens"] is None and g["last_tps"] and 5 < g["last_tps"] < 15)
gid = s.gen_begin(); s.gen_end(gid, 40, 31.5)
check("non-streamed reply fills last_tps from timings", s.gen_info()["last_tps"] == 31.5)
s.gen_chunk(999, 3); s.gen_end(999)
check("unknown ids are ignored", not s.gen_info()["active"])
gid = s.gen_begin(); s.gen_chunk(gid, 1); time.sleep(2.3)
check("stalled stream: tps falls to 0", s.gen_info()["tps"] == 0.0)
s.gen_end(gid)

print("FAILED: %s" % fails if fails else "all passed")
sys.exit(1 if fails else 0)
