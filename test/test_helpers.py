#!/usr/bin/env python3
"""Unit tests for _inside, _origin, hf_status_error, hf_net_error."""
import os, sys, socket, http.client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bizenctl as b

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

# _inside tests
check("_inside: equal paths", b._inside("/a/b", "/a/b") == True)
check("_inside: child path", b._inside("/a/b/c", "/a/b") == True)
check("_inside: deeper child", b._inside("/a/b/c/d", "/a/b") == True)
check("_inside: sibling with shared prefix (bc vs b)", b._inside("/a/bc", "/a/b") == False)
check("_inside: parent of root", b._inside("/a", "/a/b") == False)
check("_inside: unrelated", b._inside("/x/y", "/a/b") == False)
check("_inside: relative vs absolute", b._inside("relative", "/absolute") == False)
check("_inside: relative vs relative", b._inside("a/b", "a") == True)
check("_inside: mixed case different paths", b._inside("/A/B", "/a/b") == False)

# _origin tests
check("_origin: https default port", b._origin("https://example.com") == ("https", "example.com", 443))
check("_origin: http default port", b._origin("http://example.com") == ("http", "example.com", 80))
check("_origin: explicit port", b._origin("https://example.com:8443") == ("https", "example.com", 8443))
check("_origin: host case folding", b._origin("HTTPS://Example.COM") == ("https", "example.com", 443))
check("_origin: URL with path", b._origin("https://example.com/path/to/file") == ("https", "example.com", 443))
check("_origin: URL with query", b._origin("https://example.com?x=1") == ("https", "example.com", 443))
check("_origin: URL with user:pass@", b._origin("https://user:pass@example.com:443") == ("https", "example.com", 443))

# hf_status_error tests
e1 = b.hf_status_error(401, True)
check("hf_status_error 401 with token: code and message", e1.code == 403 and "gated" in e1.msg and "token rejected" in e1.msg)
e2 = b.hf_status_error(401, False)
check("hf_status_error 401 without token: code and message", e2.code == 403 and "gated" in e2.msg and "token rejected" not in e2.msg)
e3 = b.hf_status_error(403, True)
check("hf_status_error 403 with token: code and message", e3.code == 403 and "gated" in e3.msg)
e4 = b.hf_status_error(403, False)
check("hf_status_error 403 without token: code and message", e4.code == 403 and "gated" in e4.msg)
e5 = b.hf_status_error(404, True)
check("hf_status_error 404 with token: code and message", e5.code == 404 and "gated" in e5.msg)
e6 = b.hf_status_error(404, False)
check("hf_status_error 404 without token: code and message", e6.code == 404 and "gated" in e6.msg)
e7 = b.hf_status_error(429, True)
check("hf_status_error 429: code and rate limiting", e7.code == 429 and "rate limiting" in e7.msg)
e8 = b.hf_status_error(500, False)
check("hf_status_error 500: 502-class with status", e8.code == 502 and "500" in e8.msg)
e9 = b.hf_status_error(418, True)
check("hf_status_error 418: 502-class with status", e9.code == 502 and "418" in e9.msg)

# hf_net_error tests
hfe = b.HFError(404, "original")
check("hf_net_error HFError instance: same object", b.hf_net_error(hfe, "example.com") is hfe)
to_err = TimeoutError("timeout")
hfe_to = b.hf_net_error(to_err, "example.com")
check("hf_net_error TimeoutError: 504 with host", hfe_to.code == 504 and "example.com" in hfe_to.msg)
sock_to = socket.timeout("timeout")
hfe_sock = b.hf_net_error(sock_to, "example.com")
check("hf_net_error socket.timeout: 504 with host", hfe_sock.code == 504 and "example.com" in hfe_sock.msg)
os_err = OSError("connection refused")
hfe_os = b.hf_net_error(os_err, "example.com")
check("hf_net_error OSError: 502 with host", hfe_os.code == 502 and "example.com" in hfe_os.msg)
http_err = http.client.HTTPException("bad response")
hfe_http = b.hf_net_error(http_err, "example.com")
check("hf_net_error HTTPException: 502 with host", hfe_http.code == 502 and "example.com" in hfe_http.msg)
val_err = ValueError("invalid")
hfe_val = b.hf_net_error(val_err, "example.com")
check("hf_net_error ValueError: 502 generic", hfe_val.code == 502 and "unexpected error" in hfe_val.msg)

# Long OSError message truncation
long_msg = "x" * 500
long_os_err = OSError(long_msg)
hfe_long = b.hf_net_error(long_os_err, "example.com")
check("hf_net_error OSError message truncated to 300", len(hfe_long.msg) <= 300)

print("FAILED: %d" % len(fails) if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
