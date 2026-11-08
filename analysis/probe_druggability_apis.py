# -*- coding: utf-8 -*-
import json, urllib.request, urllib.error

def post(url, q):
    req = urllib.request.Request(url, data=json.dumps({"query": q}).encode(),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"HTTPERR": e.code, "body": e.read(600).decode("utf-8", "ignore")}

OT = "https://api.platform.opentargets.org/api/v4/graphql"
PH = "https://pharos-api.ncats.io/graphql"

for tname in ["clinicalTargets", "TargetPrioritisation", "SafetyLiability"]:
    d = post(OT, '{ __type(name:"%s"){ fields { name type { name kind ofType { name } } } } }' % tname)
    print("==", tname)
    if "data" in d:
        for f in d["data"]["__type"]["fields"]:
            print("  ", f["name"], "-", f["type"]["name"] or (f["type"]["ofType"] or {}).get("name"))
    else:
        print(d)

print("== ITarget input fields")
d = post(PH, '{ __type(name:"ITarget"){ inputFields { name type { name kind ofType { name } } } } }')
print(json.dumps(d)[:800])

print("== pharos REN")
print(json.dumps(post(PH, '{ target(q:{name:"REN"}){ name tdl fam sym } }'))[:500])
print("== pharos REN alt")
print(json.dumps(post(PH, '{ target(q:{gene:"REN"}){ name tdl } }'))[:400])
