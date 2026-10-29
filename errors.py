import re, collections
txt = open("progress.log").read()
blocks = txt.split("FAILED:")[1:]

causes = collections.Counter()
for b in blocks:
    # the lowest-level exceptions, before the "direct cause" wrapping
    found = re.findall(r"^(?:rasterio\.errors\.|rasterio\._err\.)?(\w*(?:Error|Exception))\s*:\s*(.{0,200})", b, re.M)
    root = [f"{t}: {m}" for t, m in found if t != "RuntimeError"]
    causes[root[0] if root else "no root cause logged"] += 1

for k, v in causes.most_common():
    print(v, k)