import re
import json
import requests

r = requests.get("https://fund.eastmoney.com/pingzhongdata/001235.js", timeout=8)
txt = r.text

m_scale = re.search(r"var\s+Data_fluctuationScale\s*=\s*(\{.*?\});", txt, re.S)
if m_scale:
    print("Scale JSON:", json.loads(m_scale.group(1)))

m_holder = re.search(r"var\s+Data_holderStructure\s*=\s*(\{.*?\});", txt, re.S)
if m_holder:
    print("Holder JSON:", json.loads(m_holder.group(1)))
