from pathlib import Path
import json,hashlib,datetime
r=Path(__file__).resolve().parents[1]
items=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or any(x in p.parts for x in ('.git','__pycache__','.pytest_cache')) or p.relative_to(r).as_posix()=='docs/SNAPSHOT_MANIFEST.json':continue
 items.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(r/'docs/SNAPSHOT_MANIFEST.json').write_text(json.dumps({'source_snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':items},indent=2)+'\n',encoding='utf-8',newline='\n')
print(len(items),'files recorded')
