from pathlib import Path
import ast,json,re,hashlib
root=Path(__file__).resolve().parents[1]
errors=[]
for f in root.rglob('*'):
 if not f.is_file() or any(x in f.parts for x in ('.git','__pycache__','.pytest_cache')):continue
 if f.stat().st_size>3000000:errors.append(str(f.relative_to(root))+': oversized file');continue
 if f.suffix.lower() in ('.pt','.safetensors','.parquet','.pem','.key'):errors.append(str(f.relative_to(root))+': forbidden artifact')
 if f.suffix=='.py':
  try:ast.parse(f.read_text(encoding='utf-8-sig'))
  except SyntaxError as e:errors.append(str(f.relative_to(root))+': syntax '+str(e.lineno))
for name in ('README.md','LICENSE','NOTICE','docs/RESEARCH_PROTOCOL.md','docs/PROJECT_STATUS.json'):
 if not (root/name).is_file():errors.append('Missing '+name)
status=json.loads((root/'docs/PROJECT_STATUS.json').read_text())
assert status['data_and_weights_included'] is False
manifest=json.loads((root/'docs/SNAPSHOT_MANIFEST.json').read_text())
assert len(manifest['files'])>0
for item in manifest['files']:
 f=root/item['path']
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:
  errors.append('Snapshot mismatch: '+item['path'])
if errors:raise SystemExit('\n'.join(errors))
print('Release packaging checks pass; not a scientific performance validation.')
