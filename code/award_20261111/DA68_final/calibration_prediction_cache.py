"""Read-only reuse of already verified, untreated parent probabilities."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
from selection_only_cache import read_rows


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read_sealed(path,expected):
    path=Path(path)
    if path.is_symlink() or digest(path)!=expected:
        raise ValueError('Verified parent cache identity changed; do not repeat inference')
    return json.loads(path.read_text())


def population_hash(data):
    h=hashlib.sha256()
    for key in ('group','day','panel','cond','mask'):
        array=np.ascontiguousarray(data[key])
        h.update(key.encode());h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode());h.update(array.tobytes())
    return h.hexdigest()


def cache_population_identity(cache_dir,panels,selection):
    """Check original split1 ordering and reconstruct the prior cal/report identities."""
    buckets={part:{key:[] for key in ('group','day','panel','cond','mask')}
             for part in ('cal','report')}
    counts=[];original_offset=0;prior_offset=0;position=0
    for pi,name in enumerate(panels):
        with zipfile.ZipFile(Path(cache_dir)/(name+'.npz')) as archive:
            split=read_rows(archive,'split');day=read_rows(archive,'day')
            groups=read_rows(archive,'group')
        if not np.all(np.diff(split.astype('int16'))>=0):
            raise ValueError('Verified prediction cache requires original block ordering')
        nt=int((split==0).sum());ns=int((split==1).sum())
        if not np.isin(split,[0,1,2]).all() or not nt or not ns:
            raise ValueError('Verified prediction cache partition differs')
        sd=day[nt:nt+ns]
        if not ((sd>=228)&(sd<=257)).all() or not (np.diff(sd)>=0).all():
            raise ValueError('Verified selection date population differs')
        nc=int((sd<=242).sum());nr=ns-nc
        sl=slice(position,position+ns)
        if not np.array_equal(selection['group'][sl],groups[nt:nt+ns]+original_offset):
            raise ValueError('Original selection row/group order differs')
        if not (selection['panel'][sl]==pi).all():
            raise ValueError('Original selection panel identity differs')
        for part,take in [('cal',slice(0,nc)),('report',slice(nc,ns))]:
            view={'group':groups[nt:nt+ns].astype('int64')+prior_offset,'day':sd,
                  'panel':selection['panel'][sl],'cond':selection['cond'][sl],
                  'mask':selection['mask'][sl]}
            for key,array in view.items():
                buckets[part][key].append(array[take])
        counts.append((nc,nr));position+=ns
        original_offset=int((groups+original_offset).max())+1
        prior_offset+=int(groups[:nt+ns].max())+1
    if position!=len(selection['x']):
        raise ValueError('Verified selection row count differs')
    merged={part:{key:np.concatenate(values) for key,values in fields.items()}
            for part,fields in buckets.items()}
    return {part:population_hash(values) for part,values in merged.items()},counts


def cached_selection_ensemble(chosen,source,selection,freeze,root,panels,contract):
    spec=contract['prediction_cache_reuse']
    if chosen['job_id']!=spec['parent_job_id'] or chosen['arm']!=spec['arm']:
        return None,{'mode':'NEW_INFERENCE_FOR_DIFFERENT_UNCACHED_MODELS'}
    # A matching parent cache must validate completely. Corruption never triggers
    # a GPU/CPU replay, a different method, or a treatment-cache rescue.
    base=Path(root)/spec['root_relative']
    receipt=read_sealed(base/'verified/rdrop-family.json',spec['receipt_sha256'])
    frozen=read_sealed(base/'verified/frozen-family.json',spec['frozen_sha256'])
    inventory=read_sealed(base/'FROZEN-ARCHIVE-MANIFEST.json',spec['inventory_sha256'])
    data=read_sealed(base/'data-manifest.json',spec['data_manifest_sha256'])
    if receipt['status']!='RDROP_CPU_VERIFIED':
        raise ValueError('Parent probability verification is incomplete')
    if Path(source['cache_dir']).resolve()!=Path(data['cache_dir']).resolve():
        raise ValueError('Verified parent data directory differs')
    for item in data['files']:
        if digest(item['path'])!=item['sha256']:
            raise ValueError('Verified input data bytes changed; no inference replay')
    pop,counts=cache_population_identity(source['cache_dir'],panels,selection)
    if pop['cal']!=frozen['calibration_population_sha256'] or pop['report']!=receipt['report_population_sha256']:
        raise ValueError('Verified parent population differs')
    models=[]
    for offset in spec['offsets']:
        name=f'F_{offset}'
        model=frozen['models'][name]
        expected=freeze['files'][chosen['job_id']][f"results/model_{chosen['arm']}_{offset}.pt"]
        if model['arm']!='F' or model['sha256']!=expected:
            raise ValueError('Verified untreated parent weights differ')
        cache=base/'verified/cpu-cache'/f'{name}.npz'
        sidecar=cache.with_suffix('.json')
        meta=read_sealed(sidecar,inventory['entries']['family/verified/cpu-cache/'+sidecar.name]['sha256'])
        identity={'model_sha256':expected,'protocol_sha256':spec['prediction_protocol_sha256'],
            'calibration_population_sha256':pop['cal'],'report_population_sha256':pop['report']}
        if meta['identity']!=identity or meta['normalizer_sha256']!=receipt['normalizer_hashes'][name]:
            raise ValueError('Verified parent model/normalizer/population identity differs')
        cache_hash=digest(cache)
        if cache.is_symlink() or cache_hash!=meta['sha256'] or cache_hash!=inventory['entries']['family/verified/cpu-cache/'+cache.name]['sha256']:
            raise ValueError('Verified parent probabilities changed; no inference replay')
        with np.load(cache,allow_pickle=False) as arrays:
            cal=arrays['cal'];report=arrays['report']
        if cal.dtype!=np.float32 or report.dtype!=np.float32:
            raise ValueError('Verified parent probability dtype differs')
        if cal.shape!=(sum(c for c,r in counts),6) or report.shape!=(sum(r for c,r in counts),6):
            raise ValueError('Verified parent probability shape differs')
        if not np.isfinite(cal).all() or not np.isfinite(report).all():
            raise ValueError('Invalid verified parent probabilities')
        # Prior cache order is panel0-cal,panel1-cal,... and separately report.
        # Original split1 order is panel0-cal,panel0-report,panel1-cal,...
        parts=[];cal_position=0;report_position=0
        for nc,nr in counts:
            parts.extend([cal[cal_position:cal_position+nc],
                          report[report_position:report_position+nr]])
            cal_position+=nc;report_position+=nr
        models.append(np.concatenate(parts,axis=0))
    probabilities=np.mean(models,axis=0)
    provenance={'mode':'REUSED_VERIFIED_UNTREATED_PARENT_CPU_PROBABILITIES',
        'parent_job_id':spec['parent_job_id'],'model_count':len(models),
        'receipt_sha256':spec['receipt_sha256'],'inventory_sha256':spec['inventory_sha256'],
        'original_split1_row_order_reconstructed':True,'new_calibration_inference':False,
        'treatment_N_B_or_Fc_used':False,
        'CPU_GPU_scope':'Inherited calibration parity only; no new full30day ensemble-grid GPU parity claim',
        'original_score_threshold_grid_and_ensemble_mean_unchanged':True}
    return probabilities,provenance
