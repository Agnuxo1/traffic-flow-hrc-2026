"""Synthetic CPU checks for the no-confirmation export; no real models/outcomes."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from types import SimpleNamespace
import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
import numpy as np
import selection_only_cache as cache
import calibration_prediction_cache as prediction_cache
import finalize_traffic as final


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def source_function_hashes(path):
    text=Path(path).read_text()
    return {node.name:hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()
            for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}


def controller_namespace(path,root):
    text=Path(path).read_text()
    tree=ast.parse(text)
    names={'SAFE_NO_CONFIRM_SOURCE_SHA','SAFE_NO_CONFIRM_EXPORT_KEY','HELD_AGENT_STEMS'}
    functions={'no_confirmation_export_permitted','finalizer','restore_timers'}
    nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and
           any(isinstance(t,ast.Name) and t.id in names for t in n.targets)) or
           (isinstance(n,ast.FunctionDef) and n.name in functions)]
    calls=[];recs=[]
    def write(p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    ns={'Path':Path,'ROOT':root,'digest':sha,'read':lambda p:json.loads(Path(p).read_text()),
        'write':write,'__file__':str(Path(path).resolve()),'subprocess':SimpleNamespace(
            run=lambda args,**kw:(calls.append(args) or SimpleNamespace(returncode=0,stdout='enabled'))),
        'event':lambda *args,**kwargs:None,'records':lambda:recs,'jobs':lambda:{},
        'stage':lambda spec,archive:recs.append({'key':spec['key'],'job_id':None})}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns,calls,recs


def check_prediction_reuse(tmp,cachedir,panels,parts,selection):
    base=tmp/'verified-parent';folder=base/'verified/cpu-cache';folder.mkdir(parents=True)
    canonical={part:{k:[] for k in ('group','day','panel','cond','mask')} for part in ('cal','report')}
    counts=[];old_offset=0;prior_offset=0
    for values in parts:
        take=values['split']==1;sd=values['day'][take]
        nc=int((sd<=242).sum());nr=int((sd>242).sum());counts.append((nc,nr))
        local_group=values['group']-old_offset
        for part,which in [('cal',sd<=242),('report',sd>242)]:
            for key in canonical[part]:
                array=(local_group[take].astype('int64')+prior_offset
                       if key=='group' else values[key][take])
                canonical[part][key].append(array[which])
        old_offset=int(values['group'].max())+1
        prior_offset+=int(local_group[values['split']!=2].max())+1
    def identity(fields):
        values={k:np.concatenate(v) for k,v in fields.items()}
        return hashlib.sha256(b''.join(
            k.encode()+str(v.dtype).encode()+str(v.shape).encode()+np.ascontiguousarray(v).tobytes()
            for k,v in values.items())).hexdigest()
    pop={part:identity(fields) for part,fields in canonical.items()}
    protocol=hashlib.sha256(b'synthetic-cache-protocol').hexdigest()
    receipt={'status':'RDROP_CPU_VERIFIED','report_population_sha256':pop['report'],'normalizer_hashes':{}}
    frozen={'calibration_population_sha256':pop['cal'],'models':{}}
    inventory={'entries':{}};freeze={'files':{'synthetic-parent':{}}};expected=[]
    for offset in (101,202,303):
        name=f'F_{offset}';model_hash=hashlib.sha256(str(offset).encode()).hexdigest()
        norm_hash=hashlib.sha256(('norm'+str(offset)).encode()).hexdigest()
        cal=((np.arange(sum(nc for nc,nr in counts)*6)+offset)%311/1000).astype('float32').reshape(-1,6)
        report=((np.arange(sum(nr for nc,nr in counts)*6)+offset)%199/1000+.6).astype('float32').reshape(-1,6)
        path=folder/(name+'.npz');np.savez(path,cal=cal,report=report)
        sidecar=path.with_suffix('.json')
        sidecar.write_text(json.dumps({'identity':{'model_sha256':model_hash,'protocol_sha256':protocol,
            'calibration_population_sha256':pop['cal'],'report_population_sha256':pop['report']},
            'normalizer_sha256':norm_hash,'sha256':sha(path)}))
        for item in (path,sidecar):
            inventory['entries']['family/verified/cpu-cache/'+item.name]={'sha256':sha(item)}
        frozen['models'][name]={'arm':'F','sha256':model_hash}
        receipt['normalizer_hashes'][name]=norm_hash
        freeze['files']['synthetic-parent'][f'results/model_A_{offset}.pt']=model_hash
        rows=[];cpos=rpos=0
        for nc,nr in counts:
            rows.append(np.vstack((cal[cpos:cpos+nc],report[rpos:rpos+nr])))
            cpos+=nc;rpos+=nr
        expected.append(np.vstack(rows))
    objects={'verified/rdrop-family.json':receipt,'verified/frozen-family.json':frozen,
             'FROZEN-ARCHIVE-MANIFEST.json':inventory,'data-manifest.json':{
                 'cache_dir':str(cachedir),'files':[{'path':str(cachedir/(name+'.npz')),
                     'sha256':sha(cachedir/(name+'.npz'))} for name in panels]}}
    for name,value in objects.items():
        (base/name).write_text(json.dumps(value))
    spec={'parent_job_id':'synthetic-parent','arm':'A','root_relative':base.name,
          'receipt_sha256':sha(base/'verified/rdrop-family.json'),
          'frozen_sha256':sha(base/'verified/frozen-family.json'),
          'inventory_sha256':sha(base/'FROZEN-ARCHIVE-MANIFEST.json'),
          'data_manifest_sha256':sha(base/'data-manifest.json'),
          'prediction_protocol_sha256':protocol,'offsets':[101,202,303]}
    args=({'job_id':'synthetic-parent','arm':'A'},{'cache_dir':str(cachedir)},selection,freeze,
          tmp,panels,{'prediction_cache_reuse':spec})
    actual,provenance=prediction_cache.cached_selection_ensemble(*args)
    assert np.array_equal(actual,np.mean(expected,axis=0))
    assert provenance['new_calibration_inference'] is False and provenance['treatment_N_B_or_Fc_used'] is False
    broken={k:v.copy() for k,v in selection.items()};broken['mask'][0,0]=0.
    try:prediction_cache.cached_selection_ensemble(args[0],args[1],broken,*args[3:])
    except ValueError:pass
    else:raise AssertionError('A different population reused cached predictions')
    path=folder/'F_101.npz';original=path.read_bytes();path.write_bytes(original+b'corruption')
    try:prediction_cache.cached_selection_ensemble(*args)
    except ValueError:pass
    else:raise AssertionError('A corrupt verified cache fell back or was accepted')
    path.write_bytes(original)
    new,source=prediction_cache.cached_selection_ensemble(
        {'job_id':'different-new-weights','arm':'A'},*args[1:])
    assert new is None and source['mode']=='NEW_INFERENCE_FOR_DIFFERENT_UNCACHED_MODELS'
    return ['verified_parent_cache_reconstructs_original_panel_order_and_exact_float32_ensemble',
            'changed_population_or_corrupt_parent_cache_fails_without_replay_or_treatment_rescue']


def run(controller):
    here=Path(__file__).resolve().parent
    contract=cache.verify_source_identity(here)
    hashes=source_function_hashes(here/'finalize_traffic.py')
    assert all(hashes[n]==h for n,h in contract['preserved_finalizer_functions'].items())
    for name,h in contract['unchanged_dependency_sha256'].items():
        assert sha(here/name)==h
    tests=['original_choose_calibration_metric_ensemble_queue_CSV_functions_and_dependencies_unchanged']
    panels=final.PANELS2
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);cachedir=tmp/'cache';cachedir.mkdir();parts=[];offset=0
        base_days=np.array([0,30,210,213,220,227,228,232,240,242,243,250,257,258,270,272],dtype='int32')
        for pi,name in enumerate(panels):
            days=np.repeat(base_days,2).copy()
            if pi:days[-2:]=260 if pi==1 else 271
            split=np.repeat(np.array([0]*6+[1]*7+[2]*3,dtype='int8'),2)
            rng=np.random.default_rng(700+pi);n=len(split)
            values={'x_t':rng.normal(size=(n,5)).astype('float32'),
                    'x_m':rng.normal(size=(n,5)).astype('float32'),
                    'y':rng.integers(0,2,size=(n,6)).astype('float32'),
                    'mask':np.ones((n,6),dtype='float32'),
                    'group':np.repeat(np.arange(16,dtype='int32'),2),
                    'day':days,'split':split,'panel':np.full(n,pi,dtype='int8'),
                    'cond':np.tile([0,1],16).astype('int8'),'pers':np.zeros(n,dtype='float32')}
            np.savez_compressed(cachedir/(name+'.npz'),**values)
            copy={k:v.copy() for k,v in values.items()}
            copy['group']=copy['group']+offset;offset=int(copy['group'].max())+1;parts.append(copy)
        original={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
        for chosen in ({},{'experiment_family':final.IOU_FAMILY}):
            expected,note=final.ensemble_calibration_partition(original,chosen)
            actual,gotnote=cache.calibration_partition_only(cachedir,panels,chosen,final.IOU_FAMILY)
            assert note==gotnote and set(expected)==set(actual)
            assert all(np.array_equal(expected[k],actual[k],equal_nan=True) and
                       expected[k].dtype==actual[k].dtype for k in expected)
        tests.append('both_calibration_populations_exact_original_arrays_dtypes_global_day_and_panel_offsets')
        reads=[];reader=cache.read_rows
        def record(archive,key,start=0,count=None):
            reads.append((key,start,count));return reader(archive,key,start,count)
        with patch.object(cache,'read_rows',record):
            cache.calibration_partition_only(cachedir,panels,{},final.IOU_FAMILY)
            cache.calibration_partition_only(cachedir,panels,{'experiment_family':final.IOU_FAMILY},final.IOU_FAMILY)
        assert all(count is not None and start+count<=26
                   for key,start,count in reads if key not in ('split','day','group'))
        tests.append('streamed_feature_outcome_and_persistence_requests_exclude_every_confirmation_row')
        for point,arm in [(0.,'A'),(.02,'B')]:
            candidate={'job_id':'synthetic','selection_results':{}}
            for off in (101,202,303):
                for a,score in [('A',.7),('B',.7+point)]:
                    candidate['selection_results'][f'{a}_{off}']={'sel_exact':{'overall':score}}
            assert final.choose_family([candidate])['arm']==arm
        tests.append('unchanged_original_paired_selection_boundary')
        root=tmp/'mock-main';root.mkdir();jobs=root/'jobs';jobs.mkdir()
        jid='synthetic'
        (jobs/jid/'work').mkdir(parents=True)
        candidate['cache_dir']=str(cachedir);candidate['model_sha256']={}
        candidate['status']='SELECTION_COMPLETE_CONFIRMATION_SEALED'
        (jobs/jid/'work/result.json').write_text(json.dumps(candidate))
        (root/'freezes/traffic-flow-bench').mkdir(parents=True)
        (root/'freezes/traffic-flow-bench/manifest.json').write_text(json.dumps(
            {'files':{contract['opaque_legacy_job_ids'][0]:{},jid:{}}}))
        opaque=jobs/contract['opaque_legacy_job_ids'][0]/'work/result.json'
        old_read=Path.read_text
        def guarded_read(path,*args,**kw):
            if path==opaque:raise AssertionError('Opaque baseline was opened')
            return old_read(path,*args,**kw)
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1),
                                   set_num_threads=lambda n:None)
        out=root/'results'
        previous=Path.cwd()
        try:
            os.chdir(root)
            with ExitStack() as stack:
                for attr,value in [('ROOT',root),('JOBS',jobs),('torch',fake_torch),
                    ('checkpoints',lambda *a:[]),
                    ('ensemble',lambda models,x:np.full((len(x),6),.6,dtype='float32')),
                    ('build_queue',lambda *a:None),('combine',lambda *a:{'portal_submission':False}),
                    ('load_cache',lambda *a:(_ for _ in ()).throw(AssertionError('Full cache opened'))),
                    ('partition',lambda *a:(_ for _ in ()).throw(AssertionError('Partition called')))]:
                    stack.enter_context(patch.object(final,attr,value))
                stack.enter_context(patch.object(sys,'argv',['finalize_traffic.py','--out',str(out)]))
                stack.enter_context(patch.object(Path,'read_text',guarded_read))
                final.main()
        finally:
            os.chdir(previous)
        result=json.loads((out/'result.json').read_text())
        assert result['confirmation_proxy'] is None and result['confirmation_evaluated'] is False
        assert result['global_confirmation_never_evaluated'] is False
        assert not result['submission_sent'] and not (out/'confirmation_proxy.json').exists()
        tests.append('mock_main_exports_without_full_cache_confirmation_calls_or_opaque_baseline_reads')
        ordinary,_=cache.calibration_partition_only(cachedir,panels,{},final.IOU_FAMILY)
        tests.extend(check_prediction_reuse(tmp,cachedir,panels,parts,ordinary))
        ctlroot=tmp/'controller';ctlroot.mkdir()
        ns,calls,recs=controller_namespace(controller,ctlroot)
        for name in ['guards-reference.json']:
            assert (here.parent/name).exists()
        (ctlroot/'confirmation-policy.json').write_text('{"block_automatic_confirmation":true}')
        (ctlroot/'traffic-flow-bench-finalizer.json').write_text('{"source_archive_sha256":"wrong"}')
        assert ns['finalizer']('traffic-flow-bench',{},[],{})['confirmation_evaluation_blocked']
        assert not recs and not calls
        held={name+'.timer':{'active':'active','enabled':'enabled'} for name in ns['HELD_AGENT_STEMS']}
        (ctlroot/'deferred-timers-before.json').write_text(json.dumps(held))
        restored=ns['restore_timers']()
        assert all(v=='user_held_not_restored' for v in restored.values()) and not calls
        tests.append('controller_blocks_unaudited_export_and_never_reactivates_six_held_agent_timers')
    return {'status':'AUTHOR_CPU_PREFLIGHT_PASSED','count':len(tests),'tests':tests,
        'source_manifest_sha256':sha(here/'source_manifest.json'),
        'source_archive_sha256':sha(here.parent/contract['source_archive_file']),
        'controller_sha256':sha(controller),'controller_base_sha256':contract['controller_base_sha256'],
        'real_model_or_outcome_data_loaded':False,'real_GPU_used':False,'confirmation_evaluated':False,
        'real_systemctl_calls':False,'PC_models_loaded':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--controller',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    value=run(Path(args.controller));Path(args.out).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))
