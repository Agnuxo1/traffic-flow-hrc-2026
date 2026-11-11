"""Freeze on the original selection population and export without confirmation."""
from pathlib import Path
import argparse,hashlib,json,sys
from itertools import zip_longest
import numpy as np
import pandas as pd
import torch
from replica_queue import (PANELS2,THRESHOLDS,H,load_cache,partition,normalised,model_for,
    probabilities,exact_score,feature_matrix,write_json)
from replica_loader import load_panel_truth
from tfb_common import REL
from promotion import IOU_FAMILY,iou_gate
from selection_only_cache import calibration_partition_only,verify_source_identity
from calibration_prediction_cache import cached_selection_ensemble

ROOT=Path('/mnt/datos/competition-campaigns/traffic-umud-20261105')
JOBS=Path('/home/ai/servidor-ia/jobs')
BASE_SHA='c3c6eeaabdcc1335480f33fae9cc37a0ec982f6f428f273bc256034bc2caebb0'

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def choose_family(candidates):
    """No confirmation fields accepted. B requires the preregistered paired selection gain."""
    choices=[]
    for c in candidates:
        if c.get('experiment_family')==IOU_FAMILY and c.get('cohort')!='primary':
            continue  # Replica checks reproducibility; it can never win by a better selection score.
        a=[c['selection_results'][f'A_{s}']['sel_exact']['overall'] for s in (101,202,303)]
        b=[c['selection_results'][f'B_{s}']['sel_exact']['overall'] for s in (101,202,303)]
        delta=np.array(b)-np.array(a)
        gate=None
        if c.get('experiment_family')==IOU_FAMILY:
            gate=iou_gate(candidates,c.get('protocol_sha256'))
            use_b=gate['passed']
        else:use_b=float(np.mean(delta))>=.01 and int((delta>0).sum())>=2
        arm='B' if use_b else 'A';score=float(np.mean(b if use_b else a))
        item=dict(job_id=c['job_id'],arm=arm,selection_mean=score,paired_selection_delta=delta.tolist())
        if gate is not None:item.update(experiment_family=IOU_FAMILY,cohort='primary',six_seed_selection_gate=gate)
        choices.append(item)
    assert choices,'No eligible primary family'
    return sorted(choices,key=lambda x:(-x['selection_mean'],x['job_id']))[0]

def ensemble_calibration_partition(d,chosen):
    if chosen.get('experiment_family')==IOU_FAMILY:
        days=int(d['day'].max())+1
        inner=(d['split']==0)&(d['day']>=days-60)
        local=dict(d,split=np.where(inner,1,9).astype('int8'))
        return partition(local,1,'x_t'),'nested checkpoint-selection days; decision gate not used for threshold'
    return partition(d,1,'x_t'),'original public selection days'

def checkpoints(job_id,arm,freeze):
    out=[]
    for offset in (101,202,303):
        p=JOBS/job_id/'work/results'/f'model_{arm}_{offset}.pt'
        assert digest(p)==freeze['files'][job_id][f'results/{p.name}']
        # Our own, hash-frozen checkpoints contain NumPy normalisers.
        ck=torch.load(p,map_location='cpu',weights_only=False)
        net=model_for(torch,len(ck['mean']),ck['cfg']).cuda();net.load_state_dict(ck['state_dict']);net.eval()
        out.append((ck,net))
    return out

def ensemble(models,x):
    return np.mean([probabilities(torch,net,torch.as_tensor(normalised(x,ck['mean'],ck['std']),device='cuda'))
                    for ck,net in models],axis=0)

def build_queue(models,threshold,out):
    predictions=[]
    for split in ('validation','private'):
        for panel in PANELS2:
            P=load_panel_truth(panel,split,truth=False)
            windows=pd.read_csv(REL/'task2'/panel/split/'window_index.csv')
            history=pd.read_parquet(REL/'task2'/panel/split/'window_history.parquet')
            template=pd.read_csv(REL/'task2'/panel/split/'sample_submission_queue.csv')
            days={v:i for i,v in enumerate(P['days'])};links={str(v):i for i,v in enumerate(P['links'])}
            topo=pd.read_csv(REL/'corridors'/panel/'network/lwr_mainline_topology.csv')
            order=dict(zip(topo.link_id.astype(str),topo.order_index))
            obs_order=np.array([order[str(v)] for v in P['links']]); sort=np.argsort(obs_order)
            for w in windows.itertuples():
                origin=pd.Timestamp(w.forecast_origin);t0=origin.hour*12+origin.minute//5;d=days[str(w.date)]
                ss=P['s'][d].copy();qq=P['q'][d].copy()
                hh=history[history.window_id==w.window_id]
                for row in hh.itertuples():
                    ts=pd.Timestamp(row.timestamp);ti=ts.hour*12+ts.minute//5
                    assert ts<origin and t0-12<=ti<t0,'No future observation allowed'
                    if str(row.link_id) in links:
                        i=links[str(row.link_id)];ss[i,ti]=row.speed_kmh;qq[i,ti]=row.flow_vph
                view=dict(P,s={d:ss},q={d:qq})
                prob=ensemble(models,feature_matrix(view,d,t0))
                target=template[template.window_id==w.window_id]
                for step in range(1,H+1):
                    stamp=(origin+pd.Timedelta(minutes=5*step)).strftime('%Y-%m-%dT%H:%M:%SZ')
                    part=target[target.timestamp==stamp]
                    z=np.interp(np.array([order[str(v)] for v in part.link_id]),obs_order[sort],prob[sort,step-1])
                    predictions.extend(dict(panel=panel,split=split,window_id=w.window_id,link_id=str(link),
                                            timestamp=stamp,queue_pred=int(value>=threshold))
                                       for link,value in zip(part.link_id,z))
            print(json.dumps(dict(stage='inference',panel=panel,split=split)),flush=True)
    frame=pd.DataFrame(predictions)
    keys=['panel','split','window_id','link_id','timestamp']
    assert not frame.duplicated(keys).any()
    frame.to_csv(out/'queue_predictions.csv',index=False)
    return frame.set_index(keys).queue_pred

def combine(predictions,out,base):
    assert digest(base)==BASE_SHA,'Frozen historical base changed'
    dest=out/'traffic_candidate.csv';rows=0;queue_rows=0
    a=pd.read_csv(base,chunksize=100000,float_precision='round_trip')
    b=pd.read_csv(REL/'submission_key.csv',usecols=['submission_id','task','panel','split','window_id','link_id','timestamp'],chunksize=100000)
    with dest.open('w') as stream:
        for frame,key in zip_longest(a,b):
            assert frame is not None and key is not None,'Base and key row counts differ'
            assert len(frame)==len(key) and np.array_equal(frame.submission_id,key.submission_id)
            mask=key.task=='queue';wanted=pd.MultiIndex.from_frame(key.loc[mask,['panel','split','window_id','link_id','timestamp']])
            got=predictions.reindex(wanted);assert not got.isna().any(),'A required queue cell is missing'
            frame.loc[mask,'queue_pred']=got.to_numpy(dtype=int)
            assert np.isfinite(frame[['speed_kmh','flow_vph','queue_pred','path_flow']].to_numpy()).all()
            assert frame.loc[mask,'queue_pred'].isin([0,1]).all()
            frame.to_csv(stream,index=False,header=(rows==0));rows+=len(frame);queue_rows+=int(mask.sum())
    assert queue_rows==len(predictions) and rows>queue_rows
    return dict(path=str(dest),sha256=digest(dest),rows=rows,queue_rows=queue_rows,base_sha256=BASE_SHA,
                preserved_tasks=['state','physics-from-state','odme'],portal_submission=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results');a=ap.parse_args()
    contract=verify_source_identity(Path(__file__).resolve().parent)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4);out=Path(a.out);out.mkdir(exist_ok=True)
    freeze=json.loads((ROOT/'freezes/traffic-flow-bench/manifest.json').read_text())
    candidates=[]
    for jid in freeze['files']:
        if jid in contract['opaque_legacy_job_ids']:continue
        p=JOBS/jid/'work/result.json'
        if not p.exists():continue
        c=json.loads(p.read_text())
        if c.get('status')=='SELECTION_COMPLETE_CONFIRMATION_SEALED':candidates.append(c|{'job_id':jid})
    assert candidates,'No complete six-model family; keep historical accepted submission unchanged'
    chosen=choose_family(candidates);write_json(out/'frozen_choice.json',chosen)
    source=next(c for c in candidates if c['job_id']==chosen['job_id'])
    selection,calibration_note=calibration_partition_only(source['cache_dir'],PANELS2,chosen,IOU_FAMILY)
    models=checkpoints(chosen['job_id'],chosen['arm'],freeze)
    pr,calibration_prediction_source=cached_selection_ensemble(
        chosen,source,selection,freeze,ROOT,PANELS2,contract)
    if pr is None:
        pr=ensemble(models,selection['x'])
    threshold=max(THRESHOLDS,key=lambda t:exact_score(pr>=t,selection['y'],selection['mask'],selection['group'],selection['panel'],selection['cond'])['overall'])
    choice=chosen|dict(threshold=threshold,selection_only=True,threshold_calibration=calibration_note,model_hashes=source['model_sha256'],calibration_prediction_source=calibration_prediction_source)
    write_json(out/'frozen_choice.json',choice)
    # This export variant never loads/evaluates confirmation or reads opaque metrics.
    del selection,pr
    prediction=build_queue(models,threshold,out)
    artifact=combine(prediction,out,ROOT/'inputs/traffic/sub_v1.csv')
    result=dict(status='PREPARED_FOR_OWNER_REVIEW',chosen=choice,artifact=artifact,confirmation_proxy=None,
                confirmation_evaluated=False,confirmation_status='SKIPPED_NO_CONFIRMATION_VARIANT',
                prior_early_confirmation_incident='incidents/early-confirmation-20b0502495c44cdf/manifest.json',
                global_confirmation_never_evaluated=False,
                official_improvement_verified=False,submission_sent=False,
                code_pdf_deadline_utc='2026-11-11T11:59:00Z')
    write_json(out/'result.json',result);write_json('result.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
