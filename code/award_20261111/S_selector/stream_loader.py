"""Task-2-only loader: stream official parquet batches instead of retaining tables."""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tfb_common import REL

def batches(files,columns):
    for file in files:
        reader=pq.ParquetFile(file)
        for batch in reader.iter_batches(batch_size=65536,columns=columns):
            yield batch.to_pandas()

def load_panel(panel,split,truth=False):
    root=REL/'corridors'/panel/split
    files=sorted(glob.glob(str(root/'mainline_states_masked'/'mask_regime=*'/'*.parquet')))
    assert files,(panel,split,'missing masked parquet')
    days=set();observed_links=set()
    for frame in batches(files,['date','link_id']):
        days.update(frame.date.astype(str).unique());observed_links.update(frame.link_id.unique())
    days=sorted(days)
    topo=pd.read_csv(REL/'corridors'/panel/'network/lwr_mainline_topology.csv').sort_values('order_index')
    links=[x for x in topo.link_id if x in observed_links]
    day_map={x:i for i,x in enumerate(days)};link_map={x:i for i,x in enumerate(links)}
    shape=(len(days),len(links),288)
    out={'panel':panel,'split':split,'days':days,'links':links,'s':np.full(shape,np.nan,dtype='float32'),'q':np.full(shape,np.nan,dtype='float32')}
    if truth:out.update(s_true=np.full(shape,np.nan,dtype='float32'),elig=np.zeros(shape,dtype=bool))
    def fill(paths,is_truth):
        cols=['date','timestamp','link_id','speed_kmh','flow_vph']
        if is_truth:cols.append('is_score_eligible')
        for frame in batches(paths,cols):
            frame=frame[frame.link_id.isin(link_map)]
            if not len(frame):continue
            day=frame.date.astype(str).map(day_map);valid=day.notna()
            frame=frame.loc[valid];day=day.loc[valid].to_numpy(dtype=int)
            position=frame.link_id.map(link_map).to_numpy(dtype=int)
            ts=pd.to_datetime(frame.timestamp,utc=True)
            ti=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(dtype=int)
            idx=(day,position,ti)
            if is_truth:
                out['s_true'][idx]=frame.speed_kmh.to_numpy(dtype='float32')
                out['elig'][idx]=frame.is_score_eligible.to_numpy(dtype=bool)
            else:
                out['s'][idx]=frame.speed_kmh.to_numpy(dtype='float32')
                out['q'][idx]=frame.flow_vph.to_numpy(dtype='float32')
    fill(files,False)
    if truth:
        truth_files=sorted(glob.glob(str(root/'mainline_states'/'*'/'*.parquet')))
        assert truth_files,(panel,'missing training labels')
        fill(truth_files,True)
    metadata=pd.read_csv(REL/'corridors'/panel/'network/links.csv').set_index('link_id')
    out['lanes']=np.array([float(metadata.at[x,'lanes']) for x in links],dtype='float32')
    out['vf']=np.array([float(metadata.at[x,'free_speed_kmh']) for x in links],dtype='float32')
    out['dow']=pd.to_datetime(pd.Series(days)).dt.dayofweek.to_numpy()
    return out
