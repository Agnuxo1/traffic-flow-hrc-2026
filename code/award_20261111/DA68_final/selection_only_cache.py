"""Exact original calibration populations, without reading confirmation payloads."""
from pathlib import Path
import hashlib
import json
import math
import zipfile
import numpy as np


def read_rows(archive, key, start=0, count=None):
    with archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):
            shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Unsupported numeric NPY format')
        if fortran or dtype.hasobject or not shape:
            raise ValueError('Expected numeric C-order rows')
        rows=shape[0]-start if count is None else count
        if start<0 or rows<0 or start+rows>shape[0]:
            raise ValueError('NPY row span is out of bounds')
        row_bytes=math.prod(shape[1:])*dtype.itemsize
        skip=start*row_bytes
        while skip:
            chunk=stream.read(min(skip,1024*1024))
            if not chunk:
                raise ValueError('Truncated skipped NPY payload')
            skip-=len(chunk)
        payload=stream.read(rows*row_bytes)
        if len(payload)!=rows*row_bytes:
            raise ValueError('Truncated requested NPY payload')
        return np.frombuffer(payload,dtype=dtype).reshape((rows,*shape[1:]))


def calibration_partition_only(cache_dir, panels, chosen, iou_family):
    """Match load_cache + ensemble_calibration_partition, using only chosen rows."""
    metadata=[]
    for name in panels:
        with zipfile.ZipFile(Path(cache_dir)/(name+'.npz')) as archive:
            # Dates/split/group IDs are metadata, not features or target outcomes.
            split=read_rows(archive,'split')
            day=read_rows(archive,'day')
            group=read_rows(archive,'group')
            if not (len(split)==len(day)==len(group)) or not len(split):
                raise ValueError('Inconsistent cache metadata')
            if not np.isin(split,[0,1,2]).all():
                raise ValueError('Unexpected cache partition')
            metadata.append((name,split,day,group))
    days=max(int(day.max()) for _,_,day,_ in metadata)+1
    is_iou=chosen.get('experiment_family')==iou_family
    fields={'x':'x_t','y':'y','mask':'mask','group':'group',
            'panel':'panel','cond':'cond','pers':'pers'}
    parts={k:[] for k in fields}
    offset=0
    for name,split,day,group in metadata:
        take=((split==0)&(day>=days-60)) if is_iou else split==1
        indices=np.flatnonzero(take)
        runs=np.split(indices,np.flatnonzero(np.diff(indices)!=1)+1) if len(indices) else []
        spans=[(int(run[0]),len(run)) for run in runs] or [(0,0)]
        with zipfile.ZipFile(Path(cache_dir)/(name+'.npz')) as archive:
            for outkey,inkey in fields.items():
                # Each requested outcome/feature span excludes split2.
                arrays=[read_rows(archive,inkey,start,count) for start,count in spans]
                selected=np.concatenate(arrays,axis=0)
                if outkey=='group':
                    selected=selected+offset
                parts[outkey].append(selected)
        # Preserve the exact original panel offsets, including unselected IDs.
        offset=int((group+offset).max())+1
    result={k:np.concatenate(v,axis=0) for k,v in parts.items()}
    note=('nested checkpoint-selection days; decision gate not used for threshold'
          if is_iou else 'original public selection days')
    return result,note


def verify_source_identity(here):
    here=Path(here)
    manifest=json.loads((here/'source_manifest.json').read_text())
    for name,expected in manifest.items():
        path=here/name
        if Path(name).name!=name or path.is_symlink():
            raise ValueError('Invalid source manifest path')
        with path.open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:
            raise ValueError('Finalizer source identity changed: '+name)
    contract=json.loads((here/'finalizer-contract.json').read_text())
    if contract['confirmation_evaluation']!='disabled':
        raise ValueError('This variant never supports confirmation evaluation')
    return contract
