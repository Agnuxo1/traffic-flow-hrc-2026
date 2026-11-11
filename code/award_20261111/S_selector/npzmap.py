"""Read-only mapping of own uncompressed NumPy archives, without loading other keys."""
import struct,zipfile
from pathlib import Path
import numpy as np

def array(path,key):
    with zipfile.ZipFile(path) as z:
        info=z.getinfo(key+'.npy');assert info.compress_type==zipfile.ZIP_STORED
        with z.open(info) as stream:
            version=np.lib.format.read_magic(stream)
            if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
            elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
            else:raise ValueError('Unsupported NPY version')
            relative=stream.tell()
    assert not dtype.hasobject
    with Path(path).open('rb') as f:f.seek(info.header_offset);header=f.read(30)
    assert header[:4]==b'PK\x03\x04';n,e=struct.unpack_from('<HH',header,26)
    return np.memmap(path,dtype=dtype,mode='r',offset=info.header_offset+30+n+e+relative,shape=shape,order='F' if fortran else 'C')
