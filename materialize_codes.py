"""Materialize the stored INT8 weight codes for a handoff model: original + permuted + faulted.

Loads the registered checkpoint through the frozen harness, quantizes to INT8 exactly as the
campaign did, applies the recorded function-equivalent permutation, and writes the unsigned
stored codes (two's-complement bytes, one per weight) of the attacked tensor for both layouts,
plus the single/few-byte faulted copy.  A self-check decodes the flipped element and asserts the
old/new weight values match what the eval artifact recorded, so the bytes are proven faithful.

Usage (run where torch + the venv are available; CPU is fine, ~minutes/model):
    PYTHONPATH=/gpfs/work2/0/prjs1717/xwang/BFA_llm \
    REGIONALIGN_HF_HOME=/gpfs/work2/0/prjs1717/xwang/huggingface_cache \
    python materialize_codes.py <model_dir>        # e.g. stablelm2-1.6b__wikitext ; omit = all

Writes <model_dir>/codes/{original,permuted,faulted}_<tensor>.u8.npy and a bytes_manifest.json.
Read-only over the campaign; only adds files under this handoff folder.
"""
import os, sys, json, glob
import numpy as np
from regionalign.harness import core as C
from regionalign.experiments.clean_mmlu import load_model

HERE=os.path.dirname(os.path.abspath(__file__))
CKPT={'llama3.2-1b':'models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08',
      'qwen2.5-1.5b':'models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323',
      'smollm2-1.7b':'models--HuggingFaceTB--SmolLM2-1.7B/snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71',
      'stablelm2-1.6b':'models--stabilityai--stablelm-2-1_6b/snapshots/f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef'}
HFHUB=os.path.join(os.environ.get('REGIONALIGN_HF_HOME','/gpfs/work2/0/prjs1717/xwang/huggingface_cache'),'hub')

class RT:  # minimal runtime for apply_permutation (needs .lm, tracks .residual)
    residual=None
    def __init__(self,lm): self.lm=lm

def codes_u8(lm,name):
    import torch
    n=lm.get_parameter(name).numel()
    idx=torch.arange(n,dtype=torch.long)
    return C.stored_codes(lm,name,idx).cpu().numpy().astype(np.uint8)

def do(model_dir):
    a=json.load(open(os.path.join(HERE,model_dir,'attack.json')))
    model_key=[k for k in CKPT if a['model'].lower().startswith(k.split('-')[0]) and k in model_dir][0]
    spec=json.load(open(os.path.join(HERE,model_dir,'permutation.json')))
    tensor=a['region']['tensor']; ckpt=os.path.join(HFHUB,CKPT[model_key])
    out=os.path.join(HERE,model_dir,'codes'); os.makedirs(out,exist_ok=True)
    print(f'[{model_dir}] load {model_key} int8 on cpu ...',flush=True)
    model,tok,audit,timings=load_model(ckpt,'int8','cpu')
    rt=RT(model)
    orig=codes_u8(model,tensor); np.save(f'{out}/original_codes.u8.npy',orig)
    C.apply_permutation(rt,spec)
    perm=codes_u8(model,tensor); np.save(f'{out}/permuted_codes.u8.npy',perm)
    # faulted copy: XOR the recorded bits into the permuted codes
    fault=perm.copy()
    for fl in a['flips']:
        e=fl['code_index_in_tensor']; b=fl['bit']; fault[e]^=(1<<b)
    np.save(f'{out}/faulted_codes.u8.npy',fault)
    # self-check: decode flipped element old/new against the eval record
    import torch
    ok=True
    for fl,val in zip(a['flips'],a['flip_values']):
        e=fl['code_index_in_tensor']; idx=torch.tensor([e])
        old=C.coded_values(model,tensor,idx,torch.tensor([int(perm[e])])).item()
        new=C.coded_values(model,tensor,idx,torch.tensor([int(fault[e])])).item()
        match=abs(old-val['old'])<1e-4 and abs(new-val['new'])<1e-4
        ok=ok and match
        print(f'   elem {e} bit {fl["bit"]}: old {old:.6f} vs {val["old"]:.6f}, new {new:.6f} vs {val["new"]:.6f}  {"OK" if match else "MISMATCH"}')
    man={'model':model_key,'tensor':tensor,'n_codes':int(orig.size),'bits':8,
         'region_byte_range':a['region']['byte_range_in_tensor'],'self_check_passed':bool(ok),
         'files':{'original':'codes/original_codes.u8.npy','permuted':'codes/permuted_codes.u8.npy',
                  'faulted':'codes/faulted_codes.u8.npy'},
         'note':'uint8 two-complement stored codes, row-major flat; region = codes[byte_start:byte_end]'}
    json.dump(man,open(os.path.join(HERE,model_dir,'bytes_manifest.json'),'w'),indent=1)
    assert ok,'self-check failed: materialized bytes do not reproduce the recorded flip'
    print(f'[{model_dir}] done, self-check PASSED -> {out}',flush=True)

if __name__=='__main__':
    dirs=[sys.argv[1]] if len(sys.argv)>1 else sorted(d for d in os.listdir(HERE)
          if os.path.isdir(os.path.join(HERE,d)) and os.path.exists(os.path.join(HERE,d,'attack.json')))
    for d in dirs: do(d)
