"""Load an LLM, evaluate MMLU accuracy on a minimal panel, and compute attack success (ASR).

Reuses the frozen harness end to end (quantization, permutation, MMLU few-shot scoring) so numbers
match the campaign.  A trial SUCCEEDS if faulted accuracy <= 0.5 x the fault-free accuracy (the
paper's >=50% relative-drop criterion); ASR = successes / trials.

  selected : inject the recorded search-selected flips -> one deterministic trial (IV-B upper bound)
  sample   : --trials random in-range flip sets inside the region -> hardware-realistic ASR (IV-C)
  codes    : load attacked-tensor bytes from --codes <u8.npy> (your own post-fault bytes)

Usage (CPU ok):
  PYTHONPATH=/gpfs/work2/0/prjs1717/xwang/BFA_llm \
  REGIONALIGN_HF_HOME=/gpfs/work2/0/prjs1717/xwang/huggingface_cache \
  python run_asr_mmlu.py qwen2.5-1.5b__mmlu --layout permuted --mode selected --n 48
Needs the MMLU Arrow in REGIONALIGN_HF_HOME (the campaign's dev panel); --n picks the first N dev rows.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find the bundled regionalign/
import numpy as np, torch
from regionalign.harness import core as C
from regionalign.experiments import llm as L
from regionalign.experiments.clean_mmlu import load_model

HERE=os.path.dirname(os.path.abspath(__file__))
HFHUB=os.path.join(os.environ.get('REGIONALIGN_HF_HOME','/gpfs/work2/0/prjs1717/xwang/huggingface_cache'),'hub')
HFHOME=os.environ.get('REGIONALIGN_HF_HOME','/gpfs/work2/0/prjs1717/xwang/huggingface_cache')
CKPT={'qwen2.5-1.5b':'models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323',
      'llama3.2-1b':'models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08',
      'stablelm2-1.6b':'models--stabilityai--stablelm-2-1_6b/snapshots/f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef',
      'smollm2-1.7b':'models--HuggingFaceTB--SmolLM2-1.7B/snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71'}

class RT:
    residual=None
    def __init__(self,lm): self.lm=lm

def accuracy(rt, rows, bs):
    return L._task_metrics(rt, rows, 'mmlu', bs)['accuracy']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('model_dir'); ap.add_argument('--layout',choices=['original','permuted'],default='permuted')
    ap.add_argument('--mode',choices=['selected','sample','codes'],default='selected')
    ap.add_argument('--n',type=int,default=48); ap.add_argument('--bs',type=int,default=8)
    ap.add_argument('--trials',type=int,default=100); ap.add_argument('--k',type=int,default=None)
    ap.add_argument('--codes',default=None); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--device',default='cpu')
    a=ap.parse_args()
    md=a.model_dir; att=json.load(open(f'{HERE}/{md}/attack.json')); key=[k for k in CKPT if k in md][0]
    tensor=att['region']['tensor']; dev=a.device
    model,tok,_,_=load_model(os.path.join(HFHUB,CKPT[key]),'int8',dev)
    frozen=f'{HERE}/samples/mmlu_min_{key}.json'
    if os.path.exists(frozen):
        rows=json.load(open(frozen))['rows'][:a.n]          # self-contained, no MMLU dataset needed
    else:
        rows=L._mmlu_rows(tok,HFHOME)[0]['dev'][:a.n]        # fallback: rebuild from the MMLU Arrow
    rt=L.FormalTextRuntime(model,tok,{'dev':rows},dev)
    if a.layout=='permuted':
        C.apply_permutation(RT(model),json.load(open(f'{HERE}/{md}/permutation.json')))
    clean_acc=accuracy(rt,rows,a.bs); thr=0.5*clean_acc
    W=model.get_parameter(tensor); n_in=W.shape[1]; Wflat=W.data.view(-1); bounds=C.row_absmax(W)
    rs=att['region']['page']*att['region']['page_bytes']; region=list(range(rs,rs+att['region']['page_bytes']))
    rep={'model_dir':md,'task':'mmlu','layout':a.layout,'mode':a.mode,'clean_accuracy':clean_acc,
         'success_threshold_accuracy':thr,'n_rows':len(rows)}

    def trial(flips):
        saved={e:float(Wflat[e]) for e,_ in flips}; applied=0
        for e,b in flips:
            code=int(C.stored_codes(model,tensor,torch.tensor([e]))[0]); nc=code^(1<<b)
            val=C.coded_values(model,tensor,torch.tensor([e]),torch.tensor([nc]))[0]
            if bool(C.value_allowed(val.float(),bounds[e//n_in])): Wflat[e]=val.to(Wflat.dtype); applied+=1
        acc=accuracy(rt,rows,a.bs)
        for e,v in saved.items(): Wflat[e]=v
        return acc,applied

    if a.mode=='selected':
        acc,applied=trial([(f['code_index_in_tensor'],f['bit']) for f in att['flips']])
        s=acc<=thr; rep.update(faulted_accuracy=acc,applied_flips=applied,success=bool(s),asr=1.0 if s else 0.0)
        print(f"[{md}] {a.layout} selected: clean acc {clean_acc:.3f} -> faulted {acc:.3f} (thr {thr:.3f}) flips={applied} {'SUCCESS' if s else 'fail'}")
    elif a.mode=='codes':
        u8=np.load(a.codes); saved={e:float(Wflat[e]) for e in region}
        vals=C.coded_values(model,tensor,torch.as_tensor(region),torch.as_tensor([int(u8[e]) for e in region]))
        Wflat[torch.as_tensor(region)]=vals.to(Wflat.dtype)
        acc=accuracy(rt,rows,a.bs)
        for e,v in saved.items(): Wflat[e]=v
        s=acc<=thr; rep.update(faulted_accuracy=acc,codes=a.codes,success=bool(s),asr=1.0 if s else 0.0)
        print(f"[{md}] {a.layout} codes: clean {clean_acc:.3f} -> {acc:.3f} (thr {thr:.3f}) {'SUCCESS' if s else 'fail'}")
    else:
        k=a.k or att['n_flips']; rng=np.random.default_rng(a.seed); ns=0
        for _ in range(a.trials):
            acc,_=trial([(int(rng.choice(region)),int(rng.integers(0,8))) for _ in range(k)])
            ns+= acc<=thr
        rep.update(k_per_trial=k,trials=a.trials,successes=int(ns),asr=ns/a.trials)
        print(f"[{md}] {a.layout} sample k={k} x{a.trials}: clean acc {clean_acc:.3f}  ASR={ns}/{a.trials}={ns/a.trials:.3f}")
    json.dump(rep,open(f'{HERE}/{md}/asr_mmlu_{a.mode}_{a.layout}.json','w'),indent=1)

if __name__=='__main__': main()
