"""Continue verified 10M weights, new optimizer, bounded balanced curriculum."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import zipfile

import torch
from curriculum_data import TASKS, build_splits, digest
from language_candidate import load_candidate, save_candidate
from pilot_10m import batch, evaluate, generate, dataset as retired_dataset

PARENT_WEIGHT_SHA='84ebd4ef85c080cb4ef4ae37bc38d9e94ec2b4460dc19fd6b320654251f7f83e'


def restore_parent(source, destination):
    source=Path(source)
    transport=json.loads((source/'bundle-sha256.json').read_text())
    combined=b''.join((source/f'bundle.part{i:02d}').read_bytes() for i in range(transport['parts']))
    if len(combined)!=transport['bytes'] or hashlib.sha256(combined).hexdigest()!=transport['sha256']:
        raise ValueError('parent transport hash mismatch')
    import io
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(combined)) as archive:
        manifest=json.loads(archive.read('checkpoint/manifest.json'))
        weights=archive.read('checkpoint/weights.pt')
        if manifest['weights_sha256']!=PARENT_WEIGHT_SHA or hashlib.sha256(weights).hexdigest()!=PARENT_WEIGHT_SHA:
            raise ValueError('wrong parent weights')
        (destination/'manifest.json').write_text(json.dumps(manifest))
        (destination/'weights.pt').write_bytes(weights)
    return load_candidate(destination)


def run(source,output):
    available=os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    if available<2_500_000_000 or shutil.disk_usage(output.parent).free<1_000_000_000:
        raise RuntimeError('insufficient pilot memory/disk')
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(1097)
    torch.set_num_threads(min(2,os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    model=restore_parent(source,output/'parent-checkpoint')
    count=sum(p.numel() for p in model.parameters())
    if count!=9_979_914:
        raise ValueError('wrong model scale')
    splits=build_splits()
    groups={task:[r for r in splits['train'] if r['task']==task] for task in TASKS}
    initial_validation=evaluate(model,splits['validation'])
    rng=random.Random(1097)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    history=[]
    model.train()
    started=time.monotonic()
    for step in range(2400):
        # Exactly one example from each task; normalize answer loss per example.
        rows=[rng.choice(groups[task]) for task in TASKS]
        x,y=batch(rows)
        optimizer.zero_grad(set_to_none=True)
        logits,_=model(x)
        losses=torch.nn.functional.cross_entropy(logits.transpose(1,2),y,reduction='none')
        mask=y!=-100
        loss=(losses.sum(1)/mask.sum(1)).mean()
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step==0 or (step+1)%100==0:
            record={'step':step+1,'balanced_loss':loss.item(),'gradient_norm':norm.item()}
            history.append(record)
            print(json.dumps(record),flush=True)
        if time.monotonic()-started>=480:
            break
    completed=step+1
    elapsed=time.monotonic()-started
    del optimizer
    manifest=save_candidate(model,output/'checkpoint')
    restored=load_candidate(output/'checkpoint')
    x,_=batch(splits['validation'][:1])
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0],restored(x)[0],rtol=0,atol=0)
    del restored
    final={k:evaluate(model,rows) for k,rows in splits.items() if k!='train'}
    selected=[]
    for task in TASKS:
        unique={r['prompt']:r for r in splits['test'] if r['task']==task}
        selected.extend(list(unique.values())[:8])
    candidate_generation=generate(model,selected)
    # First evaluation of new test happens after all updates; no checkpoint selection.
    parent=load_candidate(output/'parent-checkpoint')
    parent_test=evaluate(parent,splits['test'])
    parent_generation=generate(parent,selected)
    retired_generation=generate(model,retired_dataset()['test'][:8])
    exact=lambda rows:sum(r['exact'] for r in rows)
    report={'schema':'VN97ASC-CURRICULUM1','source_commit':os.environ.get('GITHUB_SHA'),
            'parent_run':36453857564,'parent_weights_sha256':PARENT_WEIGHT_SHA,
            'parameters':count,'seed':1097,'torch_version':str(torch.__version__),
            'optimizer_restarted':True,'steps_requested':2400,'steps_completed':completed,
            'training_seconds':elapsed,'stop_reason':'step_budget' if completed==2400 else 'wall_budget',
            'learning_rate':.0003,'batch_size':4,'balanced_per_example_loss':True,
            'initial_validation':initial_validation,'final':final,'parent_new_test':parent_test,
            'new_test_generation':candidate_generation,'parent_new_test_generation':parent_generation,
            'new_test_exact':exact(candidate_generation),'parent_new_test_exact':exact(parent_generation),
            'new_test_cases':len(selected),'retired_test_generation':retired_generation,
            'retired_test_exact':exact(retired_generation),'history':history,
            'splits':{k:{'records':len(v),'unique_prompts':len({r['prompt'] for r in v}),'sha256':digest(v)} for k,v in splits.items()},
            'checkpoint':manifest,'production_activation_authorized':False,
            'limitations':['synthetic task data and one seed; no AGI/general-language conclusion',
                           'new prompt templates/cases held out but task rules and labels shared',
                           'classification repeats four labels; record count is not unique semantic diversity',
                           'no architecture-controlled ablation or physical mobile evidence',
                           'weights saved; optimizer/RNG not saved for exact resume']}
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True))
    (output/'dataset.json').write_text(json.dumps(splits,ensure_ascii=False,indent=2))
    bundle=output/'VN97-10M-curriculum.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in ('report.json','dataset.json','checkpoint/manifest.json','checkpoint/weights.pt'):
            archive.write(output/name,name)
    with bundle.open('rb') as handle:
        parts=0
        while block:=handle.read(20*1024*1024):
            (output/f'bundle.part{parts:02d}').write_bytes(block)
            parts+=1
    (output/'bundle-sha256.json').write_text(json.dumps({'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'parts':parts,'bytes':bundle.stat().st_size}))
    print(json.dumps({'steps':completed,'seconds':elapsed,'parameters':count,'parent_exact':report['parent_new_test_exact'],
                      'candidate_exact':report['new_test_exact'],'cases':len(selected),'retired_exact':report['retired_test_exact'],
                      'test':final['test']['all']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.output)
