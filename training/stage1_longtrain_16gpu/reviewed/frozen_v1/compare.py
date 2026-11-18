#!/usr/bin/env python3
"""Prepare once, train matched native RFC arms, and report partial progress."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare',help='CPU source audit then frozen real VAE/T5 shared snapshots')
    p.add_argument('--code-dir',required=True);p.add_argument('--base-config',required=True)
    p.add_argument('--train-source',required=True);p.add_argument('--val-source',required=True)
    p.add_argument('--output',required=True);p.add_argument('--train-count',type=int,default=256)
    p.add_argument('--val-count',type=int,default=64);p.add_argument('--device',default='cuda')
    p.add_argument('--dtype',choices=('float32','bfloat16'),default='bfloat16')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--vae-batch-size',type=int,default=4)
    p.add_argument('--latent-cache-mb',type=int,default=512)
    p=sub.add_parser('train',help='Native DiT + CPU FP32 master AdamW; no T5 residency')
    p.add_argument('--code-dir',required=True);p.add_argument('--data',required=True)
    p.add_argument('--arm',choices=('default','no_history','no_g'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda');p.add_argument('--dtype',choices=('float32','bfloat16'),default='bfloat16')
    p.add_argument('--eval-every',type=int,default=100);p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    p.add_argument('--visual-count',type=int,default=4);p.add_argument('--sampling-steps',type=int,default=20)
    p=sub.add_parser('report',help='JSON/standalone HTML over any available partial arms')
    p.add_argument('--runs-root',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.mode!='report':
        sys.path.insert(0,str(Path(args.code_dir).resolve()))
    import rfc_compare as runner
    if args.mode=='prepare': result=runner.prepare(args)
    elif args.mode=='train':
        if min(args.updates,args.batch_size,args.eval_every,args.checkpoint_every,args.sampling_steps)<1:
            parser.error('updates, batch size, intervals, sampling steps must be positive')
        if args.visual_count<0: parser.error('visual count must be nonnegative')
        result=runner.train(args)
    else:
        result=runner.report(args.runs_root,args.output)
        result=dict(output=args.output,arms=list(result['arms']))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
