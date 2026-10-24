"""Derive the shared Stage12 selection without modifying source data or YAML.

The sole default exclusion was explicitly authorized on 2026-10-24. Original
historical curation audit fields are retained; curation.stage12_derived_selection
contains effective counts and the exact excluded ranges. No other repository is
dropped, and this command never substitutes a different recording or episode.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import yaml

EXCLUDED_REPO='/mnt/data/cpfs/b5/post_train_data/wipe_table/wipe_table_v1_test'


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def prepare_selection(original_yaml,output_yaml,*,exclusions_path=None):
    source=Path(original_yaml).resolve();output=Path(output_yaml).resolve()
    log=Path(exclusions_path).resolve() if exclusions_path else output.parent/'exclusions.json'
    if source in (output,log) or output==log:raise ValueError('refusing to overwrite source or overlap output paths')
    with source.open() as stream:document=yaml.load(stream,Loader=getattr(yaml,'CSafeLoader',yaml.SafeLoader))
    selections=[]
    def walk(node):
        if isinstance(node,dict):
            if 'segments_by_repo' in node:selections.append(node)
            for child in node.values():walk(child)
        elif isinstance(node,list):
            for child in node:walk(child)
    walk(document)
    if len(selections)!=1:raise ValueError('expected exactly one segments_by_repo selection')
    resolver=selections[0]
    if EXCLUDED_REPO not in resolver['segments_by_repo']:raise ValueError(f'authorized excluded repository not selected: {EXCLUDED_REPO}')
    ranges=resolver['segments_by_repo'].pop(EXCLUDED_REPO)
    weight=resolver['repo_weights'].pop(EXCLUDED_REPO)
    weights=resolver['repo_weights']
    if any(not math.isfinite(v) or v<0 for v in weights.values()) or sum(weights.values())<=0:raise ValueError('invalid remaining repository weights')
    total=sum(weights.values());resolver['repo_weights']={k:v/total for k,v in weights.items()}
    exclusion=dict(repo=EXCLUDED_REPO,reason='user explicitly authorized exclusion of missing original path on 2026-10-24',
        segments=len(ranges),frames=sum(b-a for _,a,b in ranges),ranges=ranges,original_weight=weight)
    summary=dict(original_source_yaml=str(source),original_source_yaml_sha256=_sha(source),exclusions=[exclusion],
        selected_repositories=len(resolver['segments_by_repo']),selected_segments=sum(len(v) for v in resolver['segments_by_repo'].values()),
        selected_frames=sum(b-a for rr in resolver['segments_by_repo'].values() for _,a,b in rr),
        weights_renormalized=True,source_datasets_modified=False)
    document.setdefault('curation',{})['stage12_derived_selection']=dict(summary)
    output.parent.mkdir(parents=True,exist_ok=True);log.parent.mkdir(parents=True,exist_ok=True)
    header='# Derived Stage1/Stage2 selection: user-authorized missing-path exclusion; original source unchanged.\n'
    output.write_text(header+yaml.dump(document,Dumper=getattr(yaml,'CSafeDumper',yaml.SafeDumper),allow_unicode=True,sort_keys=False))
    summary.update(derived_source_yaml=str(output),derived_source_yaml_sha256=_sha(output))
    log.write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-yaml',required=True)
    parser.add_argument('--output-yaml',required=True)
    parser.add_argument('--exclusions-json')
    args=parser.parse_args();result=prepare_selection(args.original_yaml,args.output_yaml,exclusions_path=args.exclusions_json)
    print(json.dumps({k:v for k,v in result.items() if k!='exclusions'},indent=2))


if __name__=='__main__':main()
