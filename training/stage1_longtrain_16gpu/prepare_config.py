#!/usr/bin/env python3
"""Create a separate run configuration; never edit the production snapshot."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prepare(output_dir, *, code_dir, prepared_g=None, prepared_cstar=None):
    code_dir = Path(code_dir).resolve()
    if not (code_dir / 'openwam/model/goal/goal_memory.py').is_file():
        raise ValueError('code_dir must contain the native G model')
    output_dir = Path(output_dir).resolve()
    config = json.loads((ROOT / 'runtime/G.json').read_text())
    model = json.loads((ROOT / 'runtime/G_model.json').read_text())
    assets = json.loads((ROOT / 'external_assets.json').read_text())['files']
    config.update(code_dir=str(code_dir), model_config=str(output_dir / 'G_model.json'),
                  prepared_g=str(Path(prepared_g).resolve() if prepared_g
                                 else Path(assets['G_diagnostic_manifest']['path']).parent),
                  prepared_cstar=str(Path(prepared_cstar).resolve() if prepared_cstar
                                     else Path(assets['Cstar_diagnostic_manifest']['path']).parent))
    if prepared_g is not None:
        config['data_pipeline']['validation_g'] = config['prepared_g']
    if prepared_cstar is not None:
        config['data_pipeline']['validation_cstar'] = config['prepared_cstar']
    # Refuse to overwrite any prior configuration or an active run directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, value in [('G_model.json', model), ('G.json', config)]:
        (output_dir / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    return output_dir / 'G.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--code-dir', type=Path, default=ROOT.parents[1])
    parser.add_argument('--prepared-g', type=Path)
    parser.add_argument('--prepared-cstar', type=Path)
    args = parser.parse_args()
    print(prepare(args.output_dir, code_dir=args.code_dir,
                  prepared_g=args.prepared_g, prepared_cstar=args.prepared_cstar))


if __name__ == '__main__':
    main()
