#!/usr/bin/env python3
"""Run the archived trainer with explicit, identity-tracked maintained adapters."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
BINDING_ENV = 'OPENWAM_LONGTRAIN_CODE_DIR'


def configure_code(code_dir):
    """Select the no-argument metadata import binding, preserving explicit guards."""
    sys.path.insert(0, str(ROOT / 'runtime'))
    import data_pipeline
    if Path(data_pipeline.__file__).resolve().parent != ROOT / 'runtime':
        raise RuntimeError('another data_pipeline was imported before repository entry')
    code_dir = Path(code_dir).resolve()
    data_pipeline.DEFAULT_CODE = code_dir
    # The original helper captured DEFAULT_CODE in a function default. Changing
    # only the module constant leaves metadata's dp.bind_native() using old code.
    data_pipeline.bind_native.__defaults__ = (code_dir,)
    os.environ[BINDING_ENV] = str(code_dir)
    return data_pipeline


def load_trainer():
    configure_code(ROOT.parents[1])
    sys.path.insert(0, str(ROOT / 'runtime'))
    import distributed16
    if Path(distributed16.__file__).resolve().parent != ROOT / 'runtime':
        raise RuntimeError('another distributed16 was imported before repository entry')
    distributed16.REVIEWED = ROOT / 'reviewed'
    import trainer
    # Preserve the production default diagnostic directory. The helper source
    # location changes, while data locations still come from explicit config.
    original_reviewed = Path(json.loads((ROOT / 'snapshot.json').read_text())['reviewed_path'])
    trainer.DEFAULT_CSTAR = str(original_reviewed / 'baseline_cstar/data')
    install_goal_generation(trainer)
    return trainer


def install_goal_generation(trainer):
    """Leave archived helper bytes intact; extend the strict run identity."""
    if getattr(trainer, '_goal_generation_adapter', False):
        return
    base = trainer.d.load_reviewed().base
    def generate(model, batch, *, arm, seed, index, steps=20):
        # Imports follow the code directory selected by the config/bootstrap.
        from openwam.train.goal_generation import generate_goal_latents
        return generate_goal_latents(model, batch, common_noise=base.common_noise,
                                     arm=arm, seed=seed, index=index, steps=steps)
    base.generate_latents = generate
    original_identity = trainer.runner_identity
    def identity():
        from openwam.train import goal_generation
        return {**original_identity(), 'entrypoint.py': trainer.d.sha256(Path(__file__)),
                'adapters/goal_generation.py': trainer.d.sha256(Path(goal_generation.__file__))}
    trainer.runner_identity = identity
    trainer._goal_generation_adapter = True


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'audit':
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument('--code-dir', default=str(ROOT.parents[1]))
        options, _ = probe.parse_known_args(argv[1:])
        provider = configure_code(options.code_dir)
        previous = sys.argv
        try:
            sys.argv = [__file__, *argv]
            provider.main()
        finally:
            sys.argv = previous
        return 0
    trainer = load_trainer()
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config')
    options, _ = probe.parse_known_args(argv)
    if options.config:
        config = trainer.read_config(options.config)
        configure_code(config.get('code_dir', trainer.PINNED))
    return trainer.main(argv)


# DataLoader spawn imports the main module before unpickling the metadata
# reader, before worker_init. Rebind in that interpreter as well as the parent.
if __name__ == '__mp_main__' and os.environ.get(BINDING_ENV):
    configure_code(os.environ[BINDING_ENV])


if __name__ == '__main__':
    raise SystemExit(main())
