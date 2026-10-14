"""Preserve periodic saves and always save the final successful training epoch."""
from functools import wraps
import inspect
from pathlib import Path


def with_final_checkpoint(trainer, make_state, save):
    signature = inspect.signature(trainer)

    @wraps(trainer)
    def train(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments
        optimizer = options['optimizer']
        iterations = options.get('start_iter',0)
        step_original = optimizer.step

        def step(*values, **settings):
            nonlocal iterations
            result = step_original(*values, **settings)
            iterations += 1
            return result

        optimizer.step = step
        try:
            result = trainer(*args, **kwargs)
        finally:
            optimizer.step = step_original
        epoch = options['total_epochs']
        final = Path(options['ckpt_save_dir'])/f'checkpoint_epoch_{epoch}'
        if options.get('rank',0)==0 and not final.with_suffix('.pth').exists():
            save(make_state(options['model'],optimizer,epoch,iterations),filename=str(final))
        return result

    return train
