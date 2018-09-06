"""Util functions
"""
import json
from datetime import datetime

import torch


def write_event(log, step: int, **data):
    data['step'] = step
    data['time'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def save_model(model, epoch, step, model_path):
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'step': step,
    }, str(model_path))


def dump_configs(args):
    pass
