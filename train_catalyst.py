from collections import defaultdict

from torch.optim.lr_scheduler import LambdaLR, OneCycleLR, ReduceLROnPlateau


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(config, model,):
    from functools import partial
    from torch.optim import Adam, AdamW, SGD

    OPTIMS = {'adam': partial(Adam, weight_decay=config['weight_decay']),
              'sgd': partial(SGD, momentum=config['momentum']),
              'adamw': partial(AdamW, weight_decay=config['weight_decay'])}


    adapt_params = defaultdict(list)
    adapt_params['lr'] = config['adapt_lr']
    main_params = defaultdict(list)
    main_params['lr'] = config['main_lr']

    for name, param in model.named_parameters():
        if 'adapt' in name:
            # param.requires_grad = bool(adapt_params['lr'])
            adapt_params['params'].append(param)
            #adapt_params['names'].append(name)
        else:
            main_params['params'].append(param)
            #main_params['names'].append(name)

    optimizer = OPTIMS[config['name']](params=[adapt_params, main_params])
    # optimizer = OPTIMS[config['name']](params=model.parameters())

    return optimizer


def get_lr_schedule(config, optimizer):
    schedule = config['schedule'].lower()
    print(f'{schedule} LR schedule')

    if schedule == 'linear_warmup':
        assert 'bs' in config.keys(), 'Batch size "bs" is missing from config dict'
        scheduler = LambdaLR(optimizer, lambda epoch: min(epoch / (config['warmup_period'] / config['bs']), 1.0))

    elif schedule == 'onecycle':
        assert 'epochs' in config.keys(), 'Number of epochs missing from config dict'
        assert 'steps_per_epoch' in config.keys(), '"steps_per_epoch" missing from config dict'
        lrs = [item['lr'] for item in optimizer.state_dict()['param_groups']]
        scheduler = OneCycleLR(optimizer, max_lr=lrs, epochs=config['epochs'],
                               steps_per_epoch=config['steps_per_epoch'], pct_start=config['pct_start'])
    elif schedule == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer)
    else:
        scheduler = LambdaLR(optimizer, lambda epoch: 1.)
        print('Constant LR. No schedule')

    return scheduler

