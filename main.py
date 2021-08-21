import utility
import data
import model
import loss
from option import args
from checkpoint import Checkpoint
from trainer import Trainer
import sys

utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    print('args:', args)
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    aloss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, aloss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
    checkpoint.done()

