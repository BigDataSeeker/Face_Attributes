# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "cosface"
config.network = "r18"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 63
config.lr = 0.025  # batch size is 512

config.rec = "/storage_labs/3030/BelyakovM/Face_attributes/ds/Glint360k"
config.num_classes = 7
config.num_image = 74837
config.num_epoch = 20
config.warmup_epoch = -1
config.decay_epoch = [4, 8, 12, 16]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
