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
config.network = "r34"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.1  # batch size is 512

config.rec = "/storage_labs/3030/BelyakovM/Face_attributes/ds/db_BuevichP/emochon/OZON_splited/train_rec"
config.num_classes = 7
config.num_image = 32826
config.num_epoch = 40
config.warmup_epoch = -1
config.decay_epoch = [16,24,30,36]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
