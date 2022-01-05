#!/usr/bin/env bash

set -x
NGPUS=$1
# PY_ARGS=${@:2}

# python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/kitti_models/PartA2_free.yaml --pretrained_model /home/u0610862/cruw_collector/models/PartA2_free_7872.pth

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/kitti_models/pv_rcnn.yaml --pretrained_model /home/u0610862/cruw_collector/models/pv_rcnn_8369.pth
