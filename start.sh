#!/bin/bash
docker run --gpus all --rm -v /mnt/d/ai:/models \
-v $PWD:/home/user/app \
-it stablelm