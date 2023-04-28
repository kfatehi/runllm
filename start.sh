#!/bin/bash
docker run --gpus all --rm -v /mnt/d/ai:/models \
-e TRANSFORMERS_CACHE=/models \
-it stablelm