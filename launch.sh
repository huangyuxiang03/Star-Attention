# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_PATH="/home/test/test01/hyx/Llama-3.1-8B-Instruct"


# MODEL_NAME="Llama-3.2-1B-Instruct"
# MODEL_PATH="/home/test/test01/hyx/Llama-3.2-1B-instruct"

BENCHMARK="ruler"

# Benchmark config setup
# ======================

if [[ $BENCHMARK == "ruler" ]]; then

    PROMPT_CONFIG="llama3" # Select from: ruler/data/template.py
    SCRIPT="run_ruler.py"

elif [[ $BENCHMARK == "babilong" ]]; then

    PROMPT_CONFIG="llama3" # Select from: babilong/template.py
    SCRIPT="run_babilong.py"

else
    echo "Invalid Benchmark: ${BENCHMARK}"
    exit 1
fi

# Launch Evaluation
# =================

## The `-np` config shared below is tested for the Llama-3.1-8B-Instruct model on 8 A100 GPUs

# Global Attention
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a dense -pc $PROMPT_CONFIG -l 16384
python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a ring -pc $PROMPT_CONFIG -l 16384 -np 4
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a dense -pc $PROMPT_CONFIG -l 16384 32768 65536
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a ring -pc $PROMPT_CONFIG -l 131072 -np 8

# Star Attention
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a star -pc $PROMPT_CONFIG -bs 4096 -l 16384 -np 4
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a star -pc $PROMPT_CONFIG -bs 8192 -l 32768 -np 4
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a star -pc $PROMPT_CONFIG -bs 16384 -l 65536 -np 4
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a star -pc $PROMPT_CONFIG -bs 16384 -l 131072 -np 8
