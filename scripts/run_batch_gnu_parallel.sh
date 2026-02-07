#!/bin/bash
# 使用GNU Parallel并行处理每个年份
# 8个并行任务，自动负载均衡

# 生成年份列表
years=(2016 2017 2018 2019 2020 2021 2022 2023 2024)

# 使用parallel并行处理，最多8个任务
export WORKERS_PER_JOB=27  # 216/8 = 27

parallel -j 8 \
  "python preprocessing/sst_temporal_weighted_fill.py \
    --mode batch \
    --year_start {} \
    --year_end {} \
    --utc_hour 4 \
    --workers ${WORKERS_PER_JOB}" \
  ::: "${years[@]}"

echo "所有年份处理完成！"
