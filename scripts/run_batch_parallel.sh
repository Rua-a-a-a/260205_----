#!/bin/bash
# 8进程并行处理2016-2024年数据
# 每个进程处理约1年的数据

# 进程1: 2016年
python preprocessing/sst_temporal_weighted_fill.py \
  --mode batch --year_start 2016 --year_end 2016 \
  --utc_hour 4 --workers 27 &

# 进程2: 2017年
python preprocessing/sst_temporal_weighted_fill.py \
  --mode batch --year_start 2017 --year_end 2017 \
  --utc_hour 4 --workers 27 &

# 进程3: 2018年
python preprocessing/sst_temporal_weighted_fill.py \
  --mode batch --year_start 2018 --year_end 2018 \
  --utc_hour 4 --workers 27 &

# 进程4: 2019年
python preprocessing/sst_temporal_weighted_fill.py \
  --mode batch --year_start 2019 --year_end 2019 \
  --utc_hour 4 --workers 27 &

# 进程5: 2020年
python preprocessing/sst_temporal_weighted_fill.py \
  --mode batch --year_start 2020 --year_end 2020 \
  --utc_hour 4 --workers 27 &

# 进程6: 2021年
python preprocessing/sst_temporal_weighted_fill.py \
  --mode batch --year_start 2021 --year_end 2021 \
  --utc_hour 4 --workers 27 &

# 进程7: 2022年
python preprocessing/sst_temporal_weighted_fill.py \
  --mode batch --year_start 2022 --year_end 2022 \
  --utc_hour 4 --workers 27 &

# 进程8: 2023-2024年
python preprocessing/sst_temporal_weighted_fill.py \
  --mode batch --year_start 2023 --year_end 2024 \
  --utc_hour 4 --workers 27 &

# 等待所有进程完成
wait

echo "所有年份处理完成！"
