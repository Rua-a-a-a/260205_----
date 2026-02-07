# SST时间加权填充脚本使用说明

## 功能概述

该脚本用于批量生成2016-2024年的SST（海表温度）数据，使用时间加权填充算法减少云遮挡导致的缺失值。

## 主要修改

1. **输出格式变更**：从单个HDF5文件改为按日期分层存储
   - 旧格式：`jaxa_weighted_series_04.h5`
   - 新格式：`YYYY/MM/YYYYMMDD.h5`

2. **固定UTC时间**：所有数据固定为UTC 4点采样

3. **批量年份处理**：支持一次性处理多年数据（2016-2024）

4. **输出目录**：`/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion/`

## 命令行参数

### 批量模式（推荐）

生成2016-2024年所有数据：

```bash
python preprocessing/sst_temporal_weighted_fill.py \
  --mode batch \
  --year_start 2016 \
  --year_end 2024 \
  --utc_hour 4 \
  --workers 216
```

### 单月模式

生成单个月份数据（例如2024年1月）：

```bash
python preprocessing/sst_temporal_weighted_fill.py \
  --mode single \
  --year 2024 \
  --month 1 \
  --utc_hour 4 \
  --workers 216
```

### 测试模式

运行小规模测试（72小时）：

```bash
python preprocessing/sst_temporal_weighted_fill.py \
  --mode test \
  --workers 32
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | `batch` | 执行模式：`batch`(批量年份), `single`(单月), `test`(测试) |
| `--year_start` | int | `2016` | 批量模式起始年份 |
| `--year_end` | int | `2024` | 批量模式结束年份 |
| `--year` | int | `2024` | 单月模式年份 |
| `--month` | int | `1` | 单月模式月份(1-12) |
| `--utc_hour` | int | `4` | 固定UTC时间(0-23) |
| `--workers` | int | `216` | 并行CPU核心数 |

## 输出文件结构

### 目录结构

```
/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion/
├── 2016/
│   ├── 01/
│   │   ├── 20160101.h5
│   │   ├── 20160102.h5
│   │   └── ...
│   ├── 02/
│   └── ...
├── 2017/
├── ...
└── 2024/
    ├── 01/
    │   ├── 20240101.h5
    │   ├── 20240102.h5
    │   └── ...
    └── ...
```

### HDF5文件内容

每个文件包含单日数据：

```python
{
    'sst_data': (H, W) float32,      # SST数据（开尔文）
    'missing_mask': (H, W) uint8,    # 缺失掩码（1=缺失，0=有效）
    'fill_mask': (H, W) uint8,       # 填充掩码（1=已填充，0=原始）
    'latitude': (H,) float32,        # 纬度
    'longitude': (W,) float32,       # 经度
    'timestamp': (1,) S32,           # ISO格式时间戳

    # 属性
    'series_id': 4,                  # UTC小时
    'utc_hour': 4,                   # UTC时间
    'date': '2024-01-01',           # 日期
    'shape': (451, 351),            # 空间分辨率
    'creation_date': '...'          # 创建时间
}
```

## 算法说明

### 时间加权填充

对于每个缺失像素，使用历史观测的加权平均：

```
权重(t_history) = 1 / (t_target - t_history)
填充值 = Σ(w_i × v_i) / Σ(w_i)
```

### 回溯窗口策略

1. 首先尝试24小时回溯窗口
2. 如果历史帧数 < 10，扩展到48小时窗口

## 性能优化

- **多进程并行**：使用CPU多进程处理缺失像素填充
- **内存预加载**：历史帧预加载到内存，避免重复I/O
- **分块处理**：自动计算最优chunk大小

## 预期输出

### 批量模式（2016-2024）

- **总月份数**：108个月（9年 × 12月）
- **总文件数**：约3240个文件（108月 × 30天）
- **预计总大小**：约100-200 GB（取决于压缩率）
- **处理时间**：取决于CPU核心数和I/O速度

### 数据质量

- **平均缺失率改善**：约20-30%
- **填充率**：约25%的像素被算法填充
- **缺失率范围**：16%-80%（取决于云覆盖情况）

## 统计文件

批量模式会生成统计JSON文件：

```
batch_statistics_2016_2024_utc04.json
```

包含每月的处理统计信息。

## 注意事项

1. **磁盘空间**：确保输出目录有足够空间（建议200GB+）
2. **内存需求**：建议32GB+内存
3. **CPU核心数**：根据实际CPU核心数调整`--workers`参数
4. **数据源**：确保JAXA数据源路径正确且数据完整
5. **UTC时间**：当前固定为4点，如需其他时间可修改`--utc_hour`参数

## 故障排查

### 文件加载失败

如果出现"Failed to load"警告，检查：
- JAXA数据源路径是否正确
- 对应日期的NetCDF文件是否存在
- 文件权限是否正确

### 内存不足

如果出现内存错误：
- 减少`--workers`参数
- 使用单月模式分批处理

### 处理速度慢

优化建议：
- 增加`--workers`参数（不超过CPU核心数）
- 使用SSD存储提高I/O速度
- 检查是否有其他进程占用CPU

## 示例工作流

### 完整处理流程

```bash
# 1. 批量生成2016-2024年数据
python preprocessing/sst_temporal_weighted_fill.py \
  --mode batch \
  --year_start 2016 \
  --year_end 2024 \
  --utc_hour 4 \
  --workers 216

# 2. 验证输出文件
ls -lh /data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion/2024/01/

# 3. 检查统计信息
cat /data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion/batch_statistics_2016_2024_utc04.json
```

### 增量处理

如果需要补充某个月份：

```bash
python preprocessing/sst_temporal_weighted_fill.py \
  --mode single \
  --year 2024 \
  --month 6 \
  --utc_hour 4 \
  --workers 216
```
