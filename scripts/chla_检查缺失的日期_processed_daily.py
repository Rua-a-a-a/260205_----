"""
扫描数据目录，检查缺失的日期文件
数据结构: YYYY/MM/YYYYMMDD.h5
最小单位：日级别文件
"""

import os
from datetime import datetime, timedelta

# 数据路径
DATA_DIR = "/data_new/chla_data_imputation_data_260125/chla_data_pretraining/daily_fusion_target_modified"
OUTPUT_FILE = "/home/ccy/260205_缺失填充/output/缺失日期/chla_缺失日期_processed_daily.txt"

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 获取所有年份文件夹（格式YYYY）
year_dirs = sorted([d for d in os.listdir(DATA_DIR) if d.isdigit() and len(d) == 4])

if not year_dirs:
    print("未找到年份文件夹")
    exit(1)

# 确定数据的起止日期
start_year = year_dirs[0]
end_year = year_dirs[-1]

# 找到第一个年份的第一个月
first_year_path = os.path.join(DATA_DIR, start_year)
first_months = sorted([m for m in os.listdir(first_year_path) if m.isdigit()])
start_month = first_months[0] if first_months else "01"

# 找到最后一个年份的最后一个月
last_year_path = os.path.join(DATA_DIR, end_year)
last_months = sorted([m for m in os.listdir(last_year_path) if m.isdigit()])
end_month = last_months[-1] if last_months else "12"

start_date = datetime.strptime(f"{start_year}{start_month}01", "%Y%m%d")
# 计算结束月份的最后一天
end_year_int = int(end_year)
end_month_int = int(end_month)
if end_month_int == 12:
    end_date = datetime(end_year_int + 1, 1, 1) - timedelta(days=1)
else:
    end_date = datetime(end_year_int, end_month_int + 1, 1) - timedelta(days=1)

print(f"数据范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")

# 收集所有存在的日期文件
existing_dates = set()

for year_dir in year_dirs:
    year_path = os.path.join(DATA_DIR, year_dir)
    if not os.path.isdir(year_path):
        continue

    # 遍历月份文件夹
    for month_dir in os.listdir(year_path):
        month_path = os.path.join(year_path, month_dir)
        if not os.path.isdir(month_path):
            continue

        # 检查该月份下的h5文件
        h5_files = [f for f in os.listdir(month_path) if f.endswith('.h5')]
        for h5_file in h5_files:
            # 从文件名提取日期（格式：YYYYMMDD.h5）
            date_str = h5_file.replace('.h5', '')
            if len(date_str) == 8 and date_str.isdigit():
                existing_dates.add(date_str)

print(f"找到 {len(existing_dates)} 个日期文件")

# 生成完整日期序列，找出缺失的日期文件
missing_dates = []
current_date = start_date

while current_date <= end_date:
    date_str = current_date.strftime("%Y%m%d")
    if date_str not in existing_dates:
        missing_dates.append(date_str)
    current_date += timedelta(days=1)

print(f"缺失日期文件数量: {len(missing_dates)}")

# 写入缺失的日期文件到文件
with open(OUTPUT_FILE, 'w') as f:
    for date in missing_dates:
        f.write(date + '\n')

print(f"缺失日期文件已写入: {OUTPUT_FILE}")

# 打印部分缺失日期文件示例
if missing_dates:
    print("\n缺失日期文件示例（前20个）:")
    for date in missing_dates[:20]:
        # 格式化显示：YYYY-MM-DD
        dt = datetime.strptime(date, "%Y%m%d")
        print(f"  {date} ({dt.strftime('%Y-%m-%d')})")
    if len(missing_dates) > 20:
        print(f"  ... 共 {len(missing_dates)} 个缺失日期文件")
