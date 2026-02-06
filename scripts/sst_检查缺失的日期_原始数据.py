"""
扫描数据目录，检查缺失的小时文件
数据结构: YYYYMM/DD/YYYYMMDDHHMMSS.nc
每天应有24个小时文件（00:00-23:00）
"""

import os
from datetime import datetime, timedelta

# 数据路径
DATA_DIR = "/data_new/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3"
OUTPUT_FILE = "/home/ccy/260205_缺失填充/output/缺失日期/sst_缺失日期_原始数据_小时级.txt"

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 获取所有年月文件夹（格式YYYYMM）
year_month_dirs = sorted([d for d in os.listdir(DATA_DIR) if d.isdigit() and len(d) == 6])

if not year_month_dirs:
    print("未找到年月文件夹")
    exit(1)

# 确定数据的起止日期
start_ym = year_month_dirs[0]
end_ym = year_month_dirs[-1]

start_date = datetime.strptime(start_ym + "01", "%Y%m%d")
# 计算结束月份的最后一天
end_year = int(end_ym[:4])
end_month = int(end_ym[4:6])
if end_month == 12:
    end_date = datetime(end_year + 1, 1, 1) - timedelta(days=1)
else:
    end_date = datetime(end_year, end_month + 1, 1) - timedelta(days=1)

print(f"数据范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")

# 收集所有存在的小时文件
existing_hours = set()

for ym_dir in year_month_dirs:
    ym_path = os.path.join(DATA_DIR, ym_dir)
    if not os.path.isdir(ym_path):
        continue

    # 遍历日文件夹
    for day_dir in os.listdir(ym_path):
        day_path = os.path.join(ym_path, day_dir)
        if not os.path.isdir(day_path):
            continue

        # 收集该日期下的所有nc文件
        nc_files = [f for f in os.listdir(day_path) if f.endswith('.nc')]
        for nc_file in nc_files:
            # 从文件名提取时间戳（格式：YYYYMMDDHHMMSS.nc）
            timestamp = nc_file.replace('.nc', '')
            if len(timestamp) == 14 and timestamp.isdigit():
                existing_hours.add(timestamp)

print(f"找到 {len(existing_hours)} 个小时文件")

# 生成完整的小时序列，找出缺失的小时文件
missing_hours = []
current_datetime = datetime.combine(start_date, datetime.min.time())
end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(hours=23)

while current_datetime <= end_datetime:
    timestamp = current_datetime.strftime("%Y%m%d%H0000")
    if timestamp not in existing_hours:
        missing_hours.append(timestamp)
    current_datetime += timedelta(hours=1)

print(f"缺失小时文件数量: {len(missing_hours)}")

# 写入缺失的小时文件到文件
with open(OUTPUT_FILE, 'w') as f:
    for timestamp in missing_hours:
        f.write(timestamp + '\n')

print(f"缺失小时文件已写入: {OUTPUT_FILE}")

# 打印部分缺失文件示例
if missing_hours:
    print("\n缺失小时文件示例（前20个）:")
    for timestamp in missing_hours[:20]:
        # 格式化显示：YYYY-MM-DD HH:00:00
        dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        print(f"  {timestamp} ({dt.strftime('%Y-%m-%d %H:%M:%S')})")
    if len(missing_hours) > 20:
        print(f"  ... 共 {len(missing_hours)} 个缺失小时文件")
