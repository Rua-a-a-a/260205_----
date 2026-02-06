"""
扫描数据目录，检查缺失的小时文件
数据结构: YYYY/MM/YYYYMMDDHH.nc
每天应有24个小时文件（00-23）
"""

import os
from datetime import datetime, timedelta

# 数据路径
DATA_DIR = "/data_new/renamed_data/chla_nc_after_rename/chla/R"
OUTPUT_FILE = "/home/ccy/260205_缺失填充/output/缺失日期/chla_缺失日期_原始数据_小时级.txt"

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
first_months = sorted([m for m in os.listdir(first_year_path) if m.isdigit() and len(m) == 2])
start_month = first_months[0] if first_months else "01"

# 找到最后一个年份的最后一个月
last_year_path = os.path.join(DATA_DIR, end_year)
last_months = sorted([m for m in os.listdir(last_year_path) if m.isdigit() and len(m) == 2])
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

# 收集所有存在的小时文件
existing_hours = set()

for year_dir in year_dirs:
    year_path = os.path.join(DATA_DIR, year_dir)
    if not os.path.isdir(year_path):
        continue

    # 遍历月份文件夹
    for month_dir in os.listdir(year_path):
        month_path = os.path.join(year_path, month_dir)
        if not os.path.isdir(month_path):
            continue

        # 收集该月份下的所有nc文件
        nc_files = [f for f in os.listdir(month_path) if f.endswith('.nc')]
        for nc_file in nc_files:
            # 从文件名提取时间戳（格式：YYYYMMDDHH.nc）
            timestamp = nc_file.replace('.nc', '')
            if len(timestamp) == 10 and timestamp.isdigit():
                existing_hours.add(timestamp)

print(f"找到 {len(existing_hours)} 个小时文件")

# 生成完整的小时序列，找出缺失的小时文件
missing_hours = []
current_datetime = datetime.combine(start_date, datetime.min.time())
end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(hours=23)

while current_datetime <= end_datetime:
    timestamp = current_datetime.strftime("%Y%m%d%H")
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
        # 格式化显示：YYYY-MM-DD HH:00
        dt = datetime.strptime(timestamp, "%Y%m%d%H")
        print(f"  {timestamp} ({dt.strftime('%Y-%m-%d %H:00')})")
    if len(missing_hours) > 20:
        print(f"  ... 共 {len(missing_hours)} 个缺失小时文件")
