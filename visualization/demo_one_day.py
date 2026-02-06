"""
可视化20240101一整天24小时的SST图像
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# 数据路径
DATA_DIR = "/data_new/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3/202401/01"
OUTPUT_DIR = "/home/ccy/260205_缺失填充/output/picture/demo_one_day"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 获取所有nc文件并排序
nc_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.nc')])

print(f"找到 {len(nc_files)} 个文件")

# 创建一个4x6的子图布局显示24小时数据
fig, axes = plt.subplots(4, 6, figsize=(24, 16),
                         subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

# 先读取所有数据获取统一的colorbar范围
all_sst = []
for nc_file in nc_files:
    file_path = os.path.join(DATA_DIR, nc_file)
    ds = xr.open_dataset(file_path)
    sst = ds['sea_surface_temperature'].values[0] - 273.15  # 转换为摄氏度
    all_sst.append(sst)
    ds.close()

# 计算全局最小最大值（忽略NaN）
vmin = np.nanmin(all_sst)
vmax = np.nanmax(all_sst)

# 绘制每个小时的图像
for i, nc_file in enumerate(nc_files):
    file_path = os.path.join(DATA_DIR, nc_file)
    ds = xr.open_dataset(file_path)

    sst = ds['sea_surface_temperature'].values[0] - 273.15  # 转换为摄氏度
    lat = ds['lat'].values
    lon = ds['lon'].values

    ax = axes[i]

    # 绘制SST
    im = ax.pcolormesh(lon, lat, sst, transform=ccrs.PlateCarree(),
                       cmap='jet', vmin=vmin, vmax=vmax)

    # 设置标题（显示小时）
    hour = nc_file[8:10]
    ax.set_title(f'{hour}:00 UTC', fontsize=10)

    # 添加经纬度网格
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)

    ds.close()

# 添加统一的colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('SST (°C)', fontsize=12)

# 设置总标题
fig.suptitle('Sea Surface Temperature - 2024/01/01 (24 Hours)', fontsize=16, y=0.98)

plt.tight_layout(rect=[0, 0, 0.9, 0.96])

# 保存图像
output_path = os.path.join(OUTPUT_DIR, 'sst_20240101_24hours.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"24小时合并图已保存至: {output_path}")

# 同时保存每个小时的单独图像
print("\n正在生成单独的小时图像...")

for i, nc_file in enumerate(nc_files):
    file_path = os.path.join(DATA_DIR, nc_file)
    ds = xr.open_dataset(file_path)

    sst = ds['sea_surface_temperature'].values[0] - 273.15
    lat = ds['lat'].values
    lon = ds['lon'].values

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    im = ax.pcolormesh(lon, lat, sst, transform=ccrs.PlateCarree(),
                       cmap='jet', vmin=vmin, vmax=vmax)

    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    hour = nc_file[8:10]
    ax.set_title(f'Sea Surface Temperature - 2024/01/01 {hour}:00 UTC', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label('SST (°C)', fontsize=12)

    # 保存单独图像
    single_output_path = os.path.join(OUTPUT_DIR, f'sst_20240101_{hour}00.png')
    plt.savefig(single_output_path, dpi=150, bbox_inches='tight')
    plt.close()

    ds.close()

print(f"24张单独图像已保存至: {OUTPUT_DIR}")
print("完成!")
