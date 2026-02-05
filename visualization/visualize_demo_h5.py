#!/usr/bin/env python3
"""
H5 文件可视化脚本

功能：可视化 H5 文件中的数据（SST、叶绿素等）
支持：单帧可视化、多帧对比、缺失率统计

Author: Claude Code
Date: 2026-02-05
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============ 硬编码配置 - 修改这里 ============

# H5 文件路径
H5_FILE_PATH = "/data_new/chla_data_imputation_data_260125/SST/jaxa_weighted_aligned/jaxa_weighted_series_07" \
".h5"

# 数据集名称（根据 H5 文件结构修改）
DATA_KEY = "sst_data"           # 主数据 (T, H, W)
MASK_KEY = "missing_mask"       # 缺失掩码 (T, H, W), 1=缺失, 0=有效
FILL_MASK_KEY = "fill_mask"     # 填充掩码 (T, H, W), 1=填充, 0=原始
LAT_KEY = "latitude"            # 纬度
LON_KEY = "longitude"           # 经度
TIME_KEY = "timestamps"         # 时间戳

# 可视化参数
FRAME_INDEX = 0                 # 要可视化的帧索引 (0-based)
COLORMAP = "jet"                # 颜色映射: jet, viridis, coolwarm, RdYlBu_r
FIGSIZE = (16, 10)              # 图像尺寸
DPI = 150                       # 输出分辨率

# 输出目录（自动生成文件名）
OUTPUT_DIR = "/home/ccy/260205_缺失填充/output/picture/demo/"

# ============ 硬编码配置结束 ============


def load_h5_data(h5_path):
    """加载 H5 文件数据"""
    with h5py.File(h5_path, 'r') as f:
        # 打印文件结构
        print("=" * 50)
        print(f"H5 文件: {h5_path}")
        print("=" * 50)
        print("\n数据集:")
        for key in f.keys():
            ds = f[key]
            print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")

        print("\n属性:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
        print("=" * 50)

        # 加载数据
        data = {}
        data['main'] = f[DATA_KEY][:]

        if MASK_KEY in f:
            data['mask'] = f[MASK_KEY][:]

        if FILL_MASK_KEY in f:
            data['fill_mask'] = f[FILL_MASK_KEY][:]

        if LAT_KEY in f:
            data['lat'] = f[LAT_KEY][:]

        if LON_KEY in f:
            data['lon'] = f[LON_KEY][:]

        if TIME_KEY in f:
            data['timestamps'] = [t.decode() if isinstance(t, bytes) else t for t in f[TIME_KEY][:]]

        # 保存属性
        data['attrs'] = dict(f.attrs)

    return data


def visualize_single_frame(data, frame_idx=0):
    """可视化单帧数据"""
    main_data = data['main']

    # 检查帧索引
    if frame_idx >= main_data.shape[0]:
        print(f"Error: Frame index {frame_idx} out of range (max: {main_data.shape[0]-1})")
        return

    frame = main_data[frame_idx]

    # 获取时间戳
    timestamp = ""
    if 'timestamps' in data and frame_idx < len(data['timestamps']):
        timestamp = data['timestamps'][frame_idx]

    # 获取经纬度范围
    lat = data.get('lat', None)
    lon = data.get('lon', None)

    # 创建图像
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

    # 1. 主数据
    ax = axes[0, 0]
    valid_data = frame[~np.isnan(frame)]
    if len(valid_data) > 0:
        vmin, vmax = np.percentile(valid_data, [2, 98])
    else:
        vmin, vmax = 0, 1

    # 使用经纬度作为坐标轴
    if lat is not None and lon is not None:
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        im = ax.imshow(frame, cmap=COLORMAP, vmin=vmin, vmax=vmax,
                       aspect='auto', origin='upper', extent=extent)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        im = ax.imshow(frame, cmap=COLORMAP, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xlabel("Longitude Index")
        ax.set_ylabel("Latitude Index")

    ax.set_title(f"Data (Frame {frame_idx})\n{timestamp}", fontsize=12)
    plt.colorbar(im, ax=ax, label=f"{DATA_KEY} (K)")

    # 2. 缺失掩码
    ax = axes[0, 1]
    if 'mask' in data:
        mask = data['mask'][frame_idx]
        if lat is not None and lon is not None:
            im = ax.imshow(mask, cmap='gray_r', vmin=0, vmax=1,
                          aspect='auto', origin='upper', extent=extent)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            im = ax.imshow(mask, cmap='gray_r', vmin=0, vmax=1, aspect='auto')
            ax.set_xlabel("Longitude Index")
            ax.set_ylabel("Latitude Index")
        missing_rate = mask.sum() / mask.size * 100
        ax.set_title(f"Missing Mask (white=missing)\nMissing Rate: {missing_rate:.2f}%", fontsize=12)
        plt.colorbar(im, ax=ax)
    else:
        # 从 NaN 推断缺失
        nan_mask = np.isnan(frame).astype(np.uint8)
        if lat is not None and lon is not None:
            im = ax.imshow(nan_mask, cmap='gray_r', vmin=0, vmax=1,
                          aspect='auto', origin='upper', extent=extent)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            im = ax.imshow(nan_mask, cmap='gray_r', vmin=0, vmax=1, aspect='auto')
            ax.set_xlabel("Longitude Index")
            ax.set_ylabel("Latitude Index")
        missing_rate = nan_mask.sum() / nan_mask.size * 100
        ax.set_title(f"NaN Mask (white=NaN)\nMissing Rate: {missing_rate:.2f}%", fontsize=12)
        plt.colorbar(im, ax=ax)

    # 3. 填充掩码
    ax = axes[1, 0]
    if 'fill_mask' in data:
        fill_mask = data['fill_mask'][frame_idx]
        if lat is not None and lon is not None:
            im = ax.imshow(fill_mask, cmap='Oranges', vmin=0, vmax=1,
                          aspect='auto', origin='lower', extent=extent)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            im = ax.imshow(fill_mask, cmap='Oranges', vmin=0, vmax=1, aspect='auto')
            ax.set_xlabel("Longitude Index")
            ax.set_ylabel("Latitude Index")
        fill_rate = fill_mask.sum() / fill_mask.size * 100
        ax.set_title(f"Fill Mask (orange=filled)\nFill Rate: {fill_rate:.2f}%", fontsize=12)
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "No fill mask data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Fill Mask (N/A)")
        ax.set_xlabel("Longitude Index")
        ax.set_ylabel("Latitude Index")

    # 4. 数据直方图
    ax = axes[1, 1]
    valid_data = frame[~np.isnan(frame)]
    if len(valid_data) > 0:
        ax.hist(valid_data.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(valid_data.mean(), color='red', linestyle='--', label=f'Mean: {valid_data.mean():.2f}')
        ax.axvline(np.median(valid_data), color='green', linestyle='--', label=f'Median: {np.median(valid_data):.2f}')
        ax.legend()
    ax.set_title("Data Distribution Histogram", fontsize=12)
    ax.set_xlabel(f"{DATA_KEY} (K)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    h5_name = Path(H5_FILE_PATH).stem
    plt.suptitle(f"H5 Visualization: {h5_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存或显示
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{h5_name}_frame{frame_idx:03d}.png"
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f"\nImage saved: {output_file}")
    else:
        plt.show()

    plt.close()


def print_statistics(data):
    """打印数据统计信息"""
    main_data = data['main']

    print("\n" + "=" * 50)
    print("数据统计")
    print("=" * 50)
    print(f"数据形状: {main_data.shape} (T, H, W)")
    print(f"数据类型: {main_data.dtype}")

    # 有效数据统计
    valid_data = main_data[~np.isnan(main_data)]
    print(f"\n有效数据点: {len(valid_data):,} / {main_data.size:,} ({len(valid_data)/main_data.size*100:.2f}%)")

    if len(valid_data) > 0:
        print(f"最小值: {valid_data.min():.4f}")
        print(f"最大值: {valid_data.max():.4f}")
        print(f"均值: {valid_data.mean():.4f}")
        print(f"标准差: {valid_data.std():.4f}")
        print(f"中位数: {np.median(valid_data):.4f}")

    # 每帧缺失率
    if 'mask' in data:
        mask = data['mask']
        missing_rates = mask.sum(axis=(1, 2)) / (mask.shape[1] * mask.shape[2]) * 100
        print(f"\n每帧缺失率:")
        print(f"  最小: {missing_rates.min():.2f}%")
        print(f"  最大: {missing_rates.max():.2f}%")
        print(f"  平均: {missing_rates.mean():.2f}%")

    print("=" * 50)


def main():
    """主函数"""
    print(f"\n加载文件: {H5_FILE_PATH}")

    # 检查文件是否存在
    if not Path(H5_FILE_PATH).exists():
        print(f"错误: 文件不存在 - {H5_FILE_PATH}")
        return

    # 加载数据
    data = load_h5_data(H5_FILE_PATH)

    # 打印统计信息
    print_statistics(data)

    # 可视化
    print(f"\n可视化帧索引: {FRAME_INDEX}")
    visualize_single_frame(data, frame_idx=FRAME_INDEX)


if __name__ == "__main__":
    main()
