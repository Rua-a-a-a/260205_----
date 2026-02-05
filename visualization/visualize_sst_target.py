#!/usr/bin/env python3
"""
SST Daily Fusion Target H5 文件可视化脚本

功能：可视化 sst_daily_fusion_target 目录中的单帧 H5 文件
支持：SST数据可视化、缺失率统计、数据分布直方图

Author: Claude Code
Date: 2026-02-05
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============ 硬编码配置 - 修改这里 ============

# 数据根目录
DATA_ROOT = Path("/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion_target/")

# 目标日期（格式：YYYYMMDD）
TARGET_DATE = "20240101"

# 目标小时（格式：HHMMSS，如果为None则自动查找第一个匹配的文件）
TARGET_HOUR = None  # 例如 "000000" 表示00:00:00

# 可视化参数
COLORMAP = "jet"                # 颜色映射: jet, viridis, coolwarm, RdYlBu_r
FIGSIZE = (16, 10)              # 图像尺寸
DPI = 150                       # 输出分辨率

# 输出目录
OUTPUT_DIR = Path("/home/ccy/260205_缺失填充/output/picture/sst_target/")

# ============ 硬编码配置结束 ============


def find_h5_file(data_root: Path, target_date: str, target_hour: str = None) -> Path:
    """
    查找指定日期的H5文件

    Args:
        data_root: 数据根目录
        target_date: 目标日期 (YYYYMMDD)
        target_hour: 目标小时 (HHMMSS)，如果为None则返回第一个匹配的文件

    Returns:
        H5文件路径
    """
    # 解析日期
    year = target_date[:4]
    month = target_date[4:6]

    # 构建目录路径
    target_dir = data_root / year / month

    if not target_dir.exists():
        raise FileNotFoundError(f"目录不存在: {target_dir}")

    # 查找匹配的文件
    if target_hour:
        # 精确匹配
        pattern = f"{target_date}{target_hour}.h5"
        h5_file = target_dir / pattern
        if h5_file.exists():
            return h5_file
        else:
            raise FileNotFoundError(f"文件不存在: {h5_file}")
    else:
        # 查找该日期的所有文件
        pattern = f"{target_date}*.h5"
        files = sorted(target_dir.glob(pattern))
        if files:
            return files[0]  # 返回第一个匹配的文件
        else:
            raise FileNotFoundError(f"未找到日期 {target_date} 的文件，目录: {target_dir}")


def load_h5_data(h5_path: Path) -> dict:
    """
    加载H5文件数据

    Args:
        h5_path: H5文件路径

    Returns:
        包含数据的字典
    """
    with h5py.File(h5_path, 'r') as f:
        # 打印文件结构
        print("=" * 60)
        print(f"H5 文件: {h5_path}")
        print("=" * 60)
        print("\n数据集:")
        for key in f.keys():
            ds = f[key]
            print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")

        print("\n属性:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
        print("=" * 60)

        # 加载数据
        data = {}

        # 主SST数据
        if 'daily_sst' in f:
            data['sst'] = f['daily_sst'][:]
        elif 'sst_data' in f:
            data['sst'] = f['sst_data'][:]
        elif 'sea_surface_temperature' in f:
            data['sst'] = f['sea_surface_temperature'][:]

        # 原始SST（如果存在）
        if 'original_sst' in f:
            data['original_sst'] = f['original_sst'][:]

        # 填充计数（如果存在）
        if 'fill_count' in f:
            data['fill_count'] = f['fill_count'][:]

        # 坐标
        if 'latitude' in f:
            data['lat'] = f['latitude'][:]
        if 'longitude' in f:
            data['lon'] = f['longitude'][:]

        # 保存属性
        data['attrs'] = dict(f.attrs)

    return data


def visualize_sst_data(data: dict, h5_path: Path):
    """
    可视化SST数据

    Args:
        data: 数据字典
        h5_path: H5文件路径（用于标题和输出文件名）
    """
    sst = data['sst'].copy()
    lat = data.get('lat', None)
    lon = data.get('lon', None)

    # 将0值视为缺失值（SST不可能为0K）
    sst[sst == 0] = np.nan

    # 确定子图数量
    has_original = 'original_sst' in data
    has_fill_count = 'fill_count' in data

    if has_original and has_fill_count:
        fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
        axes = axes.flatten()
    elif has_original or has_fill_count:
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes = axes.flatten()

    # 设置经纬度范围
    extent = None
    if lat is not None and lon is not None:
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # 1. 填充后的SST数据
    ax = axes[0]
    valid_data = sst[~np.isnan(sst)]
    if len(valid_data) > 0:
        vmin, vmax = np.percentile(valid_data, [2, 98])
    else:
        vmin, vmax = 290, 310  # 默认SST范围（开尔文）

    if extent:
        im = ax.imshow(sst, cmap=COLORMAP, vmin=vmin, vmax=vmax,
                       aspect='auto', origin='upper', extent=extent)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    else:
        im = ax.imshow(sst, cmap=COLORMAP, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xlabel("Longitude Index")
        ax.set_ylabel("Latitude Index")

    missing_rate = np.isnan(sst).sum() / sst.size * 100
    ax.set_title(f"SST (Filled)\nMissing: {missing_rate:.2f}%", fontsize=12)
    plt.colorbar(im, ax=ax, label="SST (K)")

    # 2. 缺失掩码
    ax = axes[1]
    nan_mask = np.isnan(sst).astype(np.uint8)
    if extent:
        im = ax.imshow(nan_mask, cmap='gray_r', vmin=0, vmax=1,
                       aspect='auto', origin='upper', extent=extent)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    else:
        im = ax.imshow(nan_mask, cmap='gray_r', vmin=0, vmax=1, aspect='auto')
        ax.set_xlabel("Longitude Index")
        ax.set_ylabel("Latitude Index")
    ax.set_title(f"Missing Mask (white=missing)\nMissing Rate: {missing_rate:.2f}%", fontsize=12)
    plt.colorbar(im, ax=ax)

    # 3. 数据直方图
    ax = axes[2]
    if len(valid_data) > 0:
        ax.hist(valid_data.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(valid_data.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_data.mean():.2f} K')
        ax.axvline(np.median(valid_data), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(valid_data):.2f} K')
        ax.legend(fontsize=10)
    ax.set_title("SST Distribution", fontsize=12)
    ax.set_xlabel("SST (K)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    plot_idx = 3

    # 4. 原始SST（如果存在）
    if has_original:
        ax = axes[plot_idx]
        original_sst = data['original_sst']
        if extent:
            im = ax.imshow(original_sst, cmap=COLORMAP, vmin=vmin, vmax=vmax,
                           aspect='auto', origin='upper', extent=extent)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
        else:
            im = ax.imshow(original_sst, cmap=COLORMAP, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_xlabel("Longitude Index")
            ax.set_ylabel("Latitude Index")
        original_missing = np.isnan(original_sst).sum() / original_sst.size * 100
        ax.set_title(f"SST (Original)\nMissing: {original_missing:.2f}%", fontsize=12)
        plt.colorbar(im, ax=ax, label="SST (K)")
        plot_idx += 1

    # 5. 填充计数（如果存在）
    if has_fill_count:
        ax = axes[plot_idx]
        fill_count = data['fill_count']
        if extent:
            im = ax.imshow(fill_count, cmap='YlOrRd', vmin=0,
                           aspect='auto', origin='upper', extent=extent)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
        else:
            im = ax.imshow(fill_count, cmap='YlOrRd', vmin=0, aspect='auto')
            ax.set_xlabel("Longitude Index")
            ax.set_ylabel("Latitude Index")
        filled_pixels = (fill_count > 0).sum()
        fill_rate = filled_pixels / fill_count.size * 100
        ax.set_title(f"Fill Count\nFilled Pixels: {fill_rate:.2f}%", fontsize=12)
        plt.colorbar(im, ax=ax, label="Source Count")
        plot_idx += 1

    # 6. 填充前后对比（如果有原始数据）
    if has_original and plot_idx < len(axes):
        ax = axes[plot_idx]
        original_sst = data['original_sst']
        diff = sst - original_sst
        diff_valid = diff[~np.isnan(diff)]
        if len(diff_valid) > 0:
            vmax_diff = np.percentile(np.abs(diff_valid), 95)
            if extent:
                im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff,
                               aspect='auto', origin='upper', extent=extent)
                ax.set_xlabel("Longitude (°E)")
                ax.set_ylabel("Latitude (°N)")
            else:
                im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='auto')
                ax.set_xlabel("Longitude Index")
                ax.set_ylabel("Latitude Index")
            ax.set_title(f"Difference (Filled - Original)\nMean: {diff_valid.mean():.4f} K", fontsize=12)
            plt.colorbar(im, ax=ax, label="Diff (K)")
        else:
            ax.text(0.5, 0.5, "No difference data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Difference (N/A)")

    # 设置总标题
    h5_name = h5_path.stem
    timestamp_str = ""
    if 'timestamp' in data['attrs']:
        ts = data['attrs']['timestamp']
        if isinstance(ts, bytes):
            ts = ts.decode()
        # 格式化时间戳
        try:
            dt = datetime.strptime(ts, "%Y%m%d%H%M%S")
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            timestamp_str = ts

    plt.suptitle(f"SST Visualization: {h5_name}\n{timestamp_str}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图像
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{h5_name}.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"\n图像已保存: {output_file}")

    plt.close()


def print_statistics(data: dict):
    """打印数据统计信息"""
    sst = data['sst'].copy()

    # 将0值视为缺失值
    sst[sst == 0] = np.nan

    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    print(f"数据形状: {sst.shape} (H, W)")
    print(f"数据类型: {sst.dtype}")

    # 坐标信息
    if 'lat' in data and 'lon' in data:
        lat, lon = data['lat'], data['lon']
        print(f"\n经纬度范围:")
        print(f"  纬度: {lat.min():.4f}°N - {lat.max():.4f}°N (共 {len(lat)} 点)")
        print(f"  经度: {lon.min():.4f}°E - {lon.max():.4f}°E (共 {len(lon)} 点)")
        if len(lat) > 1:
            print(f"  分辨率: ~{abs(lat[1]-lat[0]):.4f}°")

    # 有效数据统计
    valid_data = sst[~np.isnan(sst)]
    total_pixels = sst.size
    valid_pixels = len(valid_data)
    missing_pixels = total_pixels - valid_pixels

    print(f"\n数据覆盖:")
    print(f"  总像素: {total_pixels:,}")
    print(f"  有效像素: {valid_pixels:,} ({valid_pixels/total_pixels*100:.2f}%)")
    print(f"  缺失像素: {missing_pixels:,} ({missing_pixels/total_pixels*100:.2f}%)")

    if len(valid_data) > 0:
        print(f"\nSST统计 (开尔文):")
        print(f"  最小值: {valid_data.min():.2f} K ({valid_data.min()-273.15:.2f} °C)")
        print(f"  最大值: {valid_data.max():.2f} K ({valid_data.max()-273.15:.2f} °C)")
        print(f"  均值: {valid_data.mean():.2f} K ({valid_data.mean()-273.15:.2f} °C)")
        print(f"  标准差: {valid_data.std():.2f} K")
        print(f"  中位数: {np.median(valid_data):.2f} K ({np.median(valid_data)-273.15:.2f} °C)")

    # 填充统计
    if 'fill_count' in data:
        fill_count = data['fill_count']
        filled_pixels = (fill_count > 0).sum()
        print(f"\n填充统计:")
        print(f"  填充像素: {filled_pixels:,} ({filled_pixels/total_pixels*100:.2f}%)")
        if filled_pixels > 0:
            print(f"  平均使用源数: {fill_count[fill_count > 0].mean():.1f}")
            print(f"  最大使用源数: {fill_count.max()}")

    # 原始数据对比
    if 'original_sst' in data:
        original_sst = data['original_sst']
        original_missing = np.isnan(original_sst).sum()
        print(f"\n填充效果:")
        print(f"  原始缺失: {original_missing:,} ({original_missing/total_pixels*100:.2f}%)")
        print(f"  填充后缺失: {missing_pixels:,} ({missing_pixels/total_pixels*100:.2f}%)")
        print(f"  缺失减少: {original_missing - missing_pixels:,} ({(original_missing - missing_pixels)/total_pixels*100:.2f}%)")

    print("=" * 60)


def main():
    """主函数"""
    print(f"\n查找日期 {TARGET_DATE} 的H5文件...")

    try:
        # 查找文件
        h5_path = find_h5_file(DATA_ROOT, TARGET_DATE, TARGET_HOUR)
        print(f"找到文件: {h5_path}")

        # 加载数据
        data = load_h5_data(h5_path)

        # 打印统计信息
        print_statistics(data)

        # 可视化
        print(f"\n生成可视化图像...")
        visualize_sst_data(data, h5_path)

    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
