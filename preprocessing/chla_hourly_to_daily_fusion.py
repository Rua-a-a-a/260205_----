#!/usr/bin/env python3
"""
Hourly → Daily 融合脚本

功能：将叶绿素小时数据融合为日数据
- UTC时间转本地时间(UTC+8)
- 白天窗口：本地06:00-18:00 (±6h围绕12:00)
- 高斯权重融合，σ=3，以本地12:00为锚点
- 输出：daily_D (归一化后), missing_mask, support_map

Author: Claude
Date: 2025-01-30
"""

import os
import sys
import argparse
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


# ============ 叶绿素归一化参数 ============
CHLA_MIN = 0.01      # mg/m³ 下限截断
CHLA_MAX = 100.0     # mg/m³ 上限截断
FILL_VALUE = -131.068  # 原始数据的缺失标记

# 归一化公式: C_N = (ln(C) + 4.61) / 9.22
# 反归一化: C = exp(9.22 * C_N - 4.61)
LN_OFFSET = 4.61
LN_SCALE = 9.22


def chla_to_normalized(chla):
    """
    叶绿素浓度 → 归一化值 [0, 1]
    C_N = (ln(clip(C, 0.01, 100)) + 4.61) / 9.22
    """
    chla_clipped = np.clip(chla, CHLA_MIN, CHLA_MAX)
    c_n = (np.log(chla_clipped) + LN_OFFSET) / LN_SCALE
    c_n = np.clip(c_n, 0, 1)
    return c_n


def normalized_to_chla(c_n):
    """
    归一化值 → 叶绿素浓度 (mg/m³)
    C = exp(9.22 * C_N - 4.61)
    """
    ln_c = LN_SCALE * c_n - LN_OFFSET
    chla = np.exp(ln_c)
    chla = np.clip(chla, CHLA_MIN, CHLA_MAX)
    return chla


def gaussian_weight(delta_h, sigma=3.0):
    """
    高斯权重，以本地12:00为锚点
    w(Δh) = exp(-Δh² / (2σ²))

    Args:
        delta_h: 距离本地12:00的小时数
        sigma: 高斯标准差，默认3
    """
    return np.exp(-delta_h**2 / (2 * sigma**2))


def is_valid_chla(chla_value):
    """判断叶绿素值是否有效（非缺失）"""
    if np.isnan(chla_value):
        return False
    if chla_value <= 0:
        return False
    if np.abs(chla_value - FILL_VALUE) < 0.1:
        return False
    return True


def get_valid_mask(chla_array):
    """获取有效数据掩码 (True=有效, False=缺失)"""
    valid = ~np.isnan(chla_array)
    valid = valid & (chla_array > 0)
    valid = valid & (np.abs(chla_array - FILL_VALUE) > 0.1)
    return valid


def load_hourly_nc(nc_path, var_name='chlor_a'):
    """
    加载单个小时的NC文件

    Returns:
        chla: 叶绿素数组
        lat: 纬度数组
        lon: 经度数组
    """
    try:
        ds = xr.open_dataset(nc_path)
        chla = ds[var_name].values.astype(np.float32)
        lat = ds['latitude'].values
        lon = ds['longitude'].values
        ds.close()
        return chla, lat, lon
    except Exception as e:
        print(f"Error loading {nc_path}: {e}")
        return None, None, None


def fuse_hourly_to_daily(
    data_dir,
    year,
    month,
    day,
    utc_offset=8,
    day_window_start=6,   # 本地06:00
    day_window_end=18,    # 本地18:00
    sigma=3.0,
    var_name='chlor_a'
):
    """
    将某一天的小时数据融合为日数据

    Args:
        data_dir: 数据根目录
        year, month, day: 目标日期（本地日期）
        utc_offset: 时区偏移（中国为+8）
        day_window_start: 白天窗口开始（本地时间）
        day_window_end: 白天窗口结束（本地时间）
        sigma: 高斯权重标准差
        var_name: 叶绿素变量名

    Returns:
        daily_chla_norm: 融合后的日数据（归一化）
        missing_mask: 缺失掩码 (1=缺失, 0=有观测)
        support_map: 支持度图
        lat, lon: 坐标
    """
    # 本地日期
    local_date = datetime(year, month, day)

    # 收集该本地日所有白天小时的UTC文件
    hourly_data = []

    for local_hour in range(day_window_start, day_window_end + 1):
        # 本地时间 → UTC时间
        local_dt = datetime(year, month, day, local_hour)
        utc_dt = local_dt - timedelta(hours=utc_offset)

        # 构建文件路径
        utc_year = utc_dt.year
        utc_month = utc_dt.month
        utc_day = utc_dt.day
        utc_hour = utc_dt.hour

        filename = f"{utc_year}{utc_month:02d}{utc_day:02d}{utc_hour:02d}.nc"
        filepath = Path(data_dir) / str(utc_year) / f"{utc_month:02d}" / filename

        if filepath.exists():
            chla, lat, lon = load_hourly_nc(filepath, var_name)
            if chla is not None:
                # 计算距离本地12:00的小时数
                delta_h = local_hour - 12
                weight = gaussian_weight(delta_h, sigma)

                hourly_data.append({
                    'chla': chla,
                    'weight': weight,
                    'local_hour': local_hour,
                    'lat': lat,
                    'lon': lon
                })

    if len(hourly_data) == 0:
        return None, None, None, None, None

    # 获取网格尺寸
    lat = hourly_data[0]['lat']
    lon = hourly_data[0]['lon']
    H, W = hourly_data[0]['chla'].shape

    # 像元级加权融合（在归一化空间）
    weighted_sum = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)
    valid_weight_sum = np.zeros((H, W), dtype=np.float64)  # 用于计算support
    total_weight = 0.0

    for data in hourly_data:
        chla = data['chla']
        w = data['weight']
        total_weight += w

        # 获取有效掩码
        valid = get_valid_mask(chla)

        # 对有效像元：先归一化再加权
        chla_norm = np.zeros_like(chla)
        chla_norm[valid] = chla_to_normalized(chla[valid])

        # 加权累加
        weighted_sum += w * chla_norm * valid.astype(np.float64)
        weight_sum += w * valid.astype(np.float64)
        valid_weight_sum += w * valid.astype(np.float64)

    # 计算融合结果
    # 避免除零
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_chla_norm = np.where(
            weight_sum > 0,
            weighted_sum / weight_sum,
            np.nan
        )

    # 计算支持度 (0~1)
    support_map = valid_weight_sum / total_weight if total_weight > 0 else np.zeros((H, W))

    # 生成缺失掩码 (1=缺失, 0=有观测)
    missing_mask = (support_map == 0).astype(np.int8)

    # 将NaN位置设为0（后续baseline填充会处理）
    daily_chla_norm = np.nan_to_num(daily_chla_norm, nan=0.0)

    return daily_chla_norm.astype(np.float32), missing_mask, support_map.astype(np.float32), lat, lon


def process_single_day(args):
    """处理单天数据（用于多进程）"""
    data_dir, year, month, day, output_dir, utc_offset, sigma = args

    try:
        daily_norm, missing_mask, support, lat, lon = fuse_hourly_to_daily(
            data_dir, year, month, day,
            utc_offset=utc_offset,
            sigma=sigma
        )

        if daily_norm is None:
            return None, f"{year}-{month:02d}-{day:02d}: No data"

        # 保存结果
        date_str = f"{year}{month:02d}{day:02d}"
        out_path = Path(output_dir) / str(year) / f"{month:02d}"
        out_path.mkdir(parents=True, exist_ok=True)

        h5_file = out_path / f"{date_str}.h5"

        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('daily_chla_norm', data=daily_norm, compression='gzip')
            f.create_dataset('missing_mask', data=missing_mask, compression='gzip')
            f.create_dataset('support', data=support, compression='gzip')
            f.create_dataset('latitude', data=lat)
            f.create_dataset('longitude', data=lon)
            f.attrs['date'] = date_str
            f.attrs['utc_offset'] = utc_offset
            f.attrs['sigma'] = sigma
            f.attrs['valid_ratio'] = float(1 - missing_mask.mean())

        valid_ratio = 1 - missing_mask.mean()
        return h5_file, f"{year}-{month:02d}-{day:02d}: valid={valid_ratio*100:.1f}%"

    except Exception as e:
        return None, f"{year}-{month:02d}-{day:02d}: Error - {e}"


def process_year_month(data_dir, year, month, output_dir, utc_offset=8, sigma=3.0, num_workers=8):
    """处理一个月的数据"""
    from calendar import monthrange

    _, num_days = monthrange(year, month)

    tasks = []
    for day in range(1, num_days + 1):
        tasks.append((data_dir, year, month, day, output_dir, utc_offset, sigma))

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_day, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks),
                          desc=f"{year}-{month:02d}"):
            result, msg = future.result()
            results.append((result, msg))

    return results


def main():
    parser = argparse.ArgumentParser(description='Hourly to Daily Chla Fusion')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Input hourly data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output daily data directory')
    parser.add_argument('--year_start', type=int, default=2015,
                       help='Start year')
    parser.add_argument('--year_end', type=int, default=2025,
                       help='End year')
    parser.add_argument('--month_start', type=int, default=1,
                       help='Start month (for partial processing)')
    parser.add_argument('--month_end', type=int, default=12,
                       help='End month (for partial processing)')
    parser.add_argument('--utc_offset', type=int, default=8,
                       help='UTC offset (default: 8 for China)')
    parser.add_argument('--sigma', type=float, default=3.0,
                       help='Gaussian weight sigma (default: 3.0)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--test_single_day', action='store_true',
                       help='Test with single day only')

    args = parser.parse_args()

    print("=" * 60)
    print("Hourly → Daily Chla Fusion")
    print("=" * 60)
    print(f"Input:  {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Years:  {args.year_start} - {args.year_end}")
    print(f"UTC offset: +{args.utc_offset}")
    print(f"Gaussian sigma: {args.sigma}")
    print("=" * 60)

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.test_single_day:
        # 测试模式：只处理一天
        print("\n[TEST MODE] Processing single day: 2020-06-15")
        result, msg = process_single_day(
            (args.data_dir, 2020, 6, 15, args.output_dir, args.utc_offset, args.sigma)
        )
        print(f"Result: {msg}")
        if result:
            print(f"Saved to: {result}")

            # 显示结果统计
            with h5py.File(result, 'r') as f:
                daily = f['daily_chla_norm'][:]
                mask = f['missing_mask'][:]
                support = f['support'][:]
                print(f"\nDaily shape: {daily.shape}")
                print(f"Valid ratio: {f.attrs['valid_ratio']*100:.1f}%")
                print(f"Chla_norm range: {daily[mask==0].min():.4f} ~ {daily[mask==0].max():.4f}")
                print(f"Support range: {support.min():.4f} ~ {support.max():.4f}")
        return

    # 正式处理
    for year in range(args.year_start, args.year_end + 1):
        for month in range(args.month_start, args.month_end + 1):
            print(f"\nProcessing {year}-{month:02d}...")
            results = process_year_month(
                args.data_dir, year, month, args.output_dir,
                utc_offset=args.utc_offset,
                sigma=args.sigma,
                num_workers=args.num_workers
            )

            # 统计
            success = sum(1 for r, _ in results if r is not None)
            print(f"  Completed: {success}/{len(results)} days")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
