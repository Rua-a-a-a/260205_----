#!/usr/bin/env python3
"""
SST Hourly → Daily 融合脚本

功能：将SST小时数据融合为日数据
- UTC时间转本地时间(UTC+8)
- 白天窗口：本地06:00-18:00 (±6h围绕12:00)
- 高斯权重融合，σ=3，以本地12:00为锚点
- 输出：daily_sst (开尔文), missing_mask, support_map

输入路径: /data_new/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3/
输出路径: /data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion/

Author: Claude Code
Date: 2026-02-05
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


# ============ 路径配置 ============
INPUT_ROOT = Path('/data_new/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3/')
OUTPUT_ROOT = Path('/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion/')

# ============ SST截断参数 ============
SST_INVALID_THRESHOLD = 265.0  # K，低于此值视为无效数据（设为NaN）
SST_MIN = 271.0   # K (约-2°C，海水冰点附近)，截断下限
SST_MAX = 315.0   # K (约42°C，极端高温上限)，截断上限


def gaussian_weight(delta_h, sigma=3.0):
    """
    高斯权重，以本地12:00为锚点
    w(Δh) = exp(-Δh² / (2σ²))

    Args:
        delta_h: 距离本地12:00的小时数
        sigma: 高斯标准差，默认3
    """
    return np.exp(-delta_h**2 / (2 * sigma**2))


def get_valid_mask(sst_array):
    """获取有效数据掩码 (True=有效, False=缺失)

    有效条件：
    - 非NaN
    - 非0（0K是绝对零度，不可能是有效SST）
    - 大于等于 SST_INVALID_THRESHOLD（低于265K视为无效）
    """
    valid = ~np.isnan(sst_array)
    valid = valid & (sst_array != 0)
    valid = valid & (sst_array >= SST_INVALID_THRESHOLD)
    return valid


def load_hourly_nc(nc_path):
    """
    加载单个小时的NC文件

    Returns:
        sst: SST数组 (开尔文)
        lat: 纬度数组
        lon: 经度数组
    """
    try:
        ds = xr.open_dataset(nc_path)
        sst = ds['sea_surface_temperature'].values.astype(np.float32)
        if len(sst.shape) == 3:
            sst = sst[0]
        lat = ds['lat'].values
        lon = ds['lon'].values
        ds.close()
        return sst, lat, lon
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
    sigma=3.0
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

    Returns:
        daily_sst: 融合后的日数据（开尔文）
        missing_mask: 缺失掩码 (1=缺失, 0=有观测)
        support_map: 支持度图
        lat, lon: 坐标
    """
    # 收集该本地日所有白天小时的UTC文件
    hourly_data = []

    for local_hour in range(day_window_start, day_window_end + 1):
        # 本地时间 → UTC时间
        local_dt = datetime(year, month, day, local_hour)
        utc_dt = local_dt - timedelta(hours=utc_offset)

        # 构建文件路径
        # 输入路径结构: /jaxa_extract_L3/YYYYMM/DD/YYYYMMDDHHMMSS.nc
        utc_year = utc_dt.year
        utc_month = utc_dt.month
        utc_day = utc_dt.day
        utc_hour = utc_dt.hour

        month_str = f"{utc_year}{utc_month:02d}"
        day_str = f"{utc_day:02d}"
        filename = f"{utc_year}{utc_month:02d}{utc_day:02d}{utc_hour:02d}0000.nc"
        filepath = Path(data_dir) / month_str / day_str / filename

        if filepath.exists():
            sst, lat, lon = load_hourly_nc(filepath)
            if sst is not None:
                # 计算距离本地12:00的小时数
                delta_h = local_hour - 12
                weight = gaussian_weight(delta_h, sigma)

                hourly_data.append({
                    'sst': sst,
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
    H, W = hourly_data[0]['sst'].shape

    # 像元级加权融合
    weighted_sum = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)
    valid_weight_sum = np.zeros((H, W), dtype=np.float64)  # 用于计算support
    total_weight = 0.0

    for data in hourly_data:
        sst = data['sst']
        w = data['weight']
        total_weight += w

        # 获取有效掩码
        valid = get_valid_mask(sst)

        # 加权累加
        weighted_sum += w * np.where(valid, sst, 0.0)
        weight_sum += w * valid.astype(np.float64)
        valid_weight_sum += w * valid.astype(np.float64)

    # 计算融合结果
    # 避免除零
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_sst = np.where(
            weight_sum > 0,
            weighted_sum / weight_sum,
            np.nan
        )

    # 计算支持度 (0~1)
    support_map = valid_weight_sum / total_weight if total_weight > 0 else np.zeros((H, W))

    # 生成缺失掩码 (1=缺失, 0=有观测)
    missing_mask = (support_map == 0).astype(np.int8)

    # 对融合结果进行截断处理
    # 将超出范围的值截断到 [SST_MIN, SST_MAX]
    daily_sst = np.clip(daily_sst, SST_MIN, SST_MAX)

    # 将缺失位置设为NaN（而不是0）
    daily_sst = np.where(missing_mask == 1, np.nan, daily_sst)

    return daily_sst.astype(np.float32), missing_mask, support_map.astype(np.float32), lat, lon


def process_single_day(args):
    """处理单天数据（用于多进程）"""
    data_dir, year, month, day, output_dir, utc_offset, sigma = args

    try:
        daily_sst, missing_mask, support, lat, lon = fuse_hourly_to_daily(
            data_dir, year, month, day,
            utc_offset=utc_offset,
            sigma=sigma
        )

        if daily_sst is None:
            return None, f"{year}-{month:02d}-{day:02d}: No data"

        # 保存结果
        date_str = f"{year}{month:02d}{day:02d}"
        out_path = Path(output_dir) / str(year) / f"{month:02d}"
        out_path.mkdir(parents=True, exist_ok=True)

        h5_file = out_path / f"{date_str}.h5"

        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('daily_sst', data=daily_sst, compression='gzip')
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
    parser = argparse.ArgumentParser(description='SST Hourly to Daily Fusion (Gaussian Weighted)')
    parser.add_argument('--data_dir', type=str, default=str(INPUT_ROOT),
                       help='Input hourly data directory')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_ROOT),
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
    print("SST Hourly → Daily Fusion (Gaussian Weighted)")
    print("=" * 60)
    print(f"Input:  {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Years:  {args.year_start} - {args.year_end}")
    print(f"UTC offset: +{args.utc_offset}")
    print(f"Gaussian sigma: {args.sigma}")
    print(f"Day window: 06:00 - 18:00 (local time)")
    print(f"SST invalid threshold: <{SST_INVALID_THRESHOLD}K (set to NaN)")
    print(f"SST clip range: {SST_MIN}K - {SST_MAX}K ({SST_MIN-273.15:.1f}°C - {SST_MAX-273.15:.1f}°C)")
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
                daily = f['daily_sst'][:]
                mask = f['missing_mask'][:]
                support = f['support'][:]
                print(f"\nDaily shape: {daily.shape}")
                print(f"Valid ratio: {f.attrs['valid_ratio']*100:.1f}%")
                valid_pixels = mask == 0
                if valid_pixels.any():
                    print(f"SST range (K): {daily[valid_pixels].min():.2f} ~ {daily[valid_pixels].max():.2f}")
                    print(f"SST range (°C): {daily[valid_pixels].min()-273.15:.2f} ~ {daily[valid_pixels].max()-273.15:.2f}")
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
