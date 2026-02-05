#!/usr/bin/env python3
"""
SST H5文件裁剪与分辨率对齐脚本

功能：
1. 读取H5格式的SST数据（来自sst_hourly_to_daily_fusion.py的输出）
2. 裁剪到指定的经纬度范围
3. 重采样到目标分辨率（0.05°），与叶绿素a数据对齐
4. 保存为H5格式

输入目录结构：/data_new/.../sst_daily_fusion/YYYY/MM/YYYYMMDD.h5
输出目录结构：/data_new/.../sst_daily_fusion_target/YYYY/MM/YYYYMMDD.h5

Author: Claude
Date: 2026-02-05
"""

import argparse
import h5py
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Configuration Constants
# ============================================================

# 输入输出路径
INPUT_ROOT = Path('/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion/')
OUTPUT_ROOT = Path('/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion_target/')

# CHLA参考文件（用于获取目标网格坐标）
CHLA_REFERENCE_PATH = Path('/data_new/chla_data_imputation_data_260125/chla_data_pretraining/daily_fusion_target_modified/2015/07/20150701.h5')

# 裁剪范围
LAT_MIN = 15.0   # 南边界
LAT_MAX = 24.0   # 北边界
LON_MIN = 111.0  # 西边界
LON_MAX = 118.0  # 东边界

# 目标分辨率
TARGET_RESOLUTION = 0.05  # 度

# 默认并行数
DEFAULT_NUM_WORKERS = 32


# ============================================================
# Grid Resampling Functions
# ============================================================

def get_target_grid_from_reference(reference_path: Path = None):
    """
    从CHLA参考文件获取目标网格坐标

    Args:
        reference_path: CHLA参考文件路径，如果为None则使用默认路径

    Returns:
        Tuple of (lat_target, lon_target)
    """
    ref_path = reference_path or CHLA_REFERENCE_PATH

    if ref_path.exists():
        with h5py.File(ref_path, 'r') as f:
            lat_target = f['latitude'][:]
            lon_target = f['longitude'][:]
        return lat_target, lon_target
    else:
        # 如果参考文件不存在，根据裁剪范围和目标分辨率生成网格
        lat_target = np.arange(LAT_MIN, LAT_MAX + TARGET_RESOLUTION/2, TARGET_RESOLUTION)
        lon_target = np.arange(LON_MIN, LON_MAX + TARGET_RESOLUTION/2, TARGET_RESOLUTION)
        return lat_target, lon_target


def resample_to_target_grid(data: np.ndarray,
                            lat_src: np.ndarray,
                            lon_src: np.ndarray,
                            lat_dst: np.ndarray,
                            lon_dst: np.ndarray,
                            method: str = 'linear') -> np.ndarray:
    """
    将数据重采样到目标网格

    Args:
        data: 源数据数组 (lat, lon)
        lat_src: 源纬度数组
        lon_src: 源经度数组
        lat_dst: 目标纬度数组
        lon_dst: 目标经度数组
        method: 插值方法，'linear' 或 'nearest'

    Returns:
        重采样后的数据数组
    """
    # 处理NaN值：创建有效数据掩码
    valid_mask = ~np.isnan(data)

    # 如果全是NaN，直接返回NaN数组
    if not np.any(valid_mask):
        return np.full((len(lat_dst), len(lon_dst)), np.nan, dtype=np.float32)

    # 创建插值器
    interpolator = RegularGridInterpolator(
        (lat_src, lon_src),
        data,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )

    # 创建目标网格点
    lon_grid, lat_grid = np.meshgrid(lon_dst, lat_dst)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    # 执行插值
    resampled = interpolator(points).reshape(len(lat_dst), len(lon_dst))

    return resampled.astype(np.float32)


# ============================================================
# H5 File Processing Functions
# ============================================================

def load_h5_file(filepath: Path):
    """
    加载H5文件

    Args:
        filepath: H5文件路径

    Returns:
        Tuple of (sst, missing_mask, support, lat, lon) or None if failed
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # 读取SST数据
            sst = f['daily_sst'][:]

            # 读取缺失掩码（如果存在）
            missing_mask = None
            if 'missing_mask' in f:
                missing_mask = f['missing_mask'][:]

            # 读取支持度（如果存在）
            support = None
            if 'support' in f:
                support = f['support'][:]

            # 读取坐标
            lat = f['latitude'][:]
            lon = f['longitude'][:]

        return sst, missing_mask, support, lat, lon

    except Exception as e:
        print(f"    Error loading {filepath.name}: {e}")
        return None


def crop_data(data: np.ndarray,
              lat: np.ndarray,
              lon: np.ndarray,
              lat_min: float = LAT_MIN,
              lat_max: float = LAT_MAX,
              lon_min: float = LON_MIN,
              lon_max: float = LON_MAX):
    """
    裁剪数据到指定经纬度范围

    Args:
        data: 数据数组 (lat, lon)
        lat: 纬度数组
        lon: 经度数组
        lat_min, lat_max: 纬度范围
        lon_min, lon_max: 经度范围

    Returns:
        Tuple of (cropped_data, cropped_lat, cropped_lon)
    """
    # 找到裁剪范围的索引
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return None, None, None

    # 裁剪数据
    cropped_data = data[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    cropped_lat = lat[lat_idx]
    cropped_lon = lon[lon_idx]

    return cropped_data, cropped_lat, cropped_lon


def process_single_h5_file(args):
    """
    处理单个H5文件：裁剪 + 重采样 + 保存为H5

    Args:
        args: Tuple of (input_path, output_path, lat_target, lon_target, skip_existing)

    Returns:
        Tuple of (success, message)
    """
    input_path, output_path, lat_target, lon_target, skip_existing = args

    # 跳过已存在的文件
    if skip_existing and output_path.exists():
        return True, "Skipped (exists)"

    try:
        # 加载H5文件
        result = load_h5_file(input_path)
        if result is None:
            return False, "Failed to load"

        sst, missing_mask, support, lat, lon = result

        # Step 1: 裁剪到目标区域
        sst_cropped, lat_cropped, lon_cropped = crop_data(sst, lat, lon)
        if sst_cropped is None:
            return False, "No data in target region"

        # Step 2: 重采样到目标分辨率
        sst_resampled = resample_to_target_grid(
            sst_cropped, lat_cropped, lon_cropped,
            lat_target, lon_target, method='linear'
        )

        # 处理缺失掩码（使用最近邻插值）
        missing_mask_resampled = None
        if missing_mask is not None:
            mask_cropped, _, _ = crop_data(missing_mask.astype(float), lat, lon)
            if mask_cropped is not None:
                missing_mask_resampled = resample_to_target_grid(
                    mask_cropped, lat_cropped, lon_cropped,
                    lat_target, lon_target, method='nearest'
                ).astype(np.int8)

        # 处理支持度
        support_resampled = None
        if support is not None:
            support_cropped, _, _ = crop_data(support, lat, lon)
            if support_cropped is not None:
                support_resampled = resample_to_target_grid(
                    support_cropped, lat_cropped, lon_cropped,
                    lat_target, lon_target, method='linear'
                )

        # Step 3: 保存为H5文件
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            # 主数据
            f.create_dataset('daily_sst', data=sst_resampled,
                           dtype=np.float32, compression='gzip', compression_opts=4)

            # 缺失掩码（如果存在）
            if missing_mask_resampled is not None:
                f.create_dataset('missing_mask', data=missing_mask_resampled,
                               dtype=np.int8, compression='gzip', compression_opts=4)

            # 支持度（如果存在）
            if support_resampled is not None:
                f.create_dataset('support', data=support_resampled,
                               dtype=np.float32, compression='gzip', compression_opts=4)

            # 坐标
            f.create_dataset('latitude', data=lat_target, dtype=np.float32)
            f.create_dataset('longitude', data=lon_target, dtype=np.float32)

            # 元数据
            f.attrs['source_file'] = input_path.name
            f.attrs['source_resolution'] = '0.02 degree'
            f.attrs['target_resolution'] = f'{TARGET_RESOLUTION} degree'
            f.attrs['lat_range'] = f'{LAT_MIN}-{LAT_MAX}'
            f.attrs['lon_range'] = f'{LON_MIN}-{LON_MAX}'
            f.attrs['interpolation_method'] = 'linear (sst), nearest (mask)'
            f.attrs['creation_date'] = datetime.now().isoformat()

            # 从文件名提取时间信息
            try:
                time_str = input_path.stem  # e.g., "20200115"
                f.attrs['timestamp'] = time_str
            except:
                pass

        # 计算统计信息
        valid_ratio = (~np.isnan(sst_resampled)).sum() / sst_resampled.size * 100

        return True, f"Shape: {sst_resampled.shape}, Valid: {valid_ratio:.1f}%"

    except Exception as e:
        return False, f"Error: {str(e)}"


# ============================================================
# Batch Processing Functions
# ============================================================

def scan_h5_files(input_dir: Path, year: int = None, month: int = None):
    """
    扫描H5文件

    Args:
        input_dir: 输入目录
        year: 指定年份（可选）
        month: 指定月份（可选）

    Returns:
        List of H5 file paths
    """
    if year is not None and month is not None:
        # 扫描指定年月
        target_dir = input_dir / f"{year}" / f"{month:02d}"
        if target_dir.exists():
            return sorted(target_dir.glob("*.h5"))
        else:
            return []
    elif year is not None:
        # 扫描指定年份的所有月份
        year_dir = input_dir / f"{year}"
        if year_dir.exists():
            return sorted(year_dir.rglob("*.h5"))
        else:
            return []
    else:
        # 扫描所有文件
        return sorted(input_dir.rglob("*.h5"))


def process_batch(year: int = None,
                  month: int = None,
                  num_workers: int = DEFAULT_NUM_WORKERS,
                  skip_existing: bool = True):
    """
    批量处理NC文件

    Args:
        year: 指定年份（可选）
        month: 指定月份（可选）
        num_workers: 并行工作进程数
        skip_existing: 是否跳过已存在的文件
    """
    print("=" * 70)
    print("SST H5 File Cropping and Resampling")
    print("=" * 70)
    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Lat range: {LAT_MIN}°N - {LAT_MAX}°N")
    print(f"Lon range: {LON_MIN}°E - {LON_MAX}°E")
    print(f"Target resolution: {TARGET_RESOLUTION}°")
    print(f"Workers: {num_workers}")
    print(f"Skip existing: {skip_existing}")
    if year:
        print(f"Year: {year}")
    if month:
        print(f"Month: {month}")
    print("=" * 70)

    # 获取目标网格
    lat_target, lon_target = get_target_grid_from_reference()
    print(f"\nTarget grid: lat({len(lat_target)}), lon({len(lon_target)})")

    # 扫描H5文件
    h5_files = scan_h5_files(INPUT_ROOT, year, month)

    if not h5_files:
        print("No H5 files found!")
        return

    print(f"Found {len(h5_files)} H5 files")

    # 准备任务
    tasks = []
    for input_path in h5_files:
        # 构建输出路径：保持年/月目录结构
        relative_path = input_path.relative_to(INPUT_ROOT)
        output_path = OUTPUT_ROOT / relative_path
        tasks.append((input_path, output_path, lat_target, lon_target, skip_existing))

    # 并行处理
    success_count = 0
    skip_count = 0
    fail_count = 0

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_h5_file, tasks),
            total=len(tasks),
            desc="Processing"
        ))

    # 统计结果
    for success, msg in results:
        if success:
            if "Skipped" in msg:
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 70)
    print("Processing complete!")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed:  {fail_count}")
    print(f"Output:  {OUTPUT_ROOT}")
    print("=" * 70)


def process_batch_years(year_start: int,
                        year_end: int,
                        month_start: int = 1,
                        month_end: int = 12,
                        num_workers: int = DEFAULT_NUM_WORKERS,
                        skip_existing: bool = True):
    """
    批量处理多年的H5文件

    Args:
        year_start: 起始年份
        year_end: 结束年份
        month_start: 起始月份
        month_end: 结束月份
        num_workers: 并行工作进程数
        skip_existing: 是否跳过已存在的文件
    """
    print("=" * 70)
    print("SST H5 File Cropping and Resampling")
    print("=" * 70)
    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Years:  {year_start} - {year_end}")
    print(f"Months: {month_start} - {month_end}")
    print(f"Lat range: {LAT_MIN}°N - {LAT_MAX}°N")
    print(f"Lon range: {LON_MIN}°E - {LON_MAX}°E")
    print(f"Target resolution: {TARGET_RESOLUTION}°")
    print(f"Workers: {num_workers}")
    print(f"Skip existing: {skip_existing}")
    print("=" * 70)

    # 获取目标网格
    lat_target, lon_target = get_target_grid_from_reference()
    print(f"\nTarget grid: lat({len(lat_target)}), lon({len(lon_target)})")

    total_success = 0
    total_skip = 0
    total_fail = 0

    for year in range(year_start, year_end + 1):
        for month in range(month_start, month_end + 1):
            # 扫描该年月的H5文件
            h5_files = scan_h5_files(INPUT_ROOT, year, month)

            if not h5_files:
                continue

            print(f"\nProcessing {year}-{month:02d}: {len(h5_files)} files")

            # 准备任务
            tasks = []
            for input_path in h5_files:
                relative_path = input_path.relative_to(INPUT_ROOT)
                output_path = OUTPUT_ROOT / relative_path
                tasks.append((input_path, output_path, lat_target, lon_target, skip_existing))

            # 并行处理
            success_count = 0
            skip_count = 0
            fail_count = 0

            with Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_single_h5_file, tasks),
                    total=len(tasks),
                    desc=f"  {year}-{month:02d}"
                ))

            # 统计结果
            for success, msg in results:
                if success:
                    if "Skipped" in msg:
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1

            total_success += success_count
            total_skip += skip_count
            total_fail += fail_count

            print(f"    Success: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")

    print("\n" + "=" * 70)
    print("All Processing Complete!")
    print(f"Total Success: {total_success}")
    print(f"Total Skipped: {total_skip}")
    print(f"Total Failed:  {total_fail}")
    print(f"Output:  {OUTPUT_ROOT}")
    print("=" * 70)


def process_single_file_cli(input_path: str, output_path: str = None):
    """
    处理单个文件（命令行接口）

    Args:
        input_path: 输入H5文件路径
        output_path: 输出H5文件路径（可选）
    """
    input_path = Path(input_path)

    if output_path:
        output_path = Path(output_path)
    else:
        output_path = input_path.parent / f"{input_path.stem}_target.h5"

    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")

    # 获取目标网格
    lat_target, lon_target = get_target_grid_from_reference()

    # 处理文件
    success, msg = process_single_h5_file(
        (input_path, output_path, lat_target, lon_target, False)
    )

    if success:
        print(f"Success: {msg}")
    else:
        print(f"Failed: {msg}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='SST H5 File Cropping and Resampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files
  python sst_cut_and_change_grid.py --workers 32

  # Process specific year range
  python sst_cut_and_change_grid.py --year_start 2016 --year_end 2024 --workers 16

  # Process specific year
  python sst_cut_and_change_grid.py --year_start 2020 --year_end 2020 --workers 32

  # Process specific month range
  python sst_cut_and_change_grid.py --year_start 2020 --year_end 2020 --month_start 1 --month_end 6 --workers 32

  # Process single file
  python sst_cut_and_change_grid.py --mode single --input /path/to/file.h5 --output /path/to/output.h5

  # Reprocess all (don't skip existing)
  python sst_cut_and_change_grid.py --no-skip --workers 32
"""
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['batch', 'single'],
        default='batch',
        help='Processing mode: batch (all files) or single (one file)'
    )

    parser.add_argument(
        '--year_start',
        type=int,
        default=None,
        help='Start year to process (for batch mode)'
    )

    parser.add_argument(
        '--year_end',
        type=int,
        default=None,
        help='End year to process (for batch mode)'
    )

    parser.add_argument(
        '--month_start',
        type=int,
        default=1,
        help='Start month to process (default: 1)'
    )

    parser.add_argument(
        '--month_end',
        type=int,
        default=12,
        help='End month to process (default: 12)'
    )

    # 保留旧参数以兼容
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='Year to process (deprecated, use --year_start and --year_end)'
    )

    parser.add_argument(
        '--month',
        type=int,
        default=None,
        help='Month to process (deprecated, use --month_start and --month_end)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_NUM_WORKERS})'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Alias for --workers'
    )

    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip existing files (reprocess all)'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input H5 file path (for single mode)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output H5 file path (for single mode, optional)'
    )

    args = parser.parse_args()

    # 处理 num_workers 别名
    num_workers = args.num_workers if args.num_workers else args.workers

    if args.mode == 'batch':
        # 处理年份范围
        if args.year_start is not None and args.year_end is not None:
            # 使用新的年份范围参数
            process_batch_years(
                year_start=args.year_start,
                year_end=args.year_end,
                month_start=args.month_start,
                month_end=args.month_end,
                num_workers=num_workers,
                skip_existing=not getattr(args, 'no_skip', False)
            )
        elif args.year is not None:
            # 兼容旧参数
            process_batch(
                year=args.year,
                month=args.month,
                num_workers=num_workers,
                skip_existing=not getattr(args, 'no_skip', False)
            )
        else:
            # 处理所有文件
            process_batch(
                year=None,
                month=None,
                num_workers=num_workers,
                skip_existing=not getattr(args, 'no_skip', False)
            )
    elif args.mode == 'single':
        if not args.input:
            print("Error: --input is required for single mode")
            return
        process_single_file_cli(args.input, args.output)


if __name__ == '__main__':
    main()
