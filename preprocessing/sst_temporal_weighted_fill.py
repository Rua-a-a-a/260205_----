"""JAXA SST Temporal Weighted Filling Module

This module implements temporal weighted filling algorithm for JAXA satellite SST data
to reduce missing values caused by cloud occlusion.

Algorithm:
    For each missing pixel at target time t, fill using weighted average of historical
    observations within lookback window:

    weight(t_history) = 1 / (t - t_history)
    filled_value = sum(w_i * v_i) / sum(w_i)

Typical usage example:

    # Generate full dataset with 24 time series
    python jaxa_temporal_weighted_filling.py --mode full --workers 216

    # Run small test (72 hours)
    python jaxa_temporal_weighted_filling.py --mode test --workers 32

    # Visualize single frame
    python jaxa_temporal_weighted_filling.py --mode visualize --hour 48

Author: Claude Code
Date: 2026-01-18
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm


# ============================================================
# Configuration Constants
# ============================================================

JAXA_ROOT = Path('/data_new/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3/')
OUTPUT_ROOT = Path('/data_new/chla_data_imputation_data_260125/chla_data_pretraining/sst_daily_fusion_modified/')
TEST_OUTPUT_DIR = Path('/home/ccy/260205_缺失填充/data/processed_test')

START_DATE = datetime(2024, 1, 1, 0, 0, 0)  # Will be overridden by command line args
TOTAL_HOURS = 720  # 30 days
TEST_HOURS = 72    # 3 days for testing
INTERVAL = 24      # 24 hour interval between frames
NUM_SERIES = 24    # Number of time series (starting hours 0-23)
LOOKBACK_WINDOW = 48  # Maximum lookback window in hours
UTC_HOUR = 4       # Fixed UTC hour for data generation

DEFAULT_NUM_WORKERS = 216


# ============================================================
# Core Data Loading Functions
# ============================================================

def load_jaxa_frame(target_time: datetime) -> Tuple[Optional[np.ndarray],
                                                      Optional[np.ndarray],
                                                      Optional[np.ndarray]]:
    """Load JAXA SST data for specified time.

    Args:
        target_time: Target datetime to load.

    Returns:
        Tuple of (sst, lat, lon):
            - sst: SST array in Kelvin, shape (H, W), or None if file not found
            - lat: Latitude array, shape (H,), or None
            - lon: Longitude array, shape (W,), or None

    Note:
        Expected file structure:
        /data/sst_data/.../jaxa_extract_L3/YYYYMM/DD/YYYYMMDDHHMMSS.nc
    """
    date_str = target_time.strftime('%Y%m')
    day_str = target_time.strftime('%d')
    file_str = target_time.strftime('%Y%m%d%H%M%S')
    file_path = JAXA_ROOT / date_str / day_str / f'{file_str}.nc'

    if not file_path.exists():
        return None, None, None

    try:
        ds = xr.open_dataset(file_path)
        sst = ds.sea_surface_temperature.values
        if len(sst.shape) == 3:
            sst = sst[0]
        lat = ds.lat.values
        lon = ds.lon.values
        ds.close()
        return sst, lat, lon
    except Exception as e:
        print(f"    ⚠️  Failed to load: {file_path.name} - {e}")
        return None, None, None


def preload_history_frames(start_hour: int,
                          target_hour: int,
                          start_date: datetime = None) -> Dict[int, np.ndarray]:
    """Preload historical frames to memory for efficient access.

    Args:
        start_hour: Starting hour of the time series (for boundary checking).
        target_hour: Target hour to fill (relative to start_date).
        start_date: Starting date for the time series (overrides global START_DATE).

    Returns:
        Dictionary mapping hour -> SST array for all loaded historical frames.

    Strategy:
        1. First try 24-hour lookback window
        2. If insufficient data (< 10 frames), extend to 48-hour window
    """
    if start_date is None:
        start_date = START_DATE

    history_frames = {}

    # Try 24-hour lookback
    lookback_start = max(start_hour, target_hour - 24)
    for t in range(lookback_start, target_hour):
        t_time = start_date + timedelta(hours=t)
        sst, _, _ = load_jaxa_frame(t_time)
        if sst is not None:
            history_frames[t] = sst

    # Extend to 48-hour if needed
    if target_hour >= start_hour + 48 and len(history_frames) < 10:
        lookback_start = max(start_hour, target_hour - 48)
        for t in range(lookback_start, target_hour - 24):
            if t not in history_frames:
                t_time = start_date + timedelta(hours=t)
                sst, _, _ = load_jaxa_frame(t_time)
                if sst is not None:
                    history_frames[t] = sst

    return history_frames


# ============================================================
# Temporal Weighted Filling Algorithm
# ============================================================

def fill_single_position(args: Tuple[int, int, int, Dict[int, np.ndarray]]
                        ) -> Optional[Tuple[int, int, float, int]]:
    """Fill single missing position using temporal weighted average.

    This function is designed for parallel processing with multiprocessing.Pool.

    Args:
        args: Tuple of (x, y, target_hour, history_frames)
            - x: Row index of missing pixel
            - y: Column index of missing pixel
            - target_hour: Target hour to fill
            - history_frames: Dict mapping hour -> SST array

    Returns:
        Tuple of (x, y, filled_value, num_sources) if filling successful,
        None otherwise.

    Algorithm:
        weight(t_history) = 1 / (target_hour - t_history)
        filled_value = sum(w_i * v_i) / sum(w_i)
    """
    x, y, target_hour, history_frames = args

    weights = []
    values = []

    for t, t_sst in history_frames.items():
        if not np.isnan(t_sst[x, y]):
            time_distance = target_hour - t
            weight = 1.0 / time_distance
            weights.append(weight)
            values.append(t_sst[x, y])

    if len(weights) > 0:
        filled_value = sum(w * v for w, v in zip(weights, values)) / sum(weights)
        return (x, y, filled_value, len(weights))

    return None


def process_single_frame(start_hour: int,
                        target_hour: int,
                        num_workers: int = DEFAULT_NUM_WORKERS,
                        verbose: bool = False,
                        start_date: datetime = None) -> Tuple[Optional[np.ndarray],
                                                         Optional[np.ndarray],
                                                         Optional[np.ndarray],
                                                         Optional[datetime]]:
    """Process single frame with temporal weighted filling.

    Args:
        start_hour: Starting hour of the time series.
        target_hour: Target hour to process.
        num_workers: Number of CPU cores for parallel processing.
        verbose: Whether to print detailed progress.
        start_date: Starting date for the time series (overrides global START_DATE).

    Returns:
        Tuple of (filled_sst, original_sst, fill_info, target_time):
            - filled_sst: SST after filling, shape (H, W)
            - original_sst: Original SST before filling
            - fill_info: Number of source frames used for each pixel (uint8)
            - target_time: Target datetime

    Note:
        If target_hour == start_hour (first frame), no filling is performed.
    """
    if start_date is None:
        start_date = START_DATE

    target_time = start_date + timedelta(hours=target_hour)

    # Load target frame
    target_sst, lat, lon = load_jaxa_frame(target_time)
    if target_sst is None:
        return None, None, None, None

    original_sst = target_sst.copy()
    filled_sst = target_sst.copy()
    fill_info = np.zeros_like(target_sst, dtype=np.uint8)

    # Skip filling for initial frame
    if target_hour == start_hour:
        return filled_sst, original_sst, fill_info, target_time

    # Find missing positions
    missing_mask = np.isnan(target_sst)
    missing_positions = np.argwhere(missing_mask)

    if len(missing_positions) == 0:
        return filled_sst, original_sst, fill_info, target_time

    # Preload historical frames
    history_frames = preload_history_frames(start_hour, target_hour, start_date)

    # Prepare parallel tasks
    tasks = [(x, y, target_hour, history_frames) for x, y in missing_positions]

    # Parallel processing
    with Pool(num_workers) as pool:
        results = list(pool.imap(
            fill_single_position,
            tasks,
            chunksize=max(1, len(tasks) // (num_workers * 4))
        ))

    # Apply filling results
    for result in results:
        if result is not None:
            x, y, filled_value, num_sources = result
            filled_sst[x, y] = filled_value
            fill_info[x, y] = min(num_sources, 255)  # uint8 limit

    return filled_sst, original_sst, fill_info, target_time


# ============================================================
# Dataset Generation Functions
# ============================================================

def generate_time_series(series_id: int,
                        num_workers: int = DEFAULT_NUM_WORKERS,
                        total_hours: int = TOTAL_HOURS,
                        interval: int = INTERVAL,
                        start_date: datetime = None) -> Dict:
    """Generate complete time series with temporal weighted filling.

    Args:
        series_id: Series ID (0-23), corresponding to starting hour offset.
        num_workers: Number of CPU cores for parallel processing.
        total_hours: Total hours to process (default 720 = 30 days).
        interval: Interval between frames in hours (default 24).
        start_date: Starting date for the time series (overrides global START_DATE).

    Returns:
        Dictionary containing statistics:
            - series_id: Series ID
            - start_hour: Starting hour
            - num_frames: Number of frames generated
            - original_missing_rate_avg: Average missing rate before filling (%)
            - filled_missing_rate_avg: Average missing rate after filling (%)
            - improvement_avg: Average missing rate reduction (%)
            - file_path: Path to output HDF5 file
            - file_size_mb: File size in MB

    Output:
        HDF5 files saved to OUTPUT_ROOT/YYYY/MM/YYYYMMDD.h5

    HDF5 structure:
        Datasets:
            - sst_data: (T, H, W) float32, SST in Kelvin
            - missing_mask: (T, H, W) uint8, 1=missing, 0=valid
            - fill_mask: (T, H, W) uint8, 1=filled, 0=original
            - latitude: (H,) float32
            - longitude: (W,) float32
            - timestamps: (T,) S32, ISO format timestamps
        Attributes:
            - series_id, start_hour, num_frames, interval, shape, creation_date
    """
    if start_date is None:
        start_date = START_DATE

    print(f"\n{'='*60}")
    print(f"处理时间序列 #{series_id:02d} (起点: {series_id}h)")
    print(f"开始日期: {start_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    start_hour = 0  # 从0开始，因为start_date已经包含了utc_hour
    target_hours = list(range(start_hour, total_hours, interval))
    num_frames = len(target_hours)

    print(f"  目标帧数: {num_frames}")
    print(f"  时间范围: {start_hour}h - {target_hours[-1]}h")

    # Initialize storage
    sst_data = []
    missing_masks = []
    fill_masks = []
    timestamps = []

    original_missing_rates = []
    filled_missing_rates = []

    # Load first frame coordinates
    first_time = start_date + timedelta(hours=start_hour)
    _, lat, lon = load_jaxa_frame(first_time)

    # Process each frame
    for idx, target_hour in enumerate(tqdm(target_hours, desc=f"  序列#{series_id:02d}")):
        target_time = start_date + timedelta(hours=target_hour)

        filled_sst, original_sst, fill_info, _ = process_single_frame(
            start_hour, target_hour, num_workers, start_date=start_date
        )

        if filled_sst is None:
            print(f"    ⚠️  跳过 {target_hour}h 帧（加载失败）")
            continue

        # Statistics
        original_missing = np.isnan(original_sst)
        filled_missing = np.isnan(filled_sst)

        original_rate = original_missing.sum() / original_sst.size * 100
        filled_rate = filled_missing.sum() / filled_sst.size * 100

        original_missing_rates.append(original_rate)
        filled_missing_rates.append(filled_rate)

        # Save data
        sst_data.append(filled_sst)
        missing_masks.append(filled_missing.astype(np.uint8))
        fill_masks.append((fill_info > 0).astype(np.uint8))
        timestamps.append(target_time.isoformat())

    # Convert to numpy arrays
    sst_data = np.array(sst_data, dtype=np.float32)
    missing_masks = np.array(missing_masks, dtype=np.uint8)
    fill_masks = np.array(fill_masks, dtype=np.uint8)

    # Save to HDF5 with new naming convention: YYYY/MM/YYYYMMDD.h5
    output_files = []
    for idx, timestamp_str in enumerate(timestamps):
        target_time = datetime.fromisoformat(timestamp_str)
        year_dir = OUTPUT_ROOT / f'{target_time.year}'
        month_dir = year_dir / f'{target_time.month:02d}'
        month_dir.mkdir(exist_ok=True, parents=True)

        output_file = month_dir / f'{target_time.strftime("%Y%m%d")}.h5'

        with h5py.File(output_file, 'w') as f:
            # Single frame data
            f.create_dataset('sst_data', data=sst_data[idx], compression='gzip', compression_opts=4)
            f.create_dataset('missing_mask', data=missing_masks[idx], compression='gzip', compression_opts=4)
            f.create_dataset('fill_mask', data=fill_masks[idx], compression='gzip', compression_opts=4)

            # Coordinates
            f.create_dataset('latitude', data=lat, dtype=np.float32)
            f.create_dataset('longitude', data=lon, dtype=np.float32)

            # Timestamp
            f.create_dataset('timestamp', data=np.array([timestamp_str], dtype='S32'))

            # Metadata
            f.attrs['series_id'] = series_id
            f.attrs['utc_hour'] = UTC_HOUR
            f.attrs['date'] = target_time.strftime('%Y-%m-%d')
            f.attrs['shape'] = sst_data[idx].shape
            f.attrs['creation_date'] = datetime.now().isoformat()

        output_files.append(output_file)

    # Return statistics
    total_size_mb = sum(f.stat().st_size for f in output_files) / 1024 / 1024

    stats = {
        'series_id': series_id,
        'start_hour': start_hour,
        'num_frames': len(sst_data),
        'original_missing_rate_avg': float(np.mean(original_missing_rates)),
        'filled_missing_rate_avg': float(np.mean(filled_missing_rates)),
        'improvement_avg': float(np.mean(np.array(original_missing_rates) - np.array(filled_missing_rates))),
        'output_dir': str(OUTPUT_ROOT),
        'file_size_mb': total_size_mb
    }

    print(f"  ✓ 序列 #{series_id:02d} 完成")
    print(f"    - 帧数: {len(sst_data)}")
    print(f"    - 平均缺失率: {stats['original_missing_rate_avg']:.2f}% → {stats['filled_missing_rate_avg']:.2f}%")
    print(f"    - 总文件大小: {stats['file_size_mb']:.1f} MB")

    return stats


def generate_batch_years(year_start: int,
                        year_end: int,
                        utc_hour: int = UTC_HOUR,
                        num_workers: int = DEFAULT_NUM_WORKERS):
    """Generate SST data for multiple years with fixed UTC hour.

    Args:
        year_start: Starting year (inclusive).
        year_end: Ending year (inclusive).
        utc_hour: Fixed UTC hour for all data (default: 4).
        num_workers: Number of CPU cores for parallel processing.

    Output:
        Files saved to OUTPUT_ROOT/YYYY/MM/YYYYMMDD.h5
    """
    print("=" * 60)
    print("JAXA批量年份数据生成 - 时间加权填充")
    print("=" * 60)
    print(f"数据源: {JAXA_ROOT}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"年份范围: {year_start} - {year_end}")
    print(f"UTC时间: {utc_hour:02d}:00")
    print(f"并行核心数: {num_workers}")
    print()

    all_stats = []

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            # Calculate days in month
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            days_in_month = (next_month - datetime(year, month, 1)).days

            # Process each month (30 days fixed)
            start_date = datetime(year, month, 1, utc_hour, 0, 0)
            total_hours = min(30, days_in_month) * 24  # 30 days or less

            print(f"\n{'='*60}")
            print(f"处理 {year}-{month:02d} (UTC {utc_hour:02d}:00)")
            print(f"{'='*60}")

            try:
                stats = generate_time_series(
                    series_id=utc_hour,
                    num_workers=num_workers,
                    total_hours=total_hours,
                    interval=24,
                    start_date=start_date
                )
                all_stats.append({
                    'year': year,
                    'month': month,
                    **stats
                })
            except Exception as e:
                print(f"  ✗ {year}-{month:02d} 失败: {e}")
                continue

    # Save statistics
    stats_file = OUTPUT_ROOT / f'batch_statistics_{year_start}_{year_end}_utc{utc_hour:02d}.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'year_start': year_start,
            'year_end': year_end,
            'utc_hour': utc_hour,
            'total_months': len(all_stats),
            'months': all_stats,
            'creation_date': datetime.now().isoformat(),
            'parameters': {
                'lookback_window': LOOKBACK_WINDOW,
                'num_workers': num_workers
            }
        }, f, indent=2)

    print(f"\n✓ 统计信息已保存: {stats_file.name}")

    # Final summary
    print("\n" + "=" * 60)
    print("批量生成完成！")
    print("=" * 60)
    print(f"处理月份数: {len(all_stats)}")
    print(f"总帧数: {sum(s['num_frames'] for s in all_stats)}")
    print(f"总文件大小: {sum(s['file_size_mb'] for s in all_stats):.1f} MB")
    if all_stats:
        print(f"平均缺失率改善: {np.mean([s['improvement_avg'] for s in all_stats]):.2f}%")
    print()


def generate_full_dataset(num_workers: int = DEFAULT_NUM_WORKERS):
    """Generate full JAXA weighted dataset with all 24 time series.

    Args:
        num_workers: Number of CPU cores for parallel processing.

    Output:
        Files saved to OUTPUT_ROOT:
            - jaxa_weighted_series_00.h5 through jaxa_weighted_series_23.h5
            - dataset_statistics.json
            - dataset_statistics.png
    """
    print("=" * 60)
    print("JAXA完整数据集生成 - 时间加权填充")
    print("=" * 60)
    print(f"数据源: {JAXA_ROOT}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"时间范围: {START_DATE} + {TOTAL_HOURS}小时")
    print(f"时间序列数: {NUM_SERIES}")
    print(f"并行核心数: {num_workers}")
    print()

    # Generate all time series
    all_stats = []

    for series_id in range(NUM_SERIES):
        try:
            stats = generate_time_series(series_id, num_workers)
            all_stats.append(stats)
        except Exception as e:
            print(f"  ✗ 序列 #{series_id:02d} 失败: {e}")
            continue

    # Save statistics
    stats_file = OUTPUT_ROOT / 'dataset_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'num_series': len(all_stats),
            'total_frames': sum(s['num_frames'] for s in all_stats),
            'series': all_stats,
            'creation_date': datetime.now().isoformat(),
            'parameters': {
                'start_date': START_DATE.isoformat(),
                'total_hours': TOTAL_HOURS,
                'interval': INTERVAL,
                'lookback_window': LOOKBACK_WINDOW,
                'num_workers': num_workers
            }
        }, f, indent=2)

    print(f"\n✓ 统计信息已保存: {stats_file.name}")

    # Plot statistics
    plot_overall_statistics(all_stats)

    # Final summary
    print("\n" + "=" * 60)
    print("数据集生成完成！")
    print("=" * 60)
    print(f"总时间序列: {len(all_stats)}")
    print(f"总帧数: {sum(s['num_frames'] for s in all_stats)}")
    print(f"总文件大小: {sum(s['file_size_mb'] for s in all_stats):.1f} MB")
    print(f"平均缺失率改善: {np.mean([s['improvement_avg'] for s in all_stats]):.2f}%")
    print()


# ============================================================
# Visualization Functions
# ============================================================

def plot_overall_statistics(all_stats: List[Dict]):
    """Plot overall statistics for all time series.

    Args:
        all_stats: List of statistics dictionaries from generate_time_series().

    Output:
        PNG file saved to OUTPUT_ROOT/dataset_statistics.png
    """
    print("\n生成统计图表...")

    series_ids = [s['series_id'] for s in all_stats]
    original_rates = [s['original_missing_rate_avg'] for s in all_stats]
    filled_rates = [s['filled_missing_rate_avg'] for s in all_stats]
    improvements = [s['improvement_avg'] for s in all_stats]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Missing rate comparison
    ax = axes[0, 0]
    x = np.arange(len(series_ids))
    width = 0.35
    ax.bar(x - width/2, original_rates, width, label='Original', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, filled_rates, width, label='After Filling', color='#27ae60', alpha=0.8)
    ax.set_xlabel('Series ID (Start Hour)', fontsize=12)
    ax.set_ylabel('Missing Rate (%)', fontsize=12)
    ax.set_title('Missing Rate by Time Series', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}h' for i in series_ids])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Improvement
    ax = axes[0, 1]
    ax.plot(series_ids, improvements, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax.fill_between(series_ids, improvements, alpha=0.3, color='#3498db')
    ax.set_xlabel('Series ID (Start Hour)', fontsize=12)
    ax.set_ylabel('Missing Rate Reduction (%)', fontsize=12)
    ax.set_title('Improvement by Time Series', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.mean(improvements), color='red', linestyle='--',
               label=f'Avg: {np.mean(improvements):.2f}%')
    ax.legend()

    # 3. File size
    ax = axes[1, 0]
    file_sizes = [s['file_size_mb'] for s in all_stats]
    ax.bar(series_ids, file_sizes, color='#9b59b6', alpha=0.8)
    ax.set_xlabel('Series ID (Start Hour)', fontsize=12)
    ax.set_ylabel('File Size (MB)', fontsize=12)
    ax.set_title('HDF5 File Size by Series', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
JAXA Weighted Aligned Dataset Summary
{'='*50}

Total Time Series: {len(all_stats)}
Total Frames: {sum(s['num_frames'] for s in all_stats)}
Time Range: 0h - {TOTAL_HOURS}h (30 days)
Interval: {INTERVAL}h

Missing Rate Statistics:
  • Original Avg: {np.mean(original_rates):.2f}%
  • Filled Avg: {np.mean(filled_rates):.2f}%
  • Improvement: {np.mean(improvements):.2f}%

  • Best Series: #{series_ids[np.argmax(improvements)]}
    (Reduction: {max(improvements):.2f}%)
  • Worst Series: #{series_ids[np.argmin(improvements)]}
    (Reduction: {min(improvements):.2f}%)

File Statistics:
  • Total Size: {sum(file_sizes):.1f} MB
  • Avg Size per Series: {np.mean(file_sizes):.1f} MB
  • Compression: gzip level 4

Processing:
  • Algorithm: Temporal Weighted Filling
  • Weight Function: w = 1/dt
  • Lookback Window: {LOOKBACK_WINDOW}h
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    output_path = OUTPUT_ROOT / 'dataset_statistics.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 统计图表已保存: {output_path.name}")


def run_test(num_workers: int = 32):
    """Run small-scale test on first 72 hours.

    Args:
        num_workers: Number of CPU cores for parallel processing.

    Output:
        Visualization files saved to TEST_OUTPUT_DIR
    """
    TEST_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("JAXA 时间加权填充算法 - 测试模式")
    print("=" * 60)
    print(f"测试时间范围: {START_DATE.strftime('%Y-%m-%d')} + {TEST_HOURS}小时")
    print(f"目标帧: 0h, 24h, 48h, 72h")
    print(f"并行核心数: {num_workers}")
    print(f"输出目录: {TEST_OUTPUT_DIR}")
    print()

    target_hours = [0, 24, 48, 72]
    original_rates = []
    filled_rates = []

    for hour in target_hours:
        print(f"\n{'='*60}")
        print(f"处理 {hour}h 帧")
        print(f"{'='*60}")

        if hour == 0:
            sst, lat, lon = load_jaxa_frame(START_DATE)
            original_sst = sst
            filled_sst = sst
            fill_info = np.zeros_like(sst, dtype=int)

            missing_rate = np.isnan(sst).sum() / sst.size * 100
            original_rates.append(missing_rate)
            filled_rates.append(missing_rate)

            print(f"  0h 帧（初始帧，不填充）")
            print(f"  缺失率: {missing_rate:.2f}%")
        else:
            filled_sst, original_sst, fill_info, _ = process_single_frame(
                0, hour, num_workers, verbose=True
            )

            if filled_sst is None:
                print(f"  ⚠️  跳过 {hour}h 帧（数据加载失败）")
                continue

            original_rate = np.isnan(original_sst).sum() / original_sst.size * 100
            filled_rate = np.isnan(filled_sst).sum() / filled_sst.size * 100

            original_rates.append(original_rate)
            filled_rates.append(filled_rate)

            print(f"  缺失率改善: {original_rate:.2f}% → {filled_rate:.2f}% (减少 {original_rate - filled_rate:.2f}%)")

    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"{'='*60}")
    print(f"输出目录: {TEST_OUTPUT_DIR}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='JAXA SST Temporal Weighted Filling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate batch years (2016-2024) with fixed UTC hour
  python sst_temporal_weighted_fill.py --mode batch --year_start 2016 --year_end 2024 --utc_hour 4 --workers 216

  # Generate single month
  python sst_temporal_weighted_fill.py --mode single --year 2024 --month 1 --utc_hour 4 --workers 216

  # Run small test (72 hours)
  python sst_temporal_weighted_fill.py --mode test --workers 32
"""
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['batch', 'single', 'test', 'full'],
        default='batch',
        help='Execution mode: batch (multiple years), single (one month), test (72h), or full (legacy 24 series)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f'Number of CPU cores for parallel processing (default: {DEFAULT_NUM_WORKERS})'
    )

    parser.add_argument(
        '--year_start',
        type=int,
        default=2016,
        help='Starting year for batch mode (default: 2016)'
    )

    parser.add_argument(
        '--year_end',
        type=int,
        default=2024,
        help='Ending year for batch mode (default: 2024)'
    )

    parser.add_argument(
        '--year',
        type=int,
        default=2024,
        help='Year for single mode (default: 2024)'
    )

    parser.add_argument(
        '--month',
        type=int,
        default=1,
        help='Month for single mode (1-12, default: 1)'
    )

    parser.add_argument(
        '--utc_hour',
        type=int,
        default=UTC_HOUR,
        help=f'Fixed UTC hour for data generation (0-23, default: {UTC_HOUR})'
    )

    parser.add_argument(
        '--series',
        type=int,
        default=0,
        help='Series ID for legacy full mode (0-23, default: 0)'
    )

    args = parser.parse_args()

    if args.mode == 'batch':
        generate_batch_years(
            year_start=args.year_start,
            year_end=args.year_end,
            utc_hour=args.utc_hour,
            num_workers=args.workers
        )
    elif args.mode == 'single':
        if not (1 <= args.month <= 12):
            print(f"Error: Month must be 1-12")
            sys.exit(1)
        if not (0 <= args.utc_hour <= 23):
            print(f"Error: UTC hour must be 0-23")
            sys.exit(1)

        # Calculate days in month
        year, month = args.year, args.month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        days_in_month = (next_month - datetime(year, month, 1)).days

        start_date = datetime(year, month, 1, args.utc_hour, 0, 0)
        total_hours = min(30, days_in_month) * 24

        stats = generate_time_series(
            series_id=args.utc_hour,
            num_workers=args.workers,
            total_hours=total_hours,
            interval=24,
            start_date=start_date
        )
        print(f"\n✓ 生成完成: {stats['output_dir']}")
    elif args.mode == 'test':
        run_test(num_workers=args.workers)
    elif args.mode == 'full':
        generate_full_dataset(num_workers=args.workers)


if __name__ == '__main__':
    main()
