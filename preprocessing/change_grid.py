"""
SST数据空间分辨率重采样脚本
将0.02°分辨率的SST数据重采样为0.05°分辨率，与叶绿素a数据对齐
"""

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
from datetime import datetime


def find_date_in_files(base_path: str, target_date: str) -> tuple:
    """
    在所有h5文件中查找目标日期

    Args:
        base_path: h5文件所在目录
        target_date: 目标日期，格式为 'YYYYMMDD'

    Returns:
        (文件路径, 时间索引) 或 (None, None) 如果未找到
    """
    target_dt = datetime.strptime(target_date, '%Y%m%d')
    target_str = target_dt.strftime('%Y-%m-%dT00:00:00')

    files = sorted([f for f in os.listdir(base_path)
                   if f.endswith('.h5') and f.startswith('jaxa_weighted_series')])

    for fname in files:
        fpath = os.path.join(base_path, fname)
        with h5py.File(fpath, 'r') as f:
            timestamps = f['timestamps'][:]
            for idx, ts in enumerate(timestamps):
                ts_str = ts.decode() if isinstance(ts, bytes) else ts
                if ts_str == target_str:
                    return fpath, idx

    return None, None


def resample_sst_to_chla_grid(sst_data: np.ndarray,
                              lat_sst: np.ndarray,
                              lon_sst: np.ndarray,
                              lat_chla: np.ndarray,
                              lon_chla: np.ndarray,
                              method: str = 'linear') -> np.ndarray:
    """
    将SST数据重采样到CHLA网格

    Args:
        sst_data: SST数据数组 (lat, lon)
        lat_sst: SST纬度数组
        lon_sst: SST经度数组
        lat_chla: 目标CHLA纬度数组
        lon_chla: 目标CHLA经度数组
        method: 插值方法，'linear' 或 'nearest'

    Returns:
        重采样后的数据数组
    """
    # 创建插值器
    interpolator = RegularGridInterpolator(
        (lat_sst, lon_sst),
        sst_data,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )

    # 创建目标网格
    lon_grid, lat_grid = np.meshgrid(lon_chla, lat_chla)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    # 执行插值
    resampled = interpolator(points).reshape(len(lat_chla), len(lon_chla))

    return resampled


def process_single_date(input_path: str,
                        output_path: str,
                        target_date: str,
                        lat_chla: np.ndarray,
                        lon_chla: np.ndarray):
    """
    处理单个日期的SST数据

    Args:
        input_path: 输入h5文件路径
        output_path: 输出目录
        target_date: 目标日期 'YYYYMMDD'
        lat_chla: 目标纬度数组
        lon_chla: 目标经度数组
    """
    # 查找日期所在文件和索引
    file_path, time_idx = find_date_in_files(input_path, target_date)

    if file_path is None:
        print(f"警告: 未找到日期 {target_date} 的数据")
        return

    print(f"处理日期: {target_date}")
    print(f"  源文件: {os.path.basename(file_path)}, 时间索引: {time_idx}")

    with h5py.File(file_path, 'r') as f:
        # 读取原始数据
        sst_data = f['sst_data'][time_idx, :, :]
        lat_sst = f['latitude'][:]
        lon_sst = f['longitude'][:]

        # 读取mask数据（如果存在）
        missing_mask = f['missing_mask'][time_idx, :, :] if 'missing_mask' in f else None
        fill_mask = f['fill_mask'][time_idx, :, :] if 'fill_mask' in f else None

    # 重采样SST数据
    sst_resampled = resample_sst_to_chla_grid(
        sst_data, lat_sst, lon_sst, lat_chla, lon_chla, method='linear'
    )

    # 重采样mask数据（使用最近邻插值）
    if missing_mask is not None:
        missing_mask_resampled = resample_sst_to_chla_grid(
            missing_mask.astype(float), lat_sst, lon_sst, lat_chla, lon_chla, method='nearest'
        ).astype(np.int8)
    else:
        missing_mask_resampled = None

    if fill_mask is not None:
        fill_mask_resampled = resample_sst_to_chla_grid(
            fill_mask.astype(float), lat_sst, lon_sst, lat_chla, lon_chla, method='nearest'
        ).astype(np.uint8)
    else:
        fill_mask_resampled = None

    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"sst_resampled_{target_date}.h5")

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('sst_data', data=sst_resampled, dtype=np.float32)
        f.create_dataset('latitude', data=lat_chla, dtype=np.float32)
        f.create_dataset('longitude', data=lon_chla, dtype=np.float32)
        f.create_dataset('date', data=target_date.encode())

        if missing_mask_resampled is not None:
            f.create_dataset('missing_mask', data=missing_mask_resampled, dtype=np.int8)
        if fill_mask_resampled is not None:
            f.create_dataset('fill_mask', data=fill_mask_resampled, dtype=np.uint8)

        # 添加元数据
        f.attrs['source_resolution'] = '0.02 degree'
        f.attrs['target_resolution'] = '0.05 degree'
        f.attrs['interpolation_method'] = 'linear (sst), nearest (masks)'
        f.attrs['source_file'] = os.path.basename(file_path)

    print(f"  输出文件: {output_file}")
    print(f"  输出形状: {sst_resampled.shape}")
    print(f"  有效数据比例: {(~np.isnan(sst_resampled)).sum() / sst_resampled.size * 100:.2f}%")


def main():
    # 配置路径
    input_path = "/data_new/chla_data_imputation_data_260125/SST/jaxa_weighted_aligned/"
    output_path = "/home/ccy/260205_缺失填充/datasets/processed_grid"
    chla_reference_path = "/data_new/chla_data_imputation_data_260125/chla_data_pretraining/daily_fusion_target_modified/2015/07/20150701.h5"

    # 目标日期列表
    target_dates = ['20240101', '20240401', '20240701', '20241001']

    # 从CHLA参考文件读取目标网格坐标，确保完全对齐
    with h5py.File(chla_reference_path, 'r') as f:
        lat_chla = f['latitude'][:]
        lon_chla = f['longitude'][:]

    print("=" * 60)
    print("SST数据空间分辨率重采样")
    print("=" * 60)
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    print(f"目标分辨率: 0.05°")
    print(f"目标网格: lat({len(lat_chla)}), lon({len(lon_chla)})")
    print(f"待处理日期: {target_dates}")
    print("=" * 60)

    # 处理每个日期
    for date in target_dates:
        process_single_date(input_path, output_path, date, lat_chla, lon_chla)
        print()

    print("=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
