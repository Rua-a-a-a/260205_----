#!/usr/bin/env python3
"""
H5文件经纬度裁剪脚本

功能：将H5文件裁剪到指定的经纬度范围
- 输入目录：/data_new/chla_data_imputation_data_260125/chla_data_pretraining/daily_fusion
- 输出目录：/data_new/chla_data_imputation_data_260125/chla_data_pretraining/daily_fusion_target_modified
- 纬度范围：15.0°N - 24.0°N
- 经度范围：111.0°E - 118.0°E

Author: Claude
Date: 2026-02-02
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# 配置
INPUT_DIR = Path('/data_new/chla_data_imputation_data_260125/chla_data_pretraining/daily_fusion/')
OUTPUT_DIR = Path('/data_new/chla_data_imputation_data_260125/chla_data_pretraining/daily_fusion_target_modified/')

# 裁剪范围
LAT_MIN = 15.0  # 南边界
LAT_MAX = 24.0  # 北边界
LON_MIN = 111.0  # 西边界
LON_MAX = 118.0  # 东边界


def crop_h5_file(input_path, output_path):
    """
    裁剪单个H5文件到指定经纬度范围

    Args:
        input_path: 输入H5文件路径
        output_path: 输出H5文件路径
    """
    try:
        with h5py.File(input_path, 'r') as f_in:
            # 读取经纬度
            lat = f_in['latitude'][:]
            lon = f_in['longitude'][:]

            # 找到裁剪范围的索引
            lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
            lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)

            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]

            if len(lat_idx) == 0 or len(lon_idx) == 0:
                return False, "No data in target region"

            # 裁剪经纬度
            lat_cropped = lat[lat_idx]
            lon_cropped = lon[lon_idx]

            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 创建输出文件
            with h5py.File(output_path, 'w') as f_out:
                # 裁剪并保存所有2D数据集
                for key in f_in.keys():
                    if key in ['latitude', 'longitude']:
                        continue

                    data = f_in[key][:]

                    # 如果是2D数据，进行裁剪
                    if data.ndim == 2:
                        data_cropped = data[lat_idx[0]:lat_idx[-1]+1,
                                           lon_idx[0]:lon_idx[-1]+1]
                        f_out.create_dataset(key, data=data_cropped, compression='gzip')
                    else:
                        # 1D或其他维度数据直接复制
                        f_out.create_dataset(key, data=data, compression='gzip')

                # 保存裁剪后的经纬度
                f_out.create_dataset('latitude', data=lat_cropped)
                f_out.create_dataset('longitude', data=lon_cropped)

                # 复制属性
                for attr_key, attr_val in f_in.attrs.items():
                    f_out.attrs[attr_key] = attr_val

                # 添加裁剪信息
                f_out.attrs['cropped'] = True
                f_out.attrs['lat_range'] = f"{LAT_MIN}-{LAT_MAX}"
                f_out.attrs['lon_range'] = f"{LON_MIN}-{LON_MAX}"

        return True, f"Cropped to {lat_cropped.shape[0]}x{lon_cropped.shape[0]}"

    except Exception as e:
        return False, f"Error: {str(e)}"


def process_all_files():
    """处理所有H5文件"""

    print("=" * 70)
    print("H5 File Cropping")
    print("=" * 70)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Lat range: {LAT_MIN}°N - {LAT_MAX}°N")
    print(f"Lon range: {LON_MIN}°E - {LON_MAX}°E")
    print("=" * 70)

    # 查找所有H5文件
    h5_files = list(INPUT_DIR.rglob("*.h5"))

    if not h5_files:
        print("No H5 files found!")
        return

    print(f"\nFound {len(h5_files)} H5 files")

    # 处理每个文件
    success_count = 0
    fail_count = 0

    for input_path in tqdm(h5_files, desc="Processing"):
        # 构建输出路径（保持相同的目录结构）
        relative_path = input_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path

        # 裁剪文件
        success, msg = crop_h5_file(input_path, output_path)

        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"\nFailed: {input_path.name} - {msg}")

    print("\n" + "=" * 70)
    print(f"Processing complete!")
    print(f"Success: {success_count}")
    print(f"Failed:  {fail_count}")
    print("=" * 70)


def main():
    process_all_files()


if __name__ == '__main__':
    main()
