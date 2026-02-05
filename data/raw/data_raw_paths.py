#!/usr/bin/env python3
"""
原始数据路径配置

记录项目使用的原始数据路径，便于统一管理和移植。
"""

from pathlib import Path

# ============ 原始数据路径 ============

# 叶绿素 (Chlorophyll-a) 小时数据
CHLA_HOURLY_DIR = Path("/data_new/renamed_data/chla_nc_after_rename/chla/R/")

# 海温 (sea surface temperature) 小时数据
SST_HOURLY_DIR = Path("/data_new/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3/")

# ============ 路径验证函数 ============

def check_paths():
    """检查所有数据路径是否存在"""
    paths = {
        "CHLA_HOURLY_DIR": CHLA_HOURLY_DIR,
        "SST_HOURLY_DIR": SST_HOURLY_DIR,
    }

    print("数据路径检查:")
    print("=" * 50)
    for name, path in paths.items():
        exists = path.exists()
        status = "OK" if exists else "NOT FOUND"
        print(f"  {name}: {path}")
        print(f"    状态: {status}")
    print("=" * 50)


if __name__ == "__main__":
    check_paths()
