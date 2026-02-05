#!/usr/bin/env python3
"""
处理后数据路径配置

记录项目使用的处理后数据路径，便于统一管理和移植。
"""

from pathlib import Path

# ============ 处理后数据路径 ============

# 叶绿素 (Chlorophyll-a) 日融合数据 南海区域
CHLA_DAILY_FUSION_DIR = Path("/data_new/chla_data_imputation_data_260125/chla_data_pretraining/daily_fusion_target_modified/")

# SST 时间加权对齐数据 南海区域
SST_WEIGHTED_ALIGNED_DIR = Path("/data_new/chla_data_imputation_data_260125/SST/jaxa_weighted_aligned/")

# ============ 路径验证函数 ============

def check_paths():
    """检查所有数据路径是否存在"""
    paths = {
        "CHLA_DAILY_FUSION_DIR": CHLA_DAILY_FUSION_DIR,
        "SST_WEIGHTED_ALIGNED_DIR": SST_WEIGHTED_ALIGNED_DIR,
    }

    print("处理后数据路径检查:")
    print("=" * 60)
    for name, path in paths.items():
        exists = path.exists()
        status = "OK" if exists else "NOT FOUND"
        print(f"  {name}:")
        print(f"    路径: {path}")
        print(f"    状态: {status}")
    print("=" * 60)


if __name__ == "__main__":
    check_paths()
