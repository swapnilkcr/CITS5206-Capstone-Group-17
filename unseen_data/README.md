# Unseen client data (Zenodo NeWater Pump 1)

Source: [Zenodo 13808085](https://zenodo.org/records/13808085) — *Water Pump Dataset for a Water Supply System* (CC BY-SA 4.0).

This folder holds **non-SKAB** pump data for UI screening, retrain experiments, and capstone “unseen dataset” demos.

## Folder layout

```text
unseen_data/
  README.md                          ← this file
  zenodo_newater_pump1_raw/          ← drop Zenodo CSVs here (unchanged filenames)
  processed/
    newater_pump1_merged.csv         ← output of scripts/prepare_newater_pump1.py
    manifest.json                    ← column list, row counts, merge settings
```

## What to download for **NeWater Pump 1 “完整版”**

### A. 必下 — 直接挂在 Pump 1 上的传感器（10 个文件）

| File (exact Zenodo name) | Merged column (example) |
|--------------------------|-------------------------|
| `Current Sensor - NeWater Pump 1.csv` | `current` |
| `Power Sensor - NeWater Pump 1.csv` | `power` |
| `Energy Sensor - NeWater Pump 1.csv` | `energy` |
| `Vibration Sensor - NeWater Pump 1 Temperature.csv` | `vib_temperature` |
| `Vibration Sensor - NeWater Pump 1 X-Axis Speed.csv` | `vib_x_speed` |
| `Vibration Sensor - NeWater Pump 1 Y-Axis Speed.csv` | `vib_y_speed` |
| `Vibration Sensor - NeWater Pump 1 Z-Axis Speed.csv` | `vib_z_speed` |
| `Vibration Sensor - NeWater Pump 1 X-Axis Displacement.csv` | `vib_x_disp` |
| `Vibration Sensor - NeWater Pump 1 Y-Axis Displacement.csv` | `vib_y_disp` |
| `Vibration Sensor - NeWater Pump 1 Z-Axis Displacement.csv` | `vib_z_disp` |

### B. 强烈建议 — 与 Pump 1 配套的管路压力（命名是 Incoming **Pump 1**）

| File | Merged column |
|------|----------------|
| `Pressure Sensor - NeWater Incoming Pump 1.csv` | `pressure_incoming_p1` |

### C. 可选 — 系统级上下文（不是 Pump1 专属，但可并进宽表）

| File | Merged column | Note |
|------|----------------|------|
| `Pressure Sensor - NeWater Outgoing Pump.csv` | `pressure_outgoing` | 共用出水压力 |
| `Water Level Sensor - NeWater Tank.csv` | `tank_level` | 水箱液位，可差分近似流量趋势 |

### D. 文档（打标签用）

| File | Purpose |
|------|---------|
| `Water Pump Dataset Decription.md` | 3 类故障/事件说明；用于后续 `anomaly` 列 |

### 不要放进 Pump 1 包里的文件

- 所有 **NeWater Pump 2**、**Potable Pump 1/2** 的 Current/Power/Energy/Vibration
- `Pressure Sensor - NeWater Incoming Pump 2.csv`（属于 Pump 2）

---

## 合并后是什么？

- **一行 = 一个统一时间戳**（默认重采样 1 分钟）
- **一列 = 一个传感器**（保留真实列名，不冒充 SKAB）
- 在 UI 里作为 **unseen dataset**，用 8 列映射到 SKAB 通道做筛查；要可靠结果需在本数据上 **retrain**（见 `ui/frontend/docs.html#retrain-guide`）

## 命令

```bash
# 自动下载（Zenodo API，约 300–400 MB，需几分钟）
python scripts/download_newater_pump1.py

# 下载 + 合并一步完成
python scripts/download_newater_pump1.py --also-merge

# 若已手动放过 CSV，只合并：
python scripts/prepare_newater_pump1.py

# 3) 在 UI 上传
#    unseen_data/processed/newater_pump1_merged.csv
```

## SKAB 8 列映射参考（UI 选手动映射时可对照）

| SKAB | NeWater 建议选列 |
|------|------------------|
| Current | `current` |
| Pressure | `pressure_incoming_p1` |
| Temperature | `vib_temperature` |
| Voltage | `power` 或 `energy`（看 CSV 里是 V 还是 W） |
| Accelerometer1RMS | `vib_x_speed` 或脚本生成的 `vib_x_speed_rms` |
| Accelerometer2RMS | `vib_y_speed` 或 `vib_y_speed_rms` |
| Thermocouple | `vib_temperature`（第二路温度占位） |
| Volume Flow RateRMS | `tank_level` 差分 / 留空 |
