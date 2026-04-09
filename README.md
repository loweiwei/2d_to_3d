# 2D to 3D Toolkit

本專案提供一個簡單的方法，將 **2D 影像 + 深度資訊（Depth）** 轉換為 **3D 點雲（Point Cloud）**，並可進一步進行簡單的 3D 視覺化與分析。

---

## 📌 專案功能

- 將 RGB 影像與 Depth map 轉換成 3D 點雲
- 支援單視角 3D 重建
- 提供基本點雲處理（過濾、Bounding Box）
- 可視化 3D 點雲結果（使用 Open3D）

---

## 📂 專案結構

```
2d_to_3d/
├── examples/           # 範例程式
├── tdt3d/              # 核心程式碼
├── data/               # 測試資料（如 ScanNet）
├── requirements.txt    # 套件需求
└── README.md
```

---

## ⚙️ 環境安裝

建議使用 Python 3.9 ~ 3.11

### 1️⃣ 建立環境（可選）
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
```

### 2️⃣ 安裝套件
```bash
pip install -r requirements.txt
```

---

## 📦 requirements.txt

```
numpy
opencv-python
pytest
open3d
```

---

## 🚀 使用方式

### ▶️ 單視角 3D 重建

```bash
python examples/demo_single_view.py
```

執行後你可以：
1. 在畫面中選擇 ROI（滑鼠框選）
2. 按下 Enter
3. 產生對應的 3D 點雲

---

### ▶️ ScanNet 測試

```bash
python examples/run_scannet_minimal.py
```

---

## 🧠 流程說明（簡單版）

```
RGB Image + Depth Map
        ↓
相機內外參（Intrinsic / Extrinsic）
        ↓
2D → 3D 投影
        ↓
產生 Point Cloud
        ↓
過濾 / Bounding Box
        ↓
Open3D 視覺化
```

---

## 📤 Input / Output

### Input
- RGB 影像（.jpg / .png）
- Depth map（.png / uint16）
- 相機參數（intrinsic / extrinsic）

### Output
- 3D 點雲（Point Cloud）
- Bounding Box（可選）

---

## ⚠️ 常見問題

### ❌ 空點雲錯誤
```
ValueError: Cannot calculate AABB from empty point cloud
```

👉 可能原因：
- ROI 選錯（沒有選到物體）
- Depth 全為 0
- 相機參數錯誤

---

## 🔧 未來可擴充

- 加入 SAM（Segment Anything）自動選物件
- 多視角融合（Multi-view 3D Reconstruction）
- 與 VLM 結合（做 3D Visual Grounding）

---

## 👩‍💻 作者

GitHub: https://github.com/loweiwei

---

## 📌 備註

本專案適合用於：
- 3D Vision 入門
- 視覺轉 3D 應用練習
- 作業 / 專題展示