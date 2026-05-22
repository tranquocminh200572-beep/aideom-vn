# 📘 Bài 12 – Hướng dẫn cài đặt & tích hợp AIDEOM-VN

## Cấu trúc thư mục Bài 12

```
b12/
├── m1_forecast.py      # Module M1: Dự báo GDP Cobb-Douglas
├── m2_readiness.py     # Module M2: TOPSIS + Entropy sẵn sàng số
├── m3_allocation.py    # Module M3: LP phân bổ ngân sách ngành-vùng
├── m4_labor.py         # Module M4: Mô phỏng lao động NetJob
├── m5_risk.py          # Module M5: Đa mục tiêu + Stochastic LP
├── b12_dashboard.py    # Module M6: Dashboard Streamlit tích hợp
├── b12_wrapper.py      # Wrapper tích hợp vào app.py cũ
└── README_B12.md       # File này
```

## Bước 1: Copy thư mục b12 vào dự án

```powershell
# Từ thư mục gốc AIDEOM-VN
xcopy /E /I b12 C:\Users\LEGION\aideom-vn\modules\b12
```

## Bước 2: Cài đặt thư viện bổ sung

```powershell
cd C:\Users\LEGION\aideom-vn
venv\Scripts\Activate.ps1
pip install plotly scipy numpy pandas streamlit
```

> Bài 12 không cần pymoo, pulp, cvxpy – dùng scipy.linprog thuần.

## Bước 3: Chạy dashboard độc lập (để test)

```powershell
cd C:\Users\LEGION\aideom-vn\modules\b12
python -m streamlit run b12_dashboard.py
```

## Bước 4: Tích hợp vào app.py chính

Thêm đoạn sau vào `app.py` (trong phần routing bài):

```python
elif selected == "Bài 12":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules', 'b12'))
    import b12_dashboard  # module tự render
```

**Hoặc dùng wrapper:**

```python
elif selected == "Bài 12":
    from modules.b12.b12_wrapper import render
    render()
```

## Bước 5: Chạy app.py tổng

```powershell
cd C:\Users\LEGION\aideom-vn
python -m streamlit run app.py
```

---

## 🏗️ Kiến trúc 6 Module AIDEOM-VN

| Module | File | Kỹ thuật | Đầu ra |
|--------|------|-----------|--------|
| M1 | m1_forecast.py | Cobb-Douglas, Growth Accounting | GDP 2026-2030, TFP |
| M2 | m2_readiness.py | TOPSIS, Entropy Weight | Xếp hạng 6 vùng |
| M3 | m3_allocation.py | LP (scipy.linprog) | Phân bổ ngân sách |
| M4 | m4_labor.py | LP (NetJob optimization) | Việc làm ròng |
| M5 | m5_risk.py | Pareto frontier, Stochastic LP | Rủi ro, VSS |
| M6 | b12_dashboard.py | Streamlit, Plotly | Dashboard tương tác |

## 🎭 5 Kịch bản chính sách

| Kịch bản | Mô tả | D_2030 | AI_2030 | H_2030 |
|----------|-------|--------|---------|--------|
| S1 – Truyền thống | Trọng vốn vật chất | 25% | 90K | 32% |
| S2 – Số hóa nhanh | Ưu tiên chuyển đổi số | 35% | 100K | 34% |
| S3 – AI dẫn dắt | Tập trung AI, bán dẫn | 30% | 120K | 33% |
| S4 – Bao trùm số | Ưu tiên nhân lực, vùng yếu | 30% | 95K | 38% |
| S5 – Tối ưu cân bằng | Nghiệm AIDEOM-VN | 30% | 100K | 35% |

## ⚠️ Lưu ý kỹ thuật

- **Lambda công bằng vùng**: Mặc định 0.55 (thay vì 0.70 trong đề bài) để bài toán LP khả thi.
  Có thể nới nếu ngân sách đủ lớn (≥ 80,000 tỷ).
- **VSS âm**: Bình thường trong mô hình đơn giản hóa tuyến tính. Xem giải thích trong dashboard.
- **TFP**: Được hiệu chỉnh theo dữ liệu lịch sử 2020-2025, trung bình dùng làm A₀ 2026.

## 📚 Tài liệu tham chiếu

- Nghị quyết 57-NQ/TW (2024)
- QĐ 749/QĐ-TTg – Chuyển đổi số quốc gia
- QĐ 127/QĐ-TTg – Chiến lược AI 2030
- QĐ 411/QĐ-TTg – Kinh tế số và xã hội số
- NSO/GSO Việt Nam 2026; World Bank Vietnam 2024; WIPO GII 2025
