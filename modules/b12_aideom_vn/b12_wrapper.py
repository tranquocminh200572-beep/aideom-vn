"""
Bài 12 – Wrapper tích hợp vào app.py chính
Gọi b12_dashboard.main_content() từ app.py
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st

def render():
    """Render toàn bộ nội dung Bài 12 trong app.py."""
    # Import lazy để không block app chính
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "b12_dashboard",
            os.path.join(_HERE, "b12_dashboard.py")
        )
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        st.error(f"Lỗi tải module B12: {e}")


# Nếu chạy trực tiếp
if __name__ == "__main__":
    render()
