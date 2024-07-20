import streamlit as st

# Tiêu đề của ứng dụng
st.title('Đây là test')

# URL hình ảnh mẫu
image_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"

# Hiển thị hình ảnh
st.image(image_url, caption='Hình ảnh từ URL', use_column_width=True)
