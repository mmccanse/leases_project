import streamlit as st

def show_pdf(pdf_url):
    st.markdown(
        f"<iframe src='{pdf_url}' width='130%' height='600' style='border:none;'></iframe>",
        unsafe_allow_html=True
    )

def main():
    st.title('PDF Viewer for Document 1')
    pdf_url = "https://drive.google.com/file/d/1EF7JuGFy3ujS5iK-DGTXQ_5X_0u4GpTy/preview"  # Update this to your actual PDF URL
    show_pdf(pdf_url)

if __name__ == "__main__":
    main()

