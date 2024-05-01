import streamlit as st

def show_pdf(pdf_url):
    st.markdown(
        f"<iframe src='{pdf_url}' width='100%' height='600' style='border:none;'></iframe>",
        unsafe_allow_html=True
    )

def main():
    st.title('PDF Viewer for Document 1')
    pdf_url = "https://raw.githubusercontent.com/mmccanse/leases_project/main/pages/1_EY.pdf"  # Update this to your actual PDF URL
    show_pdf(pdf_url)

if __name__ == "__main__":
    main()
