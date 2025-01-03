import fitz  # PyMuPDF

def pdf_to_images(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    
    for i in range(len(pdf_document)):
        page = pdf_document[i]
        pix = page.get_pixmap()
        pix.save(f"{output_folder}/page_{i + 1}.png")

# 使用示例
pdf_to_images(r'D:\document\MCM\week1\problem\第一周小测题目.pdf', r'D:\document\MCM\week1\problem')