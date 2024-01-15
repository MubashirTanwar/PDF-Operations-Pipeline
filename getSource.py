import pdfplumber
import codecs
import sys

def extract_text_with_font(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

            return text
    except Exception as e:
        print(f"Error: {e}")
        return None


if len(sys.argv) != 3:
    print("usage: python getSource.py pdfPath outputDirectory.txt")
    sys.exit(1)

pdf_path = sys.argv[1]
txt_output_path = sys.argv[2]

extracted_text = extract_text_with_font(pdf_path)

if extracted_text:
    print("Text extracted and saved successfully.")
else:
    print("Failed to extract text from the PDF.")

with codecs.open(txt_output_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(extracted_text)

