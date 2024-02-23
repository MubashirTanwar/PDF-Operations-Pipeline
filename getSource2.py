from pdf2image import convert_from_path
import pytesseract
from langdetect import detect

def extract_text_from_pdf(pdf_path, output_txt_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)

    # Initialize Tesseract OCR
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Open a text file for writing
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        # Process each image
        for i, image in enumerate(images):
            # Perform OCR on the image
            text = pytesseract.image_to_string(image, lang='mar')
            
            # Detect language to ensure it's Marathi text
            detected_lang = detect(text)
            
            if detected_lang == 'mr':  # Check if the detected language is Marathi
                # Write the text to the text file, line by line
                lines = text.split('\n')
                for line in lines:
                    if line.strip():  # Ignore empty lines
                        txt_file.write(line.strip() + '\n')

if __name__ == "__main__":
    # Replace 'input.pdf' and 'output.txt' with your actual file paths
    pdf_path = 'mar-gt-50.pdf'
    output_txt_path = 'output2.txt'

    extract_text_from_pdf(pdf_path, output_txt_path)
    print(f"Text extraction complete. Results saved to '{output_txt_path}'.")
