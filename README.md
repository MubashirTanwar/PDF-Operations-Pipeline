## This repository contains Python Script to Preprocess and extract segments of Line for finetuning TrOCR on Marathi Printed Text

[Link To TrOCR NoteBook](https://github.com/MubashirTanwar/TrOCR-Marathi-Printed-Words)

 ### getLines.py extracts bounding boxes of all horizontal text sentances from the provided PDF
 - Usage
    ```bash
    python getLines.py pdfPath outputDirectory
    ```
    
 ### getSource.py extracts raw text from the ground truth PDF and save them in txt file
  - Usage
    ```bash
    python getSource.py pdfPath outputDirectory.txt
    ```

 ### getCSV.py merges line png with the corresponding extracted text

    
