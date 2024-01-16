import os
import pandas as pd

def reverse_lines_to_dataframe(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()[::-1]

    # Extract image names from lines
    image_names = [f'line_{idx}.png' for idx, line in enumerate(lines, start=1)]
    original_names = [line.strip() for line in lines]


    df = pd.DataFrame({'Image': image_names, 'Text': original_names})

    return df


# Example usage
text_file_path = r'C:\NIC AI Training\PrePro\extracted_text.txt'
image_folder_path = r'C:\NIC AI Training\PrePro\output_path\out'


result_df = reverse_lines_to_dataframe(text_file_path)

# Display the resulting DataFrame
print(result_df)


result_df.to_csv('page.csv', index=False)

