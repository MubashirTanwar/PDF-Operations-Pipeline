import os
import pandas as pd
import sys

def reverse_lines_to_dataframe(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()[::-1]

    # Extract image names from lines
    image_names = [f'line_{idx}.png' for idx, line in enumerate(lines, start=1)]
    original_names = [line.strip() for line in lines]

    df = pd.DataFrame({'Image': image_names, 'Text': original_names})

    return df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <text_file_path> <output_csv_path>")
        sys.exit(1)

    text_file_path = sys.argv[1]
    output_csv_path = sys.argv[2]

    result_df = reverse_lines_to_dataframe(text_file_path)

    # Save the resulting DataFrame to a CSV file
    result_df.to_csv(output_csv_path, index=False)

    print(f"CSV file saved to {output_csv_path}")
