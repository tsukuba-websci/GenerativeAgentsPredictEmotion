import pandas as pd
import re
from pathlib import Path
import os

def create_html_with_images_and_memories(csv_path, image_folder_path, output_html_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path, delimiter='|')

    # Start writing the HTML content
    html_content = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '    <head>',
        '        <meta charset="UTF-8">',
        '        <title>Can Generative Agents Predict Emotion?</title>',
        '        <link rel="stylesheet" type="text/css" href="styles.css">',
        '    </head>',
        '    <body>',
        '        <div class="container">',
        '            <h1 style="margin-top: 50px;"><span class="highlighted-title">Can Generative Agents Predict Emotion?</span><br><br>Appendix</h1>',
        '            <hp>In this appendix we detail each of the 5-part scenes created from the EmotionBench situations, as well as the emotional response of the agents.</p>',
        '        </div>',
        '        <div id="imageContainer" class="container">'
    ]

    # Create a mapping from ID to image filenames
    id_to_image = {i: f"{row['Emotion']}-{row['Factor'].split('-')[-1]}-{i%5+1}.png" for i, row in df.iterrows()}

    # Iterate over each row in the DataFrame to add the images and memories
    for index, row in df.iterrows():
        # Get the corresponding image filename
        image_filename = id_to_image.get(index)
        image_src = f"{image_folder_path}/{image_filename}"
        filename_without_extension, _ = os.path.splitext(image_filename)
        filename_without_extension = filename_without_extension.rsplit('-', 1)
        filename_without_extension = ' '.join(filename_without_extension)
        html_content.append(f'            <h3>{filename_without_extension}</h3>')
        html_content.append(f'            <h4>{row["Situation"]}</h4>')
        html_content.append(f'            <div class="figure">')
        html_content.append(f'                <img src="{image_src}" alt="{row["Emotion"]}" style="width:50%;">')
        html_content.append(f'            </div>')
        # Split the memories into a list

        memories = row['Memories'].split("~")

        html_content.append('<ol>')
        for i, memory in enumerate(memories):
            html_content.append(f'<li>{memory}</li>')
        html_content.append('</ol>')
        html_content.append('<br>')

    # Close the container div
    html_content.append('        </div>')

    # Continue with the closing tags for the HTML content
    html_content.extend([
        '    </body>',
        '</html>'
    ])

    # Write the HTML content to the specified output file
    with open(output_html_path, 'w') as output_file:
        output_file.write('\n'.join(html_content))

# Usage example:
csv_file_path = 'situations.csv' # Replace with your actual CSV file path
image_folder = 'figs/panas_results' # Replace with your actual image folder path
output_html = 'appendix.html' # Replace with your actual output HTML file path

# Call the function
create_html_with_images_and_memories(csv_file_path, image_folder, output_html)
