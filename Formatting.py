import pandas as pd
import os
import shutil
import glob

# Function to convert Excel (.xls) to CSV
def convert_excel_to_csv(excel_file_path, csv_file_path):
    # Read the Excel file using openpyxl
    df = pd.read_excel(excel_file_path, skiprows=4)
    df.to_csv(csv_file_path, index=False)

# Function to process CSV files and reformat them to .xls
def process_and_convert_csv_files(source_directory, destination_directory, formatted_csv_dir):
    for source_file_path in glob.glob(os.path.join(source_directory, '**', '*.csv'), recursive=True):
        source_filename = os.path.basename(source_file_path)
        oldbase, extension = os.path.splitext(source_filename)

        # Check if the file extension is .csv
        if extension == '.csv':
            # Create a new file name with .xls extension
            newname = oldbase + '.xls'
            normalized_path = os.path.normpath(source_file_path)
            path_parts = normalized_path.split(os.sep)

            # Find the part that contains 'SUB' and ends with 'on' or 'off'
            subdirectory_name = next(
                (part for part in path_parts if 'SUB' in part and (part.endswith('on') or part.endswith('off'))), None)

            # Destination directory with subdirectory name (if found)
            full_destination_dir = os.path.join(destination_directory, subdirectory_name) if subdirectory_name else destination_directory

            # Create the directory if it doesn't exist
            if not os.path.exists(full_destination_dir):
                os.makedirs(full_destination_dir)

            full_destination_file_path = os.path.join(full_destination_dir, newname)
            print(f"Copied and renamed: {source_file_path} to {full_destination_file_path}")
            shutil.copyfile(source_file_path, full_destination_file_path)

            # Now convert the copied XLS file to CSV
            csv_file_newname = oldbase + '.csv'
            full_csv_file_dir = os.path.join(formatted_csv_dir, subdirectory_name) if subdirectory_name else formatted_csv_dir

            if not os.path.exists(full_csv_file_dir):
                os.makedirs(full_csv_file_dir)

            csv_file_path = os.path.join(full_csv_file_dir, csv_file_newname)

            if not os.path.exists(csv_file_path):
                print(f"Converting to CSV: {full_destination_file_path} to {csv_file_path}")
                convert_excel_to_csv(full_destination_file_path, csv_file_path)

# Main directories for C3D and GaitCycles files
c3d_source_directory = './data/C3Dfiles/'
c3d_destination_directory = './data/C3DFormattedXLSfiles/'
c3d_formatted_csv_dir = './data/C3DFormattedCSVfiles/'

gaitcycles_source_directory = './data/GaitCyclesfiles/'
gaitcycles_destination_directory = './data/GaitCyclesFormattedXLSfiles/'
gaitcycles_formatted_csv_dir = './data/GaitCyclesFormattedCSVfiles/'

# Process C3D files
process_and_convert_csv_files(c3d_source_directory, c3d_destination_directory, c3d_formatted_csv_dir)

# Process GaitCycles files
process_and_convert_csv_files(gaitcycles_source_directory, gaitcycles_destination_directory, gaitcycles_formatted_csv_dir)

directories_to_delete = [
    c3d_source_directory,
    c3d_destination_directory,
    gaitcycles_source_directory,
    gaitcycles_destination_directory
]

for directory in directories_to_delete:
    if os.path.exists(directory):
        shutil.rmtree(directory)