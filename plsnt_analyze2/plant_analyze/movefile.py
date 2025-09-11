import os
import shutil

def main():
    source_directory = r"D:\smartfarm\picture_topview"
    destination_directory = r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_result_topview_smartfarm"

    if not os.path.exists(source_directory):
        print(f"Source directory not found: {source_directory}")
        return

    if not os.path.exists(destination_directory):
        try:
            os.makedirs(destination_directory)
            print(f"Created destination directory: {destination_directory}")
        except Exception as e:
            print(f"Cannot create destination directory: {e}")
            return

    files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]
    
    if not files:
        print(f"No files found in source directory: {source_directory}")
        return
    
    for file_name in files:
        try:
            source_path = os.path.join(source_directory, file_name)
            destination_path = os.path.join(destination_directory, file_name)
            shutil.move(source_path, destination_path)
            print(f"Successfully moved file {file_name} to {destination_directory}")
        except Exception as e:
            print(f"Error moving file {file_name}: {e}")

if __name__ == "__main__":
   main()