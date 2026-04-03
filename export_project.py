import os
import zipfile

def export_project(source_dir, output_filename):
    print(f"Creating shareable copy: {output_filename}")
    
    # Files and directories to ignore so the zip doesn't become gigabytes in size
    ignore_dirs = {'.git', 'venv', 'venv_gpu', '__pycache__', '.pytest_cache', 'cache', 'logs', 'logs_old'}
    ignore_exts = {'.pt', '.pth', '.npy', '.mp4', '.avi', '.png', '.jpg'}
    ignore_specific = {'rebuild_ucf_videos.py', 'extract_ucf_dataset.py', 'ucf crime data'}

    # Track how many files we bundled
    count = 0

    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            
            # Prune directories we don't want to enter
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.endswith('cache') and not 'data' in root]

            # Let's skip the big 'data' directory entirely if it's there
            if os.path.basename(root) == 'data' or 'ucf crime data' in root:
                dirs[:] = []
                continue

            for file in files:
                # Exclude specific files and massive extensions
                if any(file.endswith(ext) for ext in ignore_exts):
                    continue
                if file in ignore_specific:
                    continue
                
                # Exclude hidden files or system files
                if file.startswith('.'):
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_dir)
                
                zipf.write(full_path, arcname=rel_path)
                count += 1

    print(f"Done! Successfully packed {count} code files into {output_filename}")
    print("You can safely send this ZIP file to anyone. They will need to run `pip install -r requirements.txt` to run it.")

if __name__ == '__main__':
    # Execute export for the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(current_dir, 'surveillance_camera_app.zip')
    export_project(current_dir, zip_path)
