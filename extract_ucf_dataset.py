import zipfile
import os

downloads_dir = r"C:\Users\Vikas V\Downloads"
target_dir = r"d:\survillence camera\data\UCF_Crimes"
os.makedirs(target_dir, exist_ok=True)

candidates = ["archive.zip", "archive (1).zip", "archive (2).zip", "UCF_Crimes.zip", "UCF_Crimes-Train-Test-Split.zip"]

found = False
for cand in candidates:
    zpath = os.path.join(downloads_dir, cand)
    if os.path.exists(zpath):
        print(f"Checking {zpath} ...")
        try:
            with zipfile.ZipFile(zpath, 'r') as z:
                names = z.namelist()
                # Check for UCF-Crime Video Dataset
                has_anomaly_dir = any("Anomaly_Videos" in n for n in names[:2000])
                has_mp4 = any(n.endswith(".mp4") for n in names[:2000])
                
                if has_anomaly_dir or has_mp4:
                    print(f"--> Found UCF_Crime dataset in {zpath}!")
                    print(f"--> Extracting to {target_dir} ... This might take several minutes.")
                    z.extractall(target_dir)
                    print("--> Extraction completed successfully!")
                    found = True
                    break
                else:
                    print("  Not the UCF dataset (or not the video version), skipping.")
        except Exception as e:
            print(f"  Error reading {cand}: {e}")

if not found:
    print("Could not find the UCF Crime dataset zip file in the Downloads folder.")
