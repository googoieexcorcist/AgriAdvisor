import requests
import os
import gdown

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        gdown.download(url, filename, quiet=False)
        print("\nDownload complete!")
    except Exception as e:
        print(f"Error downloading with gdown: {str(e)}")
        print("Trying alternative download method...")
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as file:
                if total_size == 0:
                    file.write(response.content)
                else:
                    downloaded = 0
                    total_size = int(total_size)
                    for data in response.iter_content(chunk_size=4096):
                        downloaded += len(data)
                        file.write(data)
                        done = int(50 * downloaded / total_size)
                        print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')
            print("\nDownload complete!")
        except Exception as e:
            print(f"Error downloading model: {str(e)}")

# Model URL (using Google Drive link)
model_url = "https://drive.google.com/uc?id=1-0D73HC8X9vtxsfs7ckh4cQPA1kCNs7N"
model_filename = "disease_detection_model.h5"

if __name__ == "__main__":
    try:
        download_file(model_url, model_filename)
        print(f"Model saved as {model_filename}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}") 