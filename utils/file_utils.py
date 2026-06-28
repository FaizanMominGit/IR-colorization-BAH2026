import os
import glob
import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None


def validate_extension(file_path, expected_extension=".TIF"):
    """Validates that a file has the expected extension (case-insensitive)."""
    return file_path.lower().endswith(expected_extension.lower())

def find_file(directory, pattern):
    """Finds a file in the directory matching the pattern. Returns the first match or None."""
    search_pattern = pattern
    if not search_pattern.startswith('*'):
        search_pattern = '*' + search_pattern
    if not search_pattern.endswith('*'):
        search_pattern = search_pattern + '*'

    files = glob.glob(os.path.join(directory, search_pattern))
    tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]

    if tif_files:
        return tif_files[0]
    return files[0] if files else None

def read_tif_image(file_path):
    """
    Robust reader for TIFF satellite images.
    Tries tifffile first, falling back to OpenCV or PIL.
    """
    if tifffile is not None:
        try:
            return tifffile.imread(file_path)
        except Exception:
            pass

    try:
        import cv2
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3:
                # Transpose OpenCV (H, W, C) to standard multi-spectral (C, H, W)
                img = np.moveaxis(img, -1, 0)
            return img
    except Exception:
        pass

    try:
        from PIL import Image
        img = Image.open(file_path)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = np.moveaxis(arr, -1, 0)
        return arr
    except Exception as e:
        raise RuntimeError(f"Could not read TIFF image {file_path} using available libraries: {e}")

def write_tif_image(file_path, data, photometric=None):
    """
    Robust writer for TIFF satellite images.
    Tries tifffile first, falling back to OpenCV or PIL.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if tifffile is not None:
        try:
            if photometric:
                tifffile.imwrite(file_path, data, photometric=photometric)
            else:
                tifffile.imwrite(file_path, data)
            return
        except Exception:
            pass


    try:
        import cv2
        out_data = data
        if out_data.ndim == 3:
            out_data = np.moveaxis(out_data, 0, -1)
        cv2.imwrite(file_path, out_data)
        return
    except Exception:
        pass

    try:
        from PIL import Image
        out_data = data
        if out_data.ndim == 3:
            out_data = np.moveaxis(out_data, 0, -1)
        img = Image.fromarray(out_data)
        img.save(file_path)
        return
    except Exception as e:
        raise RuntimeError(f"Could not write TIFF image {file_path}: {e}")
