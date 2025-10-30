import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import tifffile
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    return cv2, mo, np, plt, tifffile


@app.cell
def _(mo):
    browser = mo.ui.file_browser(initial_path='data')
    browser
    return (browser,)


@app.cell
def _(browser, cv2, mo, np, tifffile):
    import h5py
    from pathlib import Path
    import shutil
    from dataclasses import dataclass
    from typing import Tuple, Sequence
    import tempfile

    @dataclass
    class Category:
        name: str
        id: int
        aliases: Tuple[str]

        def __contains__(self, item: str):
            return item in [*self.aliases, self.name]

        def of(self, set: Sequence[str]):
            return any([item in self for item in set])

    MCHERRY = Category(name = 'mcherry', id=0, aliases=["mch", "pharynx"])
    MKATE = Category(name = 'mkate', id=1, aliases=["mk", "nervechord", "nervering"])
    COMBO = Category(name = 'combo', id = 2, aliases=[])

    # --- Core processing function (no changes) ---
    def process_and_save_yolo(imstack, n_imgs, category, output_dir, image_id_counter, global_mean, global_std):
        """
        Processes an image stack and saves images and labels directly in YOLO format.
        """
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        for idx in range(n_imgs):
            # 1. Image processing
            img_to_process = imstack[idx]
            if np.mean(img_to_process) > 1 * global_std + global_mean:
                print("Skipping over exposed image")
                continue
            # delete hot pixels
            img_to_process[832,1336] = 0
            img_to_process[637, 484] = 0
            img_to_process = img_to_process[200:850, 100:]
            img_to_process = (np.float32(img_to_process) - np.float32(img_to_process).mean(axis=0))
            percentile = np.percentile(img_to_process, 90, axis=None)
            img_to_process[img_to_process < percentile] = 0
            img_normalized = cv2.normalize(img_to_process, None, 255, 255, cv2.NORM_INF, dtype=cv2.CV_8U)
            img_blurred = cv2.medianBlur(img_normalized, 1)
            patch_size = 3
            threshed = cv2.adaptiveThreshold(
                src=img_blurred,
                maxValue=255, # Correct maxValue for uint8
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV,
                blockSize=patch_size,
                C=6 if category is MKATE else 5
            )
            closed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            opened_image = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            yolo_annotations = []
            height, width = img_normalized.shape
            total_area = np.prod(img_normalized.shape)
            contours = [contour for contour in contours if cv2.contourArea(contour) > total_area * 0.01 and cv2.contourArea(contour) < total_area * 0.3]

            for contour in contours:          
                # 2. Normalize contour for YOLO format
                normalized_contour = contour.reshape(-1, 2) / np.array([width, height])
                normalized_flat = normalized_contour.flatten().tolist()

                yolo_str = f"{category.id} " + " ".join(map(str, normalized_flat))
                yolo_annotations.append(yolo_str)

            # 3. Save the image and its corresponding .txt label file
            image_filename = f"{image_id_counter}.png"
            rgb_image = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(str(images_dir / image_filename), rgb_image)

            label_filename = f"{image_id_counter}.txt"
            with open(labels_dir / label_filename, 'w') as f:
                f.write("\n".join(yolo_annotations))

            image_id_counter += 1

        return image_id_counter

    # --- Main function to orchestrate the simplified workflow ---
    def generate_yolo_dataset(*args):

        # 1. Setup a single output directory structure
        output_dir = Path("./yolo_dataset/combo")
        if output_dir.exists():
            print(f"Existsing dataset found. Removing...")
            shutil.rmtree(str(output_dir))
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)

        # 2. Get the full list of files to process
        all_files = list(browser.value) # Replace with your file list
        image_id_counter = 0

        print(f"--- Processing all {len(all_files)} files into '{output_dir}' ---")

        # 3. Process every file and save to the single output directory
        for file_info in mo.status.progress_bar(all_files):
            prefix = file_info.path.stem.split('_')[0]
            if prefix in MKATE:
                category = MKATE
            elif prefix in MCHERRY:
                category = MCHERRY
            elif prefix in COMBO:
                category = COMBO
            else:
                raise ValueError(f'File has an unrecognized category: {file_info.path.stem}.')

            # Load data and process
            file_path = file_info.path
            if file_path.suffix == '.h5':
                print("Reading H5 File")
                with tempfile.TemporaryFile(suffix='.np') as tf:
                    with h5py.File(file_path, 'r') as f:
                        image_stack = f['images']
                        np.save(file=tf, arr=image_stack)
                        image_stack = np.memmap(tf, dtype=np.uint16)
                        print("Calculating Mean...")
                        global_mean = np.mean(image_stack)
                        print("Calculating STD...")
                        global_std = np.std(image_stack)
                        print
                        image_id_counter = process_and_save_yolo(image_stack, image_stack.shape[0], category, output_dir, image_id_counter, global_mean, global_std)

            elif file_path.suffix in ['.tif', '.tiff']:
                image_stack = tifffile.imread(file_path)
                global_mean = np.mean(image_stack)
                global_std = np.std(image_stack)
                image_id_counter = process_and_save_yolo(image_stack, len(image_stack), category, output_dir, image_id_counter, global_mean, global_std) 

        print(f"\n✅ Dataset generation complete!")
        print(f"All {image_id_counter} images and labels saved in '{output_dir}'")
        print("You can now split these files into train/test sets as needed.")

    # Link to your Marimo button, for example:
    mo.ui.button(label='Run', on_click=generate_yolo_dataset)
    return MKATE, h5py


@app.cell
def _(mo):
    class indexer():
        def __init__(self):
            self.idx = 0
        def inc(self):
            self.idx += 1
            return self
        def dec(self):
            if self.idx > 0:
                self.idx -= 1
            return self
    fidx = indexer()
    imidx = indexer()




    prev_f = mo.ui.button(label='Previous File', value=fidx, on_click=lambda value: value.dec())
    next_f = mo.ui.button(label='Next File', value=fidx, on_click=lambda value: value.inc())
    prev_i = mo.ui.button(label='Previous Image', value=imidx, on_click=lambda value: value.dec())
    next_i = mo.ui.button(label='Next Image', value=imidx, on_click=lambda value: value.inc())
    mo.hstack([prev_f, prev_i, next_i, next_f])
    return next_i, prev_f, prev_i


@app.cell
def _(
    MKATE,
    browser,
    cv2,
    h5py,
    mo,
    next_i,
    np,
    plt,
    prev_f,
    prev_i,
    tifffile,
):
    print(prev_f.value.idx)
    print(next_i.value.idx)

    _safe_file_idx = prev_f.value.idx % len(browser.value)
    file_path = browser.value[_safe_file_idx].path
    print(file_path)

    if file_path.suffix == '.h5':
        with h5py.File(file_path, 'r') as f:
            image_stack = f['images'] # This is now a lazy-loaded dataset object
            n_imgs = image_stack.shape[0] - 2
    elif file_path.suffix in ['.tif', '.tiff']:
        image_stack = tifffile.imread(file_path) # TIFF files are usually smaller, loading is ok
        n_imgs = len(image_stack) - 2


    _idx = prev_i.value.idx

    prefix = file_path.stem.split('_')[0]

    img_to_process = image_stack[_idx]
    # delete hot pixels
    img_to_process[832,1336] = 0
    img_to_process[637, 484] = 0
    img_to_process = img_to_process[200:850, 100:]
    print(np.mean(img_to_process))
    img_to_process = (np.float32(img_to_process) - np.float32(img_to_process).mean(axis=0))
    percentile = np.percentile(img_to_process, 90, axis=None)
    img_to_process[img_to_process < percentile] = 0
    img_normalized = cv2.normalize(img_to_process, None, 255, 255, cv2.NORM_INF, dtype=cv2.CV_8U)
    img_blurred = cv2.medianBlur(img_normalized, 1)
    patch_size = 3
    threshed = cv2.adaptiveThreshold(
        src=img_blurred,
        maxValue=255, # Correct maxValue for uint8
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=patch_size,
        C=6 if MKATE.of(prefix) else 5
    )
    closed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    opened_image = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binary_mask = np.where(opened_image, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_annotations = []
    height, width = img_normalized.shape
    total_area = np.prod(img_normalized.shape)
    contours = [contour for contour in contours if cv2.contourArea(contour) > total_area * 0.01 and cv2.contourArea(contour) < total_area * 0.3]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 1. Create a color version of the normalized image to draw contours on
    img_with_contours = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)

    # 2. Draw the contours on the color image
    #    -1 means draw all contours, (0, 255, 0) is green, 2 is the line thickness
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

    # 3. Create the figure and plot the images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    # Plot the original image for comparison
    ax1.imshow(img_normalized, cmap='gray')
    ax1.set_title("Normalized Image")
    ax1.axis('off')

    # Plot the image with the detected contours
    ax2.imshow(img_with_contours)
    ax2.set_title("Detected Contours")
    ax2.axis('off')

    ax3.imshow(binary_mask)
    ax3.set_title("Binary Mask Image")
    ax3.axis('off')

    mo.mpl.interactive(plt.gcf())
    return


if __name__ == "__main__":
    app.run()
