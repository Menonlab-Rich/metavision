import os
import random
import argparse
import copy
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

def polygon_from_yolo(normalized_coords, img_width, img_height):
    """Converts a YOLO segmentation list of normalized coordinates to a Shapely Polygon."""
    if len(normalized_coords) < 6: # Need at least 3 points (6 values)
        return None

    # Denormalize coordinates
    denormalized_points = []
    for i in range(0, len(normalized_coords), 2):
        x = normalized_coords[i] * img_width
        y = normalized_coords[i+1] * img_height
        denormalized_points.append((x, y))

    if len(denormalized_points) < 3:
        return None

    return Polygon(denormalized_points)

def yolo_from_polygon(polygon, img_width, img_height):
    """Converts a Shapely Polygon with absolute coordinates to a normalized YOLO segmentation list."""
    if polygon.is_empty or not hasattr(polygon, 'exterior'):
        return []

    # Get exterior coordinates and normalize them
    normalized_coords = []
    for x, y in polygon.exterior.coords:
        norm_x = x / img_width
        norm_y = y / img_height
        normalized_coords.extend([norm_x, norm_y])

    # YOLO format does not repeat the last point, Shapely does. Remove it.
    if len(normalized_coords) > 2 and normalized_coords[0] == normalized_coords[-2] and normalized_coords[1] == normalized_coords[-1]:
         normalized_coords = normalized_coords[:-2]

    return normalized_coords

def create_synthetic_dataset(input_images_dir, input_labels_dir, output_dir, num_images, inplace=False):
    """
    Creates synthetic data for YOLO segmentation by overlaying annotations from one
    image onto another while preserving spatial coordinates.
    """
    # 1. Setup paths (Identical to original script)
    input_images_path = Path(input_images_dir)
    input_labels_path = Path(input_labels_dir)

    if inplace:
        output_images_path = input_images_path
        output_labels_path = input_labels_path
        print(f"⚠️ INPLACE mode enabled. Original directories will be modified.")
    else:
        output_images_path = Path(output_dir) / 'images'
        output_labels_path = Path(output_dir) / 'labels'
        output_images_path.mkdir(parents=True, exist_ok=True)
        output_labels_path.mkdir(parents=True, exist_ok=True)
        print(f"🚀 Creating synthetic dataset in: {output_dir}")

    # 2. Load and preprocess all existing YOLO data (Identical to original script)
    print("📄 Loading and parsing existing annotations...")
    annotations_by_image = {}

    image_files = list(input_images_path.glob('*.jpg')) + list(input_images_path.glob('*.png'))

    for img_path in tqdm(image_files, desc="Parsing Labels"):
        label_path = input_labels_path / (img_path.stem + '.txt')
        if label_path.exists():
            image_annotations = []
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]

                    ann_info = {'class_id': class_id, 'coords': coords, 'source_image': img_path}
                    image_annotations.append(ann_info)

            if image_annotations:
                annotations_by_image[img_path.name] = image_annotations

    if not annotations_by_image:
        print("❌ Error: No valid annotations found to use as sources.")
        return

    # 3. Main generation loop (MODIFIED LOGIC)
    image_counter = 1
    valid_image_keys = list(annotations_by_image.keys())

    for _ in tqdm(range(num_images), desc="Generating Images"):
        # Select a random base image and a different source image for overlay
        base_image_name = random.choice(valid_image_keys)
        source_overlay_name = random.choice(valid_image_keys)
        # Ensure they are not the same image
        for _ in range(2):
            while base_image_name == source_overlay_name:
                source_overlay_name = random.choice(valid_image_keys)

            base_image_path = input_images_path / base_image_name
            source_overlay_path = input_images_path / source_overlay_name

            base_image = Image.open(base_image_path).convert('RGBA')
            source_overlay_image = Image.open(source_overlay_path).convert('RGBA')
            W, H = base_image.size

            # Check if images are the same size, otherwise skip this pair
            if source_overlay_image.size != (W, H):
                print(f"⚠️ Skipping pair: {base_image_name} and {source_overlay_name} have different dimensions.")
                continue

            # Get original annotations for the base image
            base_anns_info = copy.deepcopy(annotations_by_image.get(base_image_name, []))

            # Get all annotations from the source overlay image
            source_anns_to_add = copy.deepcopy(annotations_by_image.get(source_overlay_name, []))

            # Convert original annotations to Shapely Polygons for occlusion handling
            all_polygons_for_new_image = []
            for ann_info in base_anns_info:
                poly = polygon_from_yolo(ann_info['coords'], W, H)
                if poly:
                    all_polygons_for_new_image.append({'class_id': ann_info['class_id'], 'polygon': poly})

            # --- CORE CHANGE: Paste all objects from the source image at their original locations ---
            for source_ann in source_anns_to_add:
                source_poly = polygon_from_yolo(source_ann['coords'], W, H)
                if not source_poly or source_poly.is_empty:
                    continue

                min_x, min_y, max_x, max_y = [int(v) for v in source_poly.bounds]
                w, h = max_x - min_x, max_y - min_y
                if w <= 0 or h <= 0: continue

                # Create mask for the object
                relative_polygon_tuples = [(p[0] - min_x, p[1] - min_y) for p in source_poly.exterior.coords]
                mask = Image.new('L', (w, h), 0)
                ImageDraw.Draw(mask).polygon(relative_polygon_tuples, outline=1, fill=1)

                # Crop the object using the mask from the SOURCE image
                source_img_np = np.array(source_overlay_image)
                cropped_np = source_img_np[min_y:max_y, min_x:max_x]
                if cropped_np.shape[0] != h or cropped_np.shape[1] != w: continue # Sanity check for bounds
                cropped_np[:, :, 3] = np.array(mask) * 255
                object_img = Image.fromarray(cropped_np)

                # **THE FIX**: Paste the object at its ORIGINAL coordinates, not random ones
                paste_x, paste_y = min_x, min_y
                base_image.paste(object_img, (paste_x, paste_y), object_img)

                # The pasted polygon is the same as the source polygon since coordinates are preserved
                all_polygons_for_new_image.append({'class_id': source_ann['class_id'], 'polygon': source_poly})

        # 4. Handle Occlusions (Identical to original script)
        if len(all_polygons_for_new_image) > 1:
            # Sort polygons by area descending - larger objects are less likely to be fully occluded
            all_polygons_for_new_image.sort(key=lambda x: x['polygon'].area, reverse=True)

            for i in range(len(all_polygons_for_new_image)):
                occluder = all_polygons_for_new_image[i]['polygon']
                if not occluder or not occluder.is_valid: continue

                for j in range(i + 1, len(all_polygons_for_new_image)):
                    occluded = all_polygons_for_new_image[j]['polygon']
                    if not occluded or not occluded.is_valid: continue

                    updated_occluded = occluded.difference(occluder)

                    if isinstance(updated_occluded, MultiPolygon):
                        # Keep the largest remaining part if it splits
                        if not updated_occluded.geoms:
                            updated_occluded = Polygon() # make it empty
                        else:
                            updated_occluded = max(updated_occluded.geoms, key=lambda p: p.area)

                    all_polygons_for_new_image[j]['polygon'] = updated_occluded

        # 5. Save new image and label file (Identical to original script)
        new_image_filename = f"synthetic_{image_counter}.png"
        new_label_filename = f"synthetic_{image_counter}.txt"
        base_image.convert('RGB').save(output_images_path / new_image_filename)

        with open(output_labels_path / new_label_filename, 'w') as f:
            for ann in all_polygons_for_new_image:
                poly = ann['polygon']
                if not poly.is_empty and poly.area > 10: # Filter out tiny artifacts
                    normalized_coords = yolo_from_polygon(poly, W, H)
                    if normalized_coords:
                        coord_str = ' '.join([f"{c:.6f}" for c in normalized_coords])
                        f.write(f"{ann['class_id']} {coord_str}\n")

        image_counter += 1

    print(f"\n✅ Synthetic dataset creation complete! {image_counter - 1} new images and labels created.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create synthetic data for YOLO segmentation datasets.")
    parser.add_argument('--input_images', type=str, required=True, help="Path to the directory containing source images.")
    parser.add_argument('--input_labels', type=str, required=True, help="Path to the directory containing source YOLO .txt labels.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save new images and labels.")
    parser.add_argument('--num_images', type=int, default=50, help="Number of new synthetic images to generate.")
    parser.add_argument('--inplace', action='store_true', help="If set, the original directories will be modified. Use with caution.")

    args = parser.parse_args()

    create_synthetic_dataset(
        input_images_dir=args.input_images,
        input_labels_dir=args.input_labels,
        output_dir=args.output_dir,
        num_images=args.num_images,
        inplace=args.inplace
    )
