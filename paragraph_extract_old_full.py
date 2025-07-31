import cv2
import numpy as np
import os

# ---------------------- CONFIG ---------------------- #
# list of image files to process
image_folder = [
    'Converted Paper (8)/001.png', 
    'Converted Paper (8)/002.png', 
    'Converted Paper (8)/003.png', 
    'Converted Paper (8)/004.png', 
    'Converted Paper (8)/005.png', 
    'Converted Paper (8)/006.png', 
    'Converted Paper (8)/007.png', 
    'Converted Paper (8)/008.png'
]
save_paragraphs = True
output_dir = 'output'

# threshold values for paragraph detection
LINE_HEIGHT_PIX   = 40
MIN_PARA_AREA     = 3000
MIN_WIDTH_FRAC    = 0.25
MAX_FILL_RATIO    = 0.80
MIN_TEXT_LINES    = 2
ROW_TEXT_THRESH   = 0.05
MAX_ASPECT_RATIO  = 6.0
MAX_HLINES        = 2

# create output folder if saving is enabled
if save_paragraphs:
    os.makedirs(output_dir, exist_ok=True)


# count number of black pixels in each column for histogram projection
def get_sum_of_black_pixels_in_column(binary_image):
    return np.sum(binary_image == 0, axis=0)

# find column start/end positions based on pixel density
def find_column_regions(hist, threshold=1, min_width=30):
    is_text = hist > threshold
    in_col = False
    column_regions = []

    for i in range(len(is_text)):
        if is_text[i] and not in_col:
            start = i
            in_col = True
        elif not is_text[i] and in_col:
            end = i
            in_col = False
            if (end - start) >= min_width:
                column_regions.append((start, end))
    if in_col:
        column_regions.append((start, len(hist)-1))
    return column_regions if column_regions else [(0, len(hist)-1)]

# full column detection based on projection
def detect_columns(binary_image):
    hist = get_sum_of_black_pixels_in_column(binary_image)
    return find_column_regions(hist)


# count how many horizontal text rows in a paragraph area
def count_text_rows(roi_inv):
    row_sum = np.sum(roi_inv == 255, axis=1)
    active = row_sum > ROW_TEXT_THRESH * roi_inv.shape[1]
    runs, in_run = 0, False
    for flag in active:
        if flag and not in_run:
            runs += 1
            in_run = True
        elif not flag and in_run:
            in_run = False
    return runs

# detect how many solid horizontal lines (like table lines)
def count_horizontal_lines(mask, min_len_ratio=0.8):
    h, w = mask.shape
    lines = 0
    for row in mask:
        if np.sum(row == 255) > min_len_ratio * w:
            lines += 1
    return lines


# extract paragraph regions from image and return them
def extract_paragraphs(img_path):
    # load grayscale image
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)

    # binarize (text becomes white)
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_bounds = detect_columns(255 - bin_inv)

    paragraph_boxes = []

    # process each detected column
    for (c_start, c_end) in col_bounds:
        col_w = c_end - c_start
        col_inv = bin_inv[:, c_start:c_end]

        # merge nearby text using vertical dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, LINE_HEIGHT_PIX))
        merged = cv2.dilate(col_inv, kernel, iterations=1)

        # connected component analysis to find blobs
        n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
        for i in range(1, n_lbl):
            x, y, w, h, area = stats[i]
            if area < MIN_PARA_AREA or w < MIN_WIDTH_FRAC * col_w:
                continue  # skip small or narrow blobs

            roi = col_inv[y:y+h, x:x+w]
            fill = cv2.countNonZero(roi) / (w * h)
            tl = count_text_rows(roi)
            ar = w / h
            hlines = count_horizontal_lines(roi)

            # final filter to check if it's a valid paragraph
            keep = (
                fill <= MAX_FILL_RATIO and
                tl >= MIN_TEXT_LINES and
                ar <= MAX_ASPECT_RATIO and
                hlines <= MAX_HLINES
            )

            if keep:
                abs_box = (c_start + x, y, w, h)
                paragraph_boxes.append(abs_box)

    # crop the paragraphs from original grayscale image
    para_imgs = [gray[y:y+h, x:x+w] for (x,y,w,h) in paragraph_boxes]
    return len(col_bounds), para_imgs


def main():
    for img in image_folder:
        cols, paras = extract_paragraphs(img)
        print(f"{os.path.basename(img)} → {cols} col, {len(paras)} para")
        if save_paragraphs:
            base = os.path.splitext(os.path.basename(img))[0]
            for i, p in enumerate(paras):
                cv2.imwrite(os.path.join(output_dir, f"{base}_para_{i:02d}.png"), p)

if __name__ == "__main__":
    main()
