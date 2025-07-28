import cv2
import numpy as np
import matplotlib.pyplot as plt

image_folder = [
    'Converted Paper (8)/001.png',
    'Converted Paper (8)/002.png',
    'Converted Paper (8)/003.png',
    'Converted Paper (8)/004.png',
    'Converted Paper (8)/005.png',
    'Converted Paper (8)/006.png',
    'Converted Paper (8)/007.png',
    'Converted Paper (8)/008.png',
]

# Load grayscale image and convert to binary (black and white) according to threshold (brightness 200 or less = 0, more than 200 = 255)
def to_binary(image_path):
    
    # Read image in grayscale (0)
    img = cv2.imread(image_path, 0)
    # cv2.threshold returns two outputs, [0] is the threshold and [1] is the thresholded image, so used [1]
    binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
    
    return binary

# Vertical histogram (sum of black pixels in column) for column segmentation
def vertical_histogram(binary):
    return np.sum(binary == 0, axis=0)

# Horizontal histogram (sum of black pixels in row) for line segmentation
def horizontal_histogram(binary):
    return np.sum(binary == 0, axis=1)

# Function to find regions (columns or paragraphs)
def find_regions(hist, min_width=1):
    
    regions = []
    
    # in_region to check if currently in a region
    in_region = False
    
    for i in range(len(hist)):
        # If wasnt in region and then now in region, means reached start of a region
        if hist[i] > 0 and not in_region:
            start = i
            in_region = True
            
        # If was in region and then now not in region, means reached end of a region
        elif hist[i] == 0 and in_region:
            end = i
            in_region = False
            
            # If width of region is more than min_width, add to list of regions
            if end - start > min_width:
                regions.append((start, end))
            
    return regions

# Detect and extract column regions using vertical histogram to get boundaries of columns
def segment_columns(binary):
    
    # Vertical histogram to detect columns
    v_hist = vertical_histogram(binary)
    
    # Get column boundaries based on vertical histogram
    column_bounds = find_regions(v_hist, min_width=10)
    
    columns = []
    
    # Extract column images and append to list of columns
    for x_start, x_end in column_bounds:
        
        # Extract column image from binary image
        column_img = binary[:, x_start:x_end]
        
        # Append to list of columns
        columns.append((x_start, x_end, column_img))
        
    return columns

# Detect and extract paragraphs inside a column image, based on horizontal histogram
def segment_paragraphs(column_img, x_start, line_spacing_threshold=40):
    
    # Horizontal histogram to detect lines
    h_hist = horizontal_histogram(column_img)
    
    # Get line boundaries based on horizontal histogram
    line_bounds = find_regions(h_hist, min_width=5)

    paragraphs = []

    # Start 1st paragraph with the 1st line start and end
    current_para_start = line_bounds[0][0]
    current_para_end = line_bounds[0][1]

    # For loop through all lines and group into paragraphs based on line_spacing_threshold
    for i in range(1, len(line_bounds)):
        prev_end = current_para_end
        curr_start, curr_end = line_bounds[i]

        # If current line is close to the previous line, consider as part of the same paragraph
        if curr_start - prev_end < line_spacing_threshold:
            # Extend current paragraph
            current_para_end = curr_end  
        
        # If current line is far from the previous line (exceeds line_spacing_threshold), end paragraph
        else:
            # Save current paragraph
            para_img = column_img[current_para_start:current_para_end, :]
            paragraphs.append({
                "img": para_img,
                "x": x_start,
                "y": current_para_start
            })
            
            # Start new paragraph
            current_para_start = curr_start
            current_para_end = curr_end

    # Append last paragraph
    para_img = column_img[current_para_start:current_para_end, :]
    paragraphs.append({
        "img": para_img,
        "x": x_start,
        "y": current_para_start
    })

    return paragraphs


# Extract all paragraphs from image
def extract_paragraphs(image_path):
    
    # Get binary image
    binary = to_binary(image_path)
    
    # Segment into column images
    columns = segment_columns(binary)
    
    all_paragraphs = []
    
    # For each column image, extract paragraphs
    for x_start, x_end, column_img in columns:
        paragraphs = segment_paragraphs(column_img, x_start)
        all_paragraphs.extend(paragraphs)

    return all_paragraphs

# If using normal extract_paragraphs(), it will extract table as 1 paragraph and all other paragraphs as 1 paragraph
# So need to use this adjusted function
def extract_paragraphs_004(image_path):
    
    # Get binary image
    binary = to_binary(image_path)
    
    # Extract table as paragraph first
    
    # Table location
    table_y_start = 200
    table_y_end = 400
    
    # Get table image
    table_img = binary[table_y_start:table_y_end]
    
    all_paragraphs = []
    
    # Add table image to list of paragraphs
    all_paragraphs.append({
        "img": table_img,
        "x": 140,
        "y": table_y_start
    })
    
    # Debug
    #plt.imshow(binary, cmap='gray') 
    #plt.title("binary image") 
    #plt.show()
    
    # Make copy of binary image and replace table section with white
    # 'binary2 = binary' doesnt make copy, just reference, so need to use copy()
    binary2 = binary.copy()
    
    # Replace table with white (make it empty)
    binary2[table_y_start:table_y_end, :] = 255
    
    # Continue column and paragraph extraction
    columns = segment_columns(binary2)
    
    for x_start, x_end, column_img in columns:
        paragraphs = segment_paragraphs(column_img, x_start)
        all_paragraphs.extend(paragraphs)
        
    return all_paragraphs


# Display paragraphs
def show_paragraphs(paragraphs, image_title=""):

    plt.figure(figsize=(16, 8))
    
    for i, para in enumerate(paragraphs):
        plt.subplot(2, 4, i + 1)
        plt.imshow(para["img"], cmap='gray')
        plt.title(f"x={para['x']}, y={para['y']}")
        plt.axis('off')
        
    plt.suptitle(image_title)
    plt.tight_layout()
    plt.show()

# Main
def main():

    # For loop to extract paragraphs for each image in image_folder
    for path in image_folder:
        
        # Use extract_paragraphSs_004(path) for image 004, else use normal extract_paragraphs(path)
        if '004.png' in path:
            paragraphs = extract_paragraphs_004(path)
            
        else:
            paragraphs = extract_paragraphs(path)
            
        print(f"\n{path} \n{len(paragraphs)} paragraphs")
        
        show_paragraphs(paragraphs, image_title=path)
        
    # No. of paragraphs should be 6 8 7 8 5 8 8 8 (including tables/images, was counted manually)


if __name__ == '__main__':
    main()
    