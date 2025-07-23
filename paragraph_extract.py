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
    'Converted Paper (8)/008.png'
]
show_plots = True  # For debug, set to True to show histograms


# Get sum of black pixels (binary_image == 0) in each column
def get_sum_of_black_pixels_in_column(binary_image):
    return np.sum(binary_image == 0, axis=0)


# Draw plots for debug
def draw_plot(binary_image, title=""):
    black_pixels = get_sum_of_black_pixels_in_column(binary_image)
    
    plt.figure()
    plt.plot(black_pixels)
    
    plt.title(f"{title}")
    plt.xlabel("Column Index")
    plt.ylabel("Number of Text Pixels")
    
    plt.show()

# Find column count from histogram
def find_column_regions(hist, threshold=1):

    # Make array of True/False, only columns with more than 1 black pixel = True, meaning they have text
    is_text = hist > threshold
    
    # in_column to check if currently in paragraph column
    in_column = False
    
    # Empty list that will store elements of "column here" just to be used to count number of columns
    column_regions = []

    for i in range(len(is_text)):
        # if seeing text and wasn't inside column, means new text column starts
        if is_text[i] and not in_column:
            start = i
            in_column = True
            
        # if not seeing text and was inside column, means column ended
        elif not is_text[i] and in_column:
            end = i
            in_column = False
            
            # If at least width of 10, add as column to column_regions list
            if (end - start) > 10:  
                column_regions.append("column here")


    return len(column_regions)



# Column detection
def detect_column_count(image_path):
    img = cv2.imread(image_path, 0)
    binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1] # [0] is threshold value and [1] is the thresholded image, so used [1]
    
    # Get image name by spltting by "/" and getting last element
    image_name = image_path.split("/")[-1]
    
    # If show_plots == True then draw plots for debug
    if show_plots:
        draw_plot(binary, title=image_name)
        
    black_pixels = get_sum_of_black_pixels_in_column(binary)
    num_columns = find_column_regions(black_pixels)
    
    return num_columns

# Main
def main():
    
    # Dictionary for Image -> No. of columns
    column_counts = {}
    
    # For loop to count columns for each image 
    for image_path in image_folder:
        try:
            count = detect_column_count(image_path)
            column_counts[image_path] = count
            image_name = image_path.split("/")[-1]
            print(f"{image_name} → {count} column(s)")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Print order by column count
    print("\nSorted Images by Number of Columns:")
    for image_path, cols in sorted(column_counts.items(), key=lambda x: x[1]):
        image_name = image_path.split("/")[-1]
        print(f"{image_name}: {cols} column(s)")
        
if __name__ == '__main__':
    main()