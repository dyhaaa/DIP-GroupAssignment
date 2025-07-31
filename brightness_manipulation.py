''' Importing from library '''
import cv2
import numpy as np

''' Variables '''

# Video filepath (insert your file directory to your specific video)
vid_path = r"[directory path]"

# export filepath (insert the file directory to the folder you would like to save the altered video)
# do not remove "\brightened_up_video.mp4"
exp_path = r"[directory path]\brightened_up_video.mp4"

# values that alters the video's brightness and contrast
contrast = 1.2
brightness = 50

# values that determine what is considered as night and day
vid_sample_frames = 30
day_threshold = 100

''' Functions '''

# Detect if the video is taken during nighttime
# Parameters: video_path = filepath to video, 
# sample_frames = select few amount of frames needed to check if the video is nighttime or not,
# daytime_threshold = variable to constitute what is considered daytime
def is_nighttime(video_path, sample_frames, daytime_threshold) -> bool:
    
    # Opening video file for reading
    vid = cv2.VideoCapture(video_path)
    # adding up the sum of mean pixel intensity value in the frame
    brightness_sum = 0
    # counts the amount of valid frames after processing it
    frame_count = 0
    # the number that determines the next frame to be used as a sample frame
    # Getting the total frames in the video and floor divide the amount of sample frames needed
    next_frame_period = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) // sample_frames)
    
    #Loop through each sample frame
    while frame_count < sample_frames:
        # Skipping to a specific frame to be used as a sample frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_count * next_frame_period)
        # Extracting the frame and determining if the frame is valid
        valid_frame, frame = vid.read()
        if not valid_frame:
            break
        # Change the color frame into grayscale frame
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate the mean amount of pixel intensity value and adding it to the sum of all means
        brightness_sum += np.mean(grayscale_frame)
        # increment by 1 to confirm that the current frame has been processed
        frame_count += 1
    
    # close the videocapture
    vid.release()
    
    # if frame_count is 0, then all frames extracted were not valid, therefore the video has an issue
    # also prevents 0 division when we calculate the average brightness
    if frame_count == 0:
        raise ValueError("No frames were successfully read. Error can occur due to receiving a corrupt video file, incorrect file path or the video format/codec is unsupported.")
    
    # Getting the average brightness by dividing the total sum of average grayscale pixels and the amount of sample frames used
    avg_brightness = brightness_sum / frame_count
    
    # if average brightness is less than the threshold that determines it to be daytime, return true for nighttime
    # else return false since it is daytime
    return avg_brightness < daytime_threshold
    

# Adjust the brightness accordingly
# Parameter: frame = current frame of the video
def adjust_brightness(frame, contrast, brightness):
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

# Determines whether the video was taken during day or night
# (Brightens the video if it is taken during night and export the updated video)
# Parameters: export_path = the path to the folder you want to save the altered video in
# sample_amount = the amount of frames you would like to use as a sample for checking whether if it is day or night
# threshold = determines the threshold for what is considered daytime
def determine_video(vid_path, export_path, sample_amount, threshold, contrast_val, brightness_val):
    
    # Determining if the video is considered taken during nighttime
    is_night : bool = is_nighttime(vid_path, sample_amount, threshold)
    
    # If the video was not taken during nighttime
    if not is_night:
        print("The video was recorded at daytime.")
        return # return statement to stop the function from continuing
    
    # If the video was taken during nighttime
    print("The video was recorded at nighttime.\nAdjusting video brightness, please wait a moment...")
    
    # --- Generating new video and brightening it up ---
    
    # Opening the video file for accessing
    vid = cv2.VideoCapture(vid_path)
    # Getting the video's frames per second
    fps = vid.get(cv2.CAP_PROP_FPS)
    # Getting the video's width
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Getting the video's height
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Choosing the video's codec by inputting the four character code (in this case it's mp4)
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    # instancing cv2's videowriter used to merge all frames back into a video file
    vw = cv2.VideoWriter(export_path, video_codec, fps, (width,height))
    
    # Loop to go through the entire video
    while True:
        # extracting a frame from the video, as well as determining if the frame is a valid frame
        valid_frame, frame = vid.read()
        if not valid_frame:
            break
        # if the frame is valid, alter it by adjusting the brightness of that frame
        alt_frame = adjust_brightness(frame, contrast_val, brightness_val)
        # giving the frame to the videowriter which will stitch all the frame together to form a video
        vw.write(alt_frame)
    
    # Close the videocapture and videowriter
    vid.release()
    vw.release()
    
    print(f"New adjusted video is saved to filepath: {export_path}")

''' Start Program '''
determine_video(vid_path, exp_path, vid_sample_frames, day_threshold, contrast, brightness)