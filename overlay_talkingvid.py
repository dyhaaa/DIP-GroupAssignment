#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri July 29 13:52:00 2025

@author: kaijing
"""

import cv2


main_video_path    = "/Users/kaijing/Downloads/digital_pics/street.mp4"
overlay_video_path = "/Users/kaijing/Downloads/digital_pics/talking.mp4"


main_vid    = cv2.VideoCapture(main_video_path)
overlay_vid = cv2.VideoCapture(overlay_video_path)


overlay_width  = 320   
overlay_height = 240   




while main_vid.isOpened():
    ret_main,  frame_main   = main_vid.read()
    ret_ovl,   frame_overlay = overlay_vid.read()

   
    if not ret_main:
        break

    # if talking.mp4 vid finishes before main the reloop it
    if not ret_ovl:
        overlay_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_ovl, frame_overlay = overlay_vid.read()
        if not ret_ovl:
            break

    # resizing
    small_overlay = cv2.resize(frame_overlay, (overlay_width, overlay_height))

    # place overlay vid position
    x_offset, y_offset = 10, 10
    x1, y1 = x_offset, y_offset
    x2, y2 = x1 + overlay_width, y1 + overlay_height

    #paste overlay vid onto main frame
    frame_main[y1:y2, x1:x2] = small_overlay

    
    cv2.imshow("Main with Overlay", frame_main)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

main_vid.release()
overlay_vid.release()
cv2.destroyAllWindows()

