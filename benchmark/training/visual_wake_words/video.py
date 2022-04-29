import cv2
import os
import sys

def get_images(path_to_video, output_folder):
    cam = cv2.VideoCapture(path_to_video)
    try:
        if not os.path.exists(output_folder + '/test'):
            os.makedirs(output_folder + '/test')
    except OSError:
        print('Error: Creating directory of data')
  
    # frame
    currentframe = 0
  
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if ret:
            # if video is still left continue creating images
            name = './' + output_folder + '/test/' + str(currentframe) + '.jpg'
            # print ('Creating...' + name)
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
  
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

# get_images(sys.argv[1])