# Load imports
import os
import cv2

'''
function get_images() definition
Parameter, image_directory, is the directory 
holding the images
'''


def get_images(image_directory):
    face_imgs = []
    labels = []
    imgs_quality = []
    extensions = ('jpg', 'png', 'gif')

    '''
    Each subject has their own folder with their
    images. The following line lists the names
    of the subfolders within image_directory.
    '''
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)):  # only load directories
            subfolder_files = os.listdir(
                os.path.join(image_directory, subfolder)
            )
            for file in subfolder_files:
                if file.endswith(extensions):  # grab images only
                    # read the image using openCV
                    img = cv2.imread(
                        os.path.join(image_directory, subfolder, file)
                    )

                    # convert image from BGR to grayscale
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray_img_eqhist = cv2.equalizeHist(gray_img)

                    # use CLAHE to increase the contrast of image
                    clahe = cv2.createCLAHE(clipLimit=40)
                    gray_img_clahe = clahe.apply(gray_img_eqhist)

                    # add the image to a list face_imgs
                    face_imgs.append(gray_img_clahe)
                    # add the image's label to a list face_labels
                    labels.append(subfolder)
                    # add the image's dimension to a list of imgs_quality
                    imgs_quality.append(img.shape)

    print("All images are loaded")
    # return the images and their labels
    return face_imgs, labels, imgs_quality
