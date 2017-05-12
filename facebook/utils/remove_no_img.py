import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import os
import pdb

dir_path = "../../cnnclf/data/img/humanity"

def constant_img(img):
    try:
        n = img[0,0,0]
    except:
        pdb.set_trace()
        return False
    all_equal = np.all(img == n)
    #if all_equal:
    #    cv2.imshow("image", img)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    return all_equal


def main():
    male_no_img = cv2.imread("./male_no_img.jpg")
    female_no_img = cv2.imread("./female_no_img.jpg")
    files_in_dir = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    i = 0
    for pathname in files_in_dir:
        full_path = os.path.join(dir_path, pathname)
        img = cv2.imread(full_path)
        if np.array_equal(img, male_no_img) \
                or np.array_equal(img, female_no_img) \
                or constant_img(img):
            print("Removing file {}".format(full_path))
            i += 1
            os.remove(full_path)

    print("Removed {} images".format(i))

if __name__ == "__main__":
    main()
