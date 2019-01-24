import cv2
import pts_tool as pt
import face_detector as fd
import changedBox as cb


PICTURE_DIR = 'dataset/1691766_1.jpg'
DATA_DIR = 'dataset/1691766_1.pts'

img = cv2.imread(PICTURE_DIR)
# conf, facebox = fd.get_facebox(img, threshold=0.5)
# new_conf, new_facebox = cb.change_box(img, DATA_DIR, 40)

#fd.draw_result(img, conf, facebox)
# fd.draw_result(img, new_conf, new_facebox)
points = pt.read_points(DATA_DIR)
pt.draw_landmark_point(img, points)
cv2.imshow('image',img)
cv2.waitKey (0)


