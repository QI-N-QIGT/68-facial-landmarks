# reference: https://github.com/yinguobing/cnn-facial-landmark/blob/save_model/landmark_video.py
import numpy as np
import tensorflow as tf

import cv2
import face_detector as fd
import pts_tool as pt

IMG_HEIGHT = 128
IMG_WIDTH = 128
LOGITS_TENSOR_NAME = 'logits/BiasAdd:0'
INPUT_TENSOR_NAME = 'input_to_float:0'
TEST_IMAGE_DIR = '4.jpg'                    # 'dataset/1691766_1.jpg'
META_PATH = 'model/model.ckpt-10350.meta'
MODEL_PATH = 'model/model.ckpt-10350'

def detect_marks(image_np, sess, detection_graph):

    logits_tensor = detection_graph.get_tensor_by_name(LOGITS_TENSOR_NAME)

    predictions = sess.run(
        logits_tensor,
        feed_dict={INPUT_TENSOR_NAME: image_np})

    marks = np.array(predictions).flatten()
    marks = np.reshape(marks, (-1, 2))

    return marks


def extract_face(image):

    conf, raw_boxes = fd.get_facebox(image=image, threshold=0.9)

    for box in raw_boxes:

        diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
        offset_y = int(abs(diff_height_width / 2))
        box_moved = pt.move_box(box, [0, offset_y])

        facebox = pt.get_square_box(box_moved)

        if pt.box_in_image(facebox, image):
            return facebox

    return None


def main():

    saver = tf.train.import_meta_graph(META_PATH)  # 导入图
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        saver.restore(sess,MODEL_PATH)
        graph = tf.get_default_graph()

        while True:
            frame = cv2.imread(TEST_IMAGE_DIR)
            facebox = extract_face(frame)
            if facebox is None:
                continue
            else:
                face_img = frame[
                    facebox[1]: facebox[3],
                    facebox[0]: facebox[2]]

            face_img = cv2.resize(face_img, (IMG_HEIGHT, IMG_WIDTH))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = np.array([face_img])
            landmarks = detect_marks(face_img, sess, graph)

            origin_box_size = facebox[2] - facebox[0]
            for mark in landmarks:
                mark[0] = facebox[0] + mark[0] * origin_box_size
                mark[1] = facebox[1] + mark[1] * origin_box_size
                cv2.circle(frame, (int(mark[0]), int(
                    mark[1])), 1, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.imshow("Preview", frame)
            if cv2.waitKey(10) == 27:      # Esc键
                break
    sess.close()

if __name__ == '__main__':
    main()

