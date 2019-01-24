import face_detector as fd
import cv2
import pts_tool as pt

PICTURE_DIR = 'dataset/image_0630.png'
DATA_DIR = 'dataset/image_0630.pts'
img = cv2.imread(PICTURE_DIR)

def change_box(img, DATA_DIR,pointThr=40):
    '''
    实现框的转换，使其只剩下有特征点标记的框以及改变其大小，使其能包含所有的框
    :param img: 图片
    :pointthr:超过多少个点在框里判定这个框为有效框
    :data_dir:数据路径
    :return: 新的框的坐标
    '''
    confidences, faceboxes = fd.get_facebox(img, 0.5)
    points = pt.read_points(DATA_DIR)
    newFacebox = []
    m = 0
    for i in faceboxes:
        for j in points:
            if((i[0]<j[0]<i[2]) and (i[1]<j[1]<i[3])):
                m = m+1
        if m>pointThr:
            newFacebox.append(i)
            m=0
    # print(faceboxes)
    # print(newFacebox)
    # print("$$$$$$$$$")
    newConf = [confidences[faceboxes.index(newFacebox[0])]]
    left_x, left_y, right_x, right_y = newFacebox[0]
    #将边界框的位置总体向下移动，移动的大小为边界框的高度与宽度之差的一半
    height = right_y - left_y
    width = right_x - left_x
    left_y = left_y + (height-width)/2
    right_y = right_y + (height-width)/2
    #使得其变为正方形
    if height!=width:
        left_x = left_x - (height - width)/2
        right_x = right_x + (height - width)/2
    newFacebox = [[int(left_x), int(left_y), int(right_x), int(right_y)]]
    # print(newConf,newFacebox)
    return newConf, newFacebox

# def amendBox(img, newFacebox, DATA_DIR):
#     shape = img.shape
#     x_max = shape[0]
#     y_max =  shape[1]
#     points = pt.read_points(DATA_DIR)
#     for i in points:
#         if(i[0])







