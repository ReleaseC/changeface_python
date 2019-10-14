import cv2
import numpy as np
import dlib
import time
import sys
from color_transfer import color_transfer
import argparse
import os

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


# Face 1
def get_face(img):
    global predictor
    global detector

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    faces = detector(img_gray)  # 只有一个脸的照片

    if len(faces) == 0:
        print('face count error ', len(faces))
        return False

    landmarks = predictor(img_gray, faces[0])

    landmarks_points = []
    face_points1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,26,25,24,19,18,17]
#    for index,n in enumerate(face_points1):
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmarks_points.append((x, y))

    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(convexhull)
    (x1, y1, w1, h1) = rect
    img_face1 = img[y1: y1 + h1, x1: x1 + w1]  # 输入照片的人脸区域

    # Delaunay triangulation
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    print(len(triangles))
    triangles = np.array(triangles, dtype=np.int32)
    indexes_triangles = []
    cnt = 0
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
        cnt += 1

    return img_face1, x1, y1, w1, h1, landmarks_points, indexes_triangles


if len(sys.argv) != 3:
    print('usage: python ', sys.argv[0], ' source.jpg target.jpg')
    exit(0)

img = cv2.imread(sys.argv[1])
print('source ', img.shape)
#blur img
#os.system("python3 /home/siiva/桌面/face_database/blur.py "+img)

img2 = cv2.imread(sys.argv[2])
(oh, ow, depth) = img2.shape
print('target ', ow, oh)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# get the source face image
img_face1, x1, y1, w1, h1, lm_pt1, tri_1 = get_face(img)
img_face2, x2, y2, w2, h2, lm_pt2, tri_2 = get_face(img2)
#print(x1, y1, w1, h1)
#print(x2, y2, w2, h2)

cnt = 0

while True:
    img[y1: y1 + h1, x1: x1 + w1] = img_face1
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines_space_mask = np.zeros_like(img_gray)

    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_new_face = np.zeros_like(img2)

    # 如果
    points1 = np.array(lm_pt1, np.int32)
    points2 = np.array(lm_pt2, np.int32)
    if cv2.isContourConvex(points2):
        print('convex')
    else:
        print('concave')

    face_points = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,26,25,24,19,18,17]
    print('face point =', len(face_points))
    fp = []
    for i, n in enumerate(face_points):
        fp.append(lm_pt2[n])
    points3 = np.array(fp, np.int32)
    convexhull2 = cv2.convexHull(points2)
    img_face_out = img2[y2: y2 + h2, x2: x2 + w2]  # 替换视频内的人脸区域

    # Triangulation of both faces
    cnt = 0
    for tidx in tri_1:
        #print('cnt ', cnt)
        cnt += 1
        # Triangulation of the first face -----------------
        tr1_pt1 = lm_pt1[tidx[0]]
        tr1_pt2 = lm_pt1[tidx[1]]
        tr1_pt3 = lm_pt1[tidx[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        rect1 = cv2.boundingRect(triangle1)         # get the ROI rect
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)  # create a mask for it
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],  # create WHITE mask
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Triangulation of second face --------------------
        tr2_pt1 = lm_pt2[tidx[0]]
        tr2_pt2 = lm_pt2[tidx[1]]
        tr2_pt3 = lm_pt2[tidx[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        points2 = np.array([[tr2_pt1[0] - x-1, tr2_pt1[1] - y-1],
                            [tr2_pt2[0] - x-1, tr2_pt2[1] - y-1],
                            [tr2_pt3[0] - x-1, tr2_pt3[1] - y-1]], np.int32)
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        # 需要去除黑边，使用 CV2.INTER_NEARESTs
        warped_t = cv2.warpAffine(
            cropped_triangle, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        # cv2.warpAffine(cropped_triangle, M, (w, h),
        #               warped_t, cv2.INTER_NEAREST)
        warped_t = cv2.bitwise_and(warped_t, warped_t, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_face1 = img2_new_face[y: y + h, x: x + w]
        img2_face1_gray = cv2.cvtColor(img2_face1, cv2.COLOR_BGR2GRAY)
        _, mask_tri = cv2.threshold(
            img2_face1_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_t = cv2.bitwise_and(warped_t, warped_t, mask=mask_tri)

        img2_face1 = cv2.add(img2_face1, warped_t)
        img2_new_face[y: y + h, x: x + w] = img2_face1

        #cv2.imshow('t', img2_new_face)
        #cv2.waitKey(1)

    #cv2.waitKey(0)

    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    #img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_head_mask = cv2.fillPoly(img2_face_mask, [points3], 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    

    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)
    #cv2.imshow('head mask',img2_head_mask)
    #cv2.imshow('img2_head_noface',img2_head_noface)
    #cv2.imshow('img2_new_face',img2_new_face)
    #cv2.imshow('result', result)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamlessclone2 = cv2.colorChange(result, img2_head_mask, 1.0, 1.0, 1.0)
    # [, dst[, red_mul[, green_mul[, blue_mul]]]]
    seamlessclone = cv2.seamlessClone(
        result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    seamlessclone1 = cv2.seamlessClone(
        result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
#    seamlessclone1 = cv2.textureFlattening(
#        result, img2, img2_head_mask, 30, 45,3)


    cv2.imshow('mixed', seamlessclone)  # 合成视频
    cv2.imwrite('mixed.jpg',seamlessclone)
#    cv2.imshow('result', result)  # 合成视频
    key = cv2.waitKey(0) & 0xFF
    #if key == 27 or key == 'q':
    break

cap.release()
cv2.destroyAllWindows()
videoWriter1.release()
# videoWriter2.release()
