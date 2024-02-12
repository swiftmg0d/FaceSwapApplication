import os
import cv2
import dlib
import numpy
import threading
import sys
import time
from concurrent.futures import ThreadPoolExecutor

shape_predicotr_path = os.path.abspath(
    os.path.expanduser(os.path.expandvars("assets/shape_predictor_68_face_landmarks.dat")))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predicotr_path)

# 36 - 41 : right eye
# 42 - 47 : left eye
# 27 - 34 : nose
# 48 - 60 : mouth
# 22 - 26 : left brow
# 17 - 21 : right brow


whole_face_points = (
    [22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 17, 18, 19, 20, 21, 27, 28, 29, 30, 31, 32, 33,
     34,
     48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60])

# right eye, left eye, nose, right brow, left brow, motuh
whole_face_points_ordered = [
    [42, 43, 44, 45, 46, 47, 36, 37, 38, 39, 40, 41, 22, 23, 24, 25, 26, 17, 18, 19, 20, 21, 27, 28, 29, 30, 31, 32, 33,
     34, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
]


def read_image(file_name):
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (image.shape[1], image.shape[0]))
    return image


def get_landmarks(image):
    try:
        rectangle = detector(image, 1)
        if len(rectangle) > 1:
            raise Exception()
        if len(rectangle) == 0:
            raise Exception()
        return numpy.matrix(
            [[predictor_value.x, predictor_value.y] for predictor_value in predictor(image, rectangle[0]).parts()])
    except Exception as e:
        print(f"Error in get_landmarks: {e}")


def transformation_from_points(points_of_img1_landmarks, points_of_img2_landmarks):
    points_of_img1_landmarks = points_of_img1_landmarks.astype(numpy.float64)
    points_of_img2_landmarks = points_of_img2_landmarks.astype(numpy.float64)

    c1 = numpy.mean(points_of_img1_landmarks, axis=0)
    points_of_img1_landmarks -= c1
    c2 = numpy.mean(points_of_img2_landmarks, axis=0)
    points_of_img2_landmarks -= c2

    s1 = numpy.std(points_of_img1_landmarks)
    points_of_img1_landmarks /= s1
    s2 = numpy.std(points_of_img2_landmarks)
    points_of_img2_landmarks /= s2

    U, S, Vt = numpy.linalg.svd(points_of_img1_landmarks.T * points_of_img2_landmarks)

    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def get_face_mask(img, landmarks):
    img = numpy.zeros(img.shape[:2], dtype=numpy.float64)

    for group in whole_face_points_ordered:
        points = cv2.convexHull(landmarks[group])
        cv2.fillConvexPoly(img, points, 1)

    img = numpy.array([img, img, img]).transpose((1, 2, 0))

    img = (cv2.GaussianBlur(img, (11, 11), 0) > 0) * 1.0
    img = cv2.GaussianBlur(img, (11, 1), 0)

    return img


def warp_image(img, align, img_shape):
    new_img = numpy.zeros(img_shape, dtype=img.dtype)

    cv2.warpAffine(img,
                   align[:2],
                   (img_shape[1], img_shape[0]),
                   dst=new_img,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return new_img


def correct_colors_of_images(img1, img2, landmarks_of_image1):
    blur_amount = 0.7 * numpy.linalg.norm(
        numpy.mean(landmarks_of_image1[[42, 43, 44, 45, 46, 47]], axis=0) -
        numpy.mean(landmarks_of_image1[[36, 37, 38, 39, 40, 41]], axis=0))

    blur_amount = int(blur_amount)
    blur_amount = blur_amount + 1 if blur_amount % 2 == 0 else blur_amount

    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

    return (img2.astype(numpy.float64) * img1_blur.astype(numpy.float64) /
            img2_blur.astype(numpy.float64))


def check_if_exist():
    directory = "result"
    os.makedirs(directory, exist_ok=True)


def swap_images(img1, img2,count):
    start = time.time()

    img1 = read_image(img1)
    img2 = read_image(img2)

    img1_landmark = get_landmarks(img1)
    img2_landmark = get_landmarks(img2)

    mask_of_img1 = get_face_mask(img1, img1_landmark)
    mask_of_img2 = get_face_mask(img2, img2_landmark)

    align = transformation_from_points(img1_landmark[whole_face_points],
                                       img2_landmark[whole_face_points])

    warped_mask_of_img1 = warp_image(mask_of_img2, align, img1.shape)

    combined_mask_of_img1 = numpy.max([mask_of_img1, warped_mask_of_img1],
                                      axis=0)
    warped_mask_of_img2 = warp_image(img2, align, img1.shape)

    warped_corrected_img2 = correct_colors_of_images(img1, warped_mask_of_img2, img1_landmark)

    final_img = img1 * (1.0 - combined_mask_of_img1) + warped_corrected_img2 * combined_mask_of_img1

    check_if_exist()
    cv2.imwrite(f'result/swapped{count}.jpg', final_img)
    path = 'result/swapped'
    path += str(count)
    path += "./jpg"
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
def swap(number):
    start = time.time()
    for x in range(2, 2+number):
        path_1 = f'img/1.1 - Copy ({x}).jpg'
        path_2 = f'img/1.2 - Copy ({x}).jpg'
        swap_images(path_1, path_2, x)

    print(f"Finished in {time.time() - start} seconds")

if __name__ == '__main__':
    print("Insert number of pair images to swap:")
    print(swap(int(input())))




