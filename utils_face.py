import numpy as np
from skimage import transform
import logging


def fix_eyes(a, b, c, num_eyes):
    """
    a, b -> eye coordinates
    c -> nose coordinates
    """
    logger = logging.getLogger(__name__)
    nose = c
    if num_eyes == 1:
        # specified eye, if only 1 eye specified, leftEye = rightEye
        b = np.asarray([a[0], c[1]])  # specifications to form a right-angled triangle
        ac = a - c
        bc = b - c
        cosine_angle = np.dot(ac, bc) / (np.linalg.norm(ac) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))

        rotFactor = -1 if c[1] > a[1] else 1

        angleFactor = (angle - 30) / 90  # 30 degrees is considered the right angle for dogs/cats
        if a[0] < c[0]:  # left eye specified
            leye, reye = a, [c[0], a[1] - angleFactor * (a[1] - c[1]) * rotFactor]
        elif a[0] > c[0]:  # right eye specified
            leye, reye = [c[0], a[1] - angleFactor * (a[1] - c[1]) * rotFactor], a
    elif num_eyes == 2:
        ab = b - a
        ca = c - a
        inters = a + np.dot(ca, ab) / np.dot(ab, ab) * ab
        angle = np.degrees(np.arctan2(c[1] - inters[1], c[0] - inters[0]))
        angle = 360 + angle if angle < 0 else angle
        if angle >= 0 and angle < 90:
            if a[1] > b[1]:
                leye, reye = a, b
            else:
                leye, reye = b, a
        elif angle >= 90 and angle < 180:
            if a[0] < b[0]:
                leye, reye = a, b
            else:
                leye, reye = b, a
        elif angle >= 180 and angle < 270:
            if a[1] < b[1]:
                leye, reye = a, b
            else:
                leye, reye = b, a
        else:
            if a[0] > b[0]:
                leye, reye = a, b
            else:
                leye, reye = b, a
    else:
        logger.info('error more that two eyes')
        leye, reye, nose = a, b, c
    return leye, reye, nose


def solve_dog(coord, SIZE):
    logger = logging.getLogger(__name__)
    logger.debug("solving dog coordenates")
    A = np.array([
        [coord[0], -coord[1], 1, 0],
        [coord[1], coord[0], 0, 1],
        [coord[2], -coord[3], 1, 0],
        [coord[3], coord[2], 0, 1],
    ])

    h, w, c = SIZE

    a, b, c = coord[0:2], coord[2:4], coord[4:6]
    ba = b - a
    ca = c - a
    inters = a + np.dot(ca, ba) / np.dot(ba, ba) * ba
    d_ab = np.sqrt(np.sum(np.square(a - b)))
    d_cint = np.sqrt(np.sum(np.square(c - inters)))
    # If nose is too close zoom in
    if d_cint / d_ab < 0.5:
        # r = d / norm_ab
        y = 1 / 2 - 1 / 12
        x1 = 1 / 2 - 1 / 5
        x2 = 1 / 2 + 1 / 5
    else:
        y = 1 / 2 - 1 / 12
        x1 = 1 / 2 - 1 / 6
        x2 = 1 / 2 + 1 / 6
    b = [x1 * h, y * w, x2 * h, y * w]
    # b = [1*h/3,1*w/3,2*h/3,1*w/3]
    # b = [0.7*h/2.4,0.7*w/2.4,1.7*h/2.4,0.7*w/2.4]
    sol = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))
    logger.debug("solved dog coordenates")
    return np.array([[sol[0], -sol[1], sol[2]], [sol[1], sol[0], sol[3]], [0, 0, 1]])


def solve_cat(coord, SIZE):
    logger = logging.getLogger(__name__)
    logger.debug("solving cat coordenates")
    A = np.array([
        [coord[0], -coord[1], 1, 0],
        [coord[1], coord[0], 0, 1],
        [coord[2], -coord[3], 1, 0],
        [coord[3], coord[2], 0, 1],
    ])

    h, w, c = SIZE

    ab = np.array([coord[0] - coord[2], coord[1] - coord[3]])
    ac = np.array([coord[0] - coord[4], coord[1] - coord[5]])
    norm_ab = np.linalg.norm(ab)
    d = ac.dot(ab) / norm_ab
    # If the head is too turned then we zoom out
    if d < 0 or d > norm_ab:
        # r = d / norm_ab
        y = 1 / 2 + 1 / 12
        x1 = 1 / 2 - 1 / 8
        x2 = 1 / 2 + 1 / 8
    else:
        y = 1 / 2 + 1 / 12
        x1 = 1 / 2 - 1 / 6
        x2 = 1 / 2 + 1 / 6
    b = [x1 * h, y * w, x2 * h, y * w]
    sol = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))
    logger.debug("solved dog coordenates")
    return np.array([[sol[0], -sol[1], sol[2]], [sol[1], sol[0], sol[3]], [0, 0, 1]])


def solve_comb(coord, box_type, SIZE):
    if box_type == 'cat':
        return solve_cat(coord, SIZE)
    else:
        return solve_dog(coord, SIZE)


def get_face_aligned(img_np, leye, reye, nose, pet_type, SIZE=(224, 224, 3)):
    logger = logging.getLogger(__name__)
    coord = np.array([leye, reye, nose]).reshape(6, )
    M = solve_comb(coord, pet_type, SIZE)
    logger.debug("coordinates solved for face alignment")
    transformation = transform.warp(img_np, np.linalg.inv(M))
    h, w, c = SIZE
    transformation = transformation[:h, :w]
    if transformation.shape != SIZE:
        transformation = transform.resize(transformation, SIZE)
    logger.debug("got transformation & scaled")
    return (transformation * 255).astype('uint8')
