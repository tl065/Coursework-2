import cv2
import numpy as np
import logging

def read_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    return img1, img2

def detect_and_describe(image):
    descriptor = cv2.SIFT_create()
    kps, features = descriptor.detectAndCompute(image, None)
    return (kps, features)

def match_keypoints(featuresA, featuresB, k=2, ratio=0.75):
    matches = []
    featuresA = np.array(featuresA)
    featuresB = np.array(featuresB)

    for i in range(len(featuresA)):
        distances = np.linalg.norm(featuresB - featuresA[i], axis=1)
        indices = np.argsort(distances)[:k]
        d1, d2 = distances[indices[0]], distances[indices[1]]
        if d1 < ratio * d2:
            matches.append((i, indices[0]))

    return matches

def compute_homography(kpsA, kpsB, matches):
    if len(matches) > 4:
        ptsA = np.float32([kpsA[m[0]].pt for m in matches])  # m[0] for index in featuresA
        ptsB = np.float32([kpsB[m[1]].pt for m in matches])  # m[1] for index in featuresB
        H, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)
        return H

    return None

def warp_and_stitch(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)

    [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:translation_dist[1] + h1, translation_dist[0]:translation_dist[0] + w1] = img1

    return output_img

def remove_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    height, width = gray.shape

    # Initialize variables to store the boundaries of non-black content
    top = 0
    bottom = height - 1
    left = 0
    right = width - 1

    # Find top
    for i in range(height):
        if np.max(gray[i]) > 0:
            top = i
            break

    # Find bottom
    for i in range(height - 1, -1, -1):
        if np.max(gray[i]) > 0:
            bottom = i
            break

    # Find left
    for i in range(width):
        if np.max(gray[:, i]) > 0:
            left = i
            break

    # Find right
    for i in range(width - 1, -1, -1):
        if np.max(gray[:, i]) > 0:
            right = i
            break

    # Crop the image
    cropped = image[top:bottom+1, left:right+1]

    return cropped

def main():
    img1_path = 'p1.jpg'
    img2_path = 'p2.jpg'
    img1, img2 = read_images(img1_path, img2_path)

    kpsA, featuresA = detect_and_describe(img1)
    kpsB, featuresB = detect_and_describe(img2)
    matches = match_keypoints(featuresA, featuresB)

    H = compute_homography(kpsA, kpsB, matches)

    if H is not None:
        stitched_img = warp_and_stitch(img1, img2, H)
        result = remove_black_borders(stitched_img)
        # stitched_img[0:img1.shape[0], 0:img1.shape[1]] = img1

        cv2.imwrite('Resulting image using p.jpg', result)
    else:
        print("Not enough matches to compute homography.")

if __name__ == '__main__':
    main()
