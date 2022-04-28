import numpy as np
import cv2
import os
import heapq

def load_images_from_folder(folder):
    """
    Load images from a folder
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), 0)
        if img is not None:
            images.append(img)
    return images



def draw_outline(ref, query, model):
    """
        Draw outline of reference image in the query image.
        This is just an example to show the steps involved.
        You can modify to suit your needs.
        Inputs:
            ref: reference image
            query: query image
            model: estimated transformation from query to reference image
    """
    h,w = ref.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,model)
    

    return dst
    



def draw_inliers(img1, img2, kp1, kp2, matches, matchesMask):
    """
        Draw inlier between images
        img1 / img2: reference/query  img
        kp1 / kp2: their keypoints
        matches : list of (good) matches after ratio test
        matchesMask: Inlier mask returned in cv2.findHomography() 
    """
    matchesMask = matchesMask.ravel().tolist()
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
    
    return img3



def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
    
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)


def createMatcher(method):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    return bf




"""------------------------------------------    Path   ------------------------------------------------------------"""

path_book_cover_Reference = '/Users/malujie/Computer-vision/assignment 2/clean/A2_smvs/book_covers/Reference'
path_book_cover_Query = '/Users/malujie/Computer-vision/assignment 2/clean/A2_smvs/book_covers/Query'

path_paintings_Reference = '/Users/malujie/Computer-vision/assignment 2/clean/A2_smvs/paintings/Reference'
path_painting_Query = '/Users/malujie/Computer-vision/assignment 2/clean/A2_smvs/paintings/Query'

path_landmarks_Reference = '/Users/malujie/Computer-vision/assignment 2/clean/A2_smvs/landmarks/Reference'
path_landmarks_Query = '/Users/malujie/Computer-vision/assignment 2/clean/A2_smvs/landmarks/Query'

path_book_cover_no_reference = '/Users/malujie/Computer-vision/assignment 2/clean/A2_smvs/book_covers/no reference set'
path_landmarks_no_reference = '/Users/malujie/Computer-vision/assignment 2/clean/A2_smvs/landmarks/no reference set'



"""------------------------------------------     Do matches   -------------------------------------------------------"""

def match_k(k, path_ref, path_Q, method):  
    """
        use top-k method

        k: top-k
        path: path of reference and query images
        method: feature detection method

        returns numbers of successful pairs of images
    """
    book_ref = load_images_from_folder(path_ref)
    book_Q = load_images_from_folder(path_Q)

    # list = []
    count = 0
    for y in range(0,100):
        list = []
        q_kp, q_des = detectAndDescribe(book_Q[y], method)
        for x in range(0, 100):
            kp, des = detectAndDescribe(book_ref[x], method)

            bf_sift = createMatcher(method)
            matches_sift = bf_sift.knnMatch(des, q_des, k=2) 

            good_sift = []
            good_sift_without_list = []
            for m,n in matches_sift:
                if m.distance < 0.8 * n.distance:
                    good_sift.append([m])
                    good_sift_without_list.append(m)
        
            if len(good_sift_without_list) > 4:
                src_pts = np.float32([kp[m.queryIdx].pt for m in good_sift_without_list]).reshape(-1,1,2)
                dst_pts = np.float32([q_kp[m.trainIdx].pt for m in good_sift_without_list]).reshape(-1,1,2)

                model, matchesMask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
                inliers_number = np.sum(matchesMask)
                list.append(inliers_number)
            else:
                list.append(0)

        value = heapq.nlargest(k, list)

        for i in value:
            index = list.index(i)

            if y == index:
                count += 1
                break

    return count


def match_k_resize(k, factor, path_ref, path_Q, method):
    """
        use top-k matches with resize

        k: top-k
        factor: resize factor
        path: path of reference and query images
        method: feature detection method

        returns numbers of successful pairs of images
    """


    book_ref = load_images_from_folder(path_ref)
    book_Q = load_images_from_folder(path_Q)

    # list = []
    count = 0
    for y in range(0,100):
        list = []
        book_Q[y] = cv2.resize(book_Q[y], (0,0), fx=factor, fy=factor)
        q_kp, q_des = detectAndDescribe(book_Q[y], method)
        for x in range(0, 100):
            kp, des = detectAndDescribe(book_ref[x], method)

            bf_sift = createMatcher(method)
            matches_sift = bf_sift.knnMatch(des, q_des, k=2) 

            good_sift = []
            good_sift_without_list = []
            for m,n in matches_sift:
                if m.distance < 0.8 * n.distance:
                    good_sift.append([m])
                    good_sift_without_list.append(m)
        
            if len(good_sift_without_list) > 4:
                src_pts = np.float32([kp[m.queryIdx].pt for m in good_sift_without_list]).reshape(-1,1,2)
                dst_pts = np.float32([q_kp[m.trainIdx].pt for m in good_sift_without_list]).reshape(-1,1,2)

                model, matchesMask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
                inliers_number = np.sum(matchesMask)
                list.append(inliers_number)
            else:
                list.append(0)

        value = heapq.nlargest(k, list)

        # print(value)
        for i in value:
            index = list.index(i)

            if y == index:
                count += 1
                break

    return count



"""------------------------------------------    Not in data set   --------------------------------------------------"""

def get_min_score(path_ref, path_Q, method):
    """

        path: path of reference and query images
        method: feature detection method

        returns minimum score
    """

    land_ref = load_images_from_folder(path_ref)
    land_Q = load_images_from_folder(path_Q)

    max_list = []
    for y in range(0,100):
        list = []
        
        q_kp, q_des = detectAndDescribe(land_Q[y], method)
        for x in range(0, 100):
            kp, des = detectAndDescribe(land_ref[x], method)

            bf_sift = createMatcher(method)
            matches_sift = bf_sift.knnMatch(des, q_des, k=2) 

            good_sift = []
            good_sift_without_list = []
            for m,n in matches_sift:
                if m.distance < 0.8 * n.distance:
                    good_sift.append([m])
                    good_sift_without_list.append(m)
        
            if len(good_sift_without_list) > 4:
                src_pts = np.float32([kp[m.queryIdx].pt for m in good_sift_without_list]).reshape(-1,1,2)
                dst_pts = np.float32([q_kp[m.trainIdx].pt for m in good_sift_without_list]).reshape(-1,1,2)

                model, matchesMask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
                inliers_number = np.sum(matchesMask)
                list.append(inliers_number)
            else:
                list.append(0)
    
        max_value = max(list)
        index = list.index(max_value)

        if index == y:  
            max_list.append(max_value)
            # print ("The number", y, "image is the best, it has best score :", max_value, "(points matches).")
            
    return min(max_list)


def not_in_data_set(k, path_ref, path_Q, num, method, thershold):
    """

        k: top-k
        path_ref: path of reference images
        path_Q: path of query images
        num: number of images in data set (no reference data set)
        method: feature detection method
        thershold: thershold of minimum score

        returns numbers of successful pairs of images
    """

    land_ref = load_images_from_folder(path_ref)
    land_Q = load_images_from_folder(path_Q)

    count = 0
    for y in range(0,num-1):
        list = []
        q_kp, q_des = detectAndDescribe(land_Q[y], method)
        for x in range(0, 100):
            kp, des = detectAndDescribe(land_ref[x], method)

            bf_sift = createMatcher(method)
            matches_sift = bf_sift.knnMatch(des, q_des, k=2) 

            good_sift = []
            good_sift_without_list = []
            for m,n in matches_sift:
                if m.distance < 0.8 * n.distance:
                    good_sift.append([m])
                    good_sift_without_list.append(m)
        
            if len(good_sift_without_list) > 4:
                src_pts = np.float32([kp[m.queryIdx].pt for m in good_sift_without_list]).reshape(-1,1,2)
                dst_pts = np.float32([q_kp[m.trainIdx].pt for m in good_sift_without_list]).reshape(-1,1,2)

                model, matchesMask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
                inliers_number = np.sum(matchesMask)
                list.append(inliers_number)
            else:
                list.append(0)

    max_value = heapq.nlargest(k, list)

    for i in max_value:
        if i < thershold:
            count += 1
    
    return count
