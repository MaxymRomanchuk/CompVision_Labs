import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def match(desc_1, desc_2, ratio=0.85):
    
    match1 = []
    match2 = []
    distances = {}
    
    for i in range(desc_1.shape[0]):
        if np.std(desc_1[i,:])!=0:

            # Get L1 norm
            d = desc_2-desc_1[i,:]
            d = np.linalg.norm(d, ord=1, axis=1)

            # Sort indexes desc
            orders = np.argsort(d).tolist()

            # Check if pair is good enough
            if d[orders[0]]/d[orders[1]]<=ratio:

                # Add pair
                match1.append((i,orders[0]))
                distances[f'{i}-{orders[0]}'] = d[orders[0]]
    
    # Recalculate pairs for cross-check of matching
    for i in range(desc_2.shape[0]):
        if np.std(desc_2[i,:])!=0:

            d = desc_1-desc_2[i,:]
            d = np.linalg.norm(d, ord=1, axis=1)

            orders = np.argsort(d).tolist()

            if d[orders[0]]/d[orders[1]]<=ratio:
                match2.append((orders[0],i))
                distances[f'{orders[0]}-{i}'] = d[orders[0]]
    
    # Make pairs unique (exclude multiple connections of a point)
    match = list(set(match1).intersection(set(match2)))

    # Add distances
    return [(pair[0], pair[1], distances[f'{pair[0]}-{pair[1]}']) for pair in match]

def start(path_1, path_2, ratio):
    img_1 = cv.imread(path_1, cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread(path_2, cv.IMREAD_GRAYSCALE)
    
    # Get key points and descriptors
    surf = cv.xfeatures2d.SURF_create(10)
    kp_1, desc_1 = surf.detectAndCompute(img_1, None)
    kp_2, desc_2 = surf.detectAndCompute(img_2, None)

    # Custom matcher
    match_list = match(desc_1, desc_2, ratio)
    matches = sorted([ cv.DMatch(point[0], point[1], point[2]) for point in match_list], key= lambda x: x.distance)

    # Brute-force matcher
    bf_matcher = cv.BFMatcher(cv.NORM_L1)
    bf_mtc = sorted(bf_matcher.match(desc_1, desc_2), key= lambda x: x.distance)

    # FLANN matcher
    flann = cv.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    fl_mtc = sorted(flann.match(desc_1, desc_2), key= lambda x: x.distance)


    draw_params = dict( 
        matchColor = (0,255,0),
        singlePointColor = (255,0,0),
        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    match_img = cv.drawMatches(img_1, kp_1, img_2, kp_2, matches[:40], None, **draw_params)
    
    bf_match_img = cv.drawMatches(img_1, kp_1, img_2, kp_2, bf_mtc[:40], None, **draw_params)

    fl_match_img = cv.drawMatches(img_1, kp_1, img_2, kp_2, fl_mtc[:40], None, **draw_params)
    return match_img, bf_match_img, fl_match_img

paths = [
    ['hand_1.jpg', 'hand_2.jpg'],
    ['tools_1.jpg', 'tools_2.jpg']
]
for pair in paths:
    images = start(*pair, ratio=0.8)
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(images[0])
    axes[0].set_title('Custom function')
    axes[1].imshow(images[1])
    axes[1].set_title('OpenCV Brute-Force Matcher')
    axes[2].imshow(images[2])
    axes[2].set_title('OpenCV FLANN Matcher')
    plt.show()