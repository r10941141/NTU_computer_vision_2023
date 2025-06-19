import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    SUM1 = 0
    orb = cv2.ORB_create()
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        #print(im1,im1.shape[1] )
        SUM1  += im1.shape[1]       
        print(SUM1)
        # TODO: 1.feature detection & matching

        FP1, mp1 = orb.detectAndCompute(im1, None)
        FP2, mp2 = orb.detectAndCompute(im2, None)
        matches = bfm.knnMatch(mp1, mp2, k=2)        
        u1 = []
        v1 = []

        for i,j in matches:
            if i.distance < 0.73 * j.distance:
                u1.append(FP1[i.queryIdx].pt)
                v1.append(FP2[i.trainIdx].pt)
        u1 = np.array(u1)
        v1 = np.array(v1)

        x = 5000
        TH = 4
        f1 = 0
        HNmax = np.eye(3)
        for i in range(0, x+1):
            id_u = np.zeros((4,2))
            id_v = np.zeros((4,2)) 
            for j in range(4):
                idx = random.randint(0, len(u1)-1)
                id_u[j] = u1[idx]
                id_v[j] = v1[idx]
                
            H = solve_homography(id_v, id_u)
            r1 = np.ones((1,len(u1)))
            M = np.concatenate( (np.transpose(v1), r1), axis=0)
            W = np.concatenate( (np.transpose(u1), r1), axis=0)             
            b1 = np.dot(H,M)
            b1 = np.divide(b1, b1[-1,:])
            
            err  = np.linalg.norm((b1-W)[:-1,:], ord=1, axis=0)
            f2 = sum(err<TH)

            if f2 > f1:
                f1 = f2
                HNmax = H

        last_best_H = last_best_H.dot(HNmax)
        output = warping(im2, dst, last_best_H, 0, im2.shape[0], SUM1, SUM1+im2.shape[1], direction='b') 

    return output

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)