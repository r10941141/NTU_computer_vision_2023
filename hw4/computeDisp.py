import numpy as np
import cv2.ximgproc as xip


def dist(D1, D2):

    #return np.sum( np.square(M1 - M2) )
    return np.sum(np.abs( D1 - D2 ))
 


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)


    cost_lr = np.zeros((max_disp+1, h, w), dtype=np.float32)
    cost_rl = np.zeros((max_disp+1, h, w), dtype=np.float32)

    for s in range(max_disp+1):
        for x in range(w): 
            xl = max(x-s, 0)
            xr = min(x+s, w-1)
            for y in range(h):
                cost_lr[s, y, x] = dist(Il[y, x], Ir[y, xl])
                cost_rl[s, y, x] = dist(Ir[y, x], Il[y, xr])
        cost_lr[s,] = xip.jointBilateralFilter(Il, cost_lr[s,], 35, 5, 5)
        cost_rl[s,] = xip.jointBilateralFilter(Ir, cost_rl[s,], 35, 5, 5)  


    w_disp_l = np.argmin(cost_lr, axis=0)
    w_disp_r = np.argmin(cost_rl, axis=0)


    for y in range(h):
        for x in range(w):
            if x-w_disp_l[y,x] >= 0 and w_disp_l[y,x] == w_disp_r[y,x-w_disp_l[y,x]]:
                continue
            else:
                w_disp_l[y,x] = -1

    for y in range(h):
        for x in range(w):
            if w_disp_l[y,x] == -1:
                l = 0
                r = 0

                while x-l>=0 and w_disp_l[y,x-l] == -1:
                    l+=1
                if x-l < 0:
                    hole_L = max_disp 
                else:
                    hole_L = w_disp_l[y,x-l]

                while x+r<=w-1 and w_disp_l[y,x+r] == -1:
                    r+=1
                if x+r > w-1:
                    hole_R = max_disp
                else:
                    hole_R = w_disp_l[y, x+r]


                w_disp_l[y,x] = min(hole_L, hole_R)
   
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), w_disp_l.astype(np.uint8), 18, 1)
    return labels.astype(np.uint8)