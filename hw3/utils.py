import numpy as np


def solve_homography(u, v):
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('same size!')
        return None
    if N < 4:
        print('N')

    # TODO: 1.forming A
    ux = u[:,0].reshape((N,1))
    uy = u[:,1].reshape((N,1))
    vx = v[:,0].reshape((N,1))
    vy = v[:,1].reshape((N,1))
    
    matrix_a   = np.concatenate( (ux, uy, np.ones((N,1)), np.zeros((N,3)), -1*np.multiply(ux,vx), -1*np.multiply(uy,vx), -1*vx), axis=1 );
    matrix_b = np.concatenate( (np.zeros((N,3)), ux, uy, np.ones((N,1)), -1*np.multiply(ux, vy), -1*np.multiply(uy,vy), -1*vy), axis=1 );
    A     = np.concatenate( (matrix_a, matrix_b), axis=0 );

    # TODO: 2.solve H with A
    U, S, VT = np.linalg.svd(A)
    h = VT[-1,:]/VT[-1,-1]
    H = h.reshape(3, 3)
    return H



def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    xc, yc = np.meshgrid(np.arange(xmin, xmax, 1), np.arange(ymin, ymax, 1), sparse = False)
    xr = xc.reshape(( 1,(xmax-xmin)*(ymax-ymin) ))
    yr = yc.reshape(( 1,(xmax-xmin)*(ymax-ymin) ))
    r1 =  np.ones(( 1,(xmax-xmin)*(ymax-ymin) ))
    M = np.concatenate((xr, yr, r1), axis = 0)

    if direction == 'b':
        b1 = np.dot(H_inv,M)
        b1 = np.divide(b1, b1[-1,:]) 
        sy = np.round( b1[1,:].reshape((ymax-ymin, xmax-xmin)) ).astype(int)
        sx = np.round( b1[0,:].reshape((ymax-ymin, xmax-xmin)) ).astype(int)  
 

        mask1 = (0<sy)*(sy<h_src)
        mask2 = (0<sx)*(sx<w_src)
        mask   = mask1*mask2

        dst[yc[mask], xc[mask]] = src[sy[mask], sx[mask]]

        pass

    elif direction == 'f':
        b1 = np.dot(H,M)
        b1 = np.divide(b1, b1[-1,:])
        sy = np.round(b1[1,:].reshape(ymax-ymin,xmax-xmin)).astype(int)
        sx = np.round(b1[0,:].reshape(ymax-ymin,xmax-xmin)).astype(int)


        dst[np.clip(sy, 0, dst.shape[0]-1), np.clip(sx, 0, dst.shape[1]-1)] = src

        pass

    return dst
