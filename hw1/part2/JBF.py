import numpy as np
import cv2


import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)

        output = np.zeros(img.shape)   

        padded_img = padded_img.astype('float64')

        padded_guidance = padded_guidance.astype('float64')
        padded_guidance = padded_guidance/255

        gs = np.zeros((self.wndw_size , self.wndw_size))
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                gs[i][j] = np.exp (((i-self.pad_w)**2+(j-self.pad_w)**2)/(-2*(self.sigma_s**2)) )



        for i in range(self.pad_w , np.shape(padded_guidance)[0]-self.pad_w):
            for j in range(self.pad_w , np.shape(padded_guidance)[1]-self.pad_w):

                ci = ((padded_guidance[i,j]-padded_guidance[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1])**2 /(-2*self.sigma_r**2))

                if len(ci.shape)==3:
                    ci = ci.sum(axis=2) 

                gr=np.exp(ci)
                g =np.multiply(gs, gr)            
                w =g.sum()

                ip=padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]   

                for k in range(np.shape(img)[2]):
                    output[i-self.pad_w][j-self.pad_w][k] =  np.multiply(g,ip[:,:,k]).sum()/w
       

        return np.clip(output, 0, 255).astype(np.uint8)