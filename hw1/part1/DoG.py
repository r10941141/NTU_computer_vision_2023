
import numpy as np
import cv2
#kernel>>ksize sigma>>sigmaX
class Difference_of_Gaussian(object):             
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):

      keypoints=None

      for x in range(self.num_octaves):
        if x != 0:
          gray_image = cv2.resize(X_img[self.num_DoG_images_per_octave-1], (image.shape[1]//(2**(x)), image.shape[0]//(2**(x))),interpolation = cv2.INTER_NEAREST)

          X_img = np.zeros((self.num_DoG_images_per_octave, np.shape(image)[0]//(2**(x)), np.shape(image)[1]//(2**(x))), dtype=np.float64)
          Y_img = np.zeros((self.num_DoG_images_per_octave, np.shape(image)[0]//(2**(x)), np.shape(image)[1]//(2**(x))), dtype=np.float64)
        else:
          gray_image = image

          X_img = np.zeros((self.num_DoG_images_per_octave, np.shape(image)[0], np.shape(image)[1]), dtype=np.float64)
          Y_img = np.zeros((self.num_DoG_images_per_octave, np.shape(image)[0], np.shape(image)[1]), dtype=np.float64)
        
        for k in range(self.num_DoG_images_per_octave):
          gaussian_images=cv2.GaussianBlur(gray_image , ksize = (0, 0), sigmaX = self.sigma**(k+1))
          
          X_img[k]=gaussian_images
 
          if k==0:
            dog_images=(cv2.subtract(gray_image ,X_img[k] ))
          else :
            dog_images=(cv2.subtract(X_img[k-1] ,X_img[k]))
          

          Y_img[k]=dog_images

          #Y_img[k]=dog1_normalized

          #dog1_normalized = cv2.normalize(dog_images, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
          #cv2.imwrite('C:\\Users\\chengan_huang\\Desktop\\computerview\\hw1_material\\part1\\testdata\\DoG'+str(x)+str(k+1)+'.png', dog1_normalized)

        for i in range(1, len(Y_img)-1):
          for j in range(1, Y_img[i].shape[0]-1):
            for k in range(1, Y_img[i].shape[1]-1):
              neighbors = [Y_img[i-1:i+2, j-1:j+2, k-1:k+2]]


              if np.absolute(Y_img[i,j,k])>=(self.threshold) and (Y_img[i,j,k]==np.max(neighbors) or Y_img[i,j,k]==np.min(neighbors)):
                #point=[j, np.shape(image)[1] - k]
                point = np.array([j*(2**(x)),  k*(2**(x))])

                if keypoints is not None:
                  keypoints = np.vstack((keypoints, point))
                  
                else:
                  keypoints = point


      keypoints = np.unique(keypoints, axis=0)


      keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
      return keypoints
