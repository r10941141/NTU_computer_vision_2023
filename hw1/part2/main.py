import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='C:\\Users\\chengan_huang\\Desktop\\computerview\\hw1_material\\part2\\testdata\\1.png', help='path to input image')
    parser.add_argument('--setting_path', default='C:\\Users\\chengan_huang\\Desktop\\computerview\\hw1_material\\part2\\testdata\\1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    


    high=0
    low=999999999
    ### TODO ###
    RGB=np.array(
    [[0.0,0.0,1.0],
    [0.0,1.0,0.0],
    [0.1,0.0,0.9],
    [0.1,0.4,0.5],
    [0.8,0.2,0.0]])
    sigma_s = 2
    sigma_r = 0.1
    
    i = RGB.shape[0]-1
    for k in range(RGB.shape[0]):
        if k<RGB.shape[0]:
            img_gray = RGB[k,0]*img_rgb[:,:,0] + RGB[k,1]*img_rgb[:,:,1] + RGB[k,2]*img_rgb[:,:,2]

        JBF = Joint_bilateral_filter(sigma_s, sigma_r)
        bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
        
        bf_out = cv2.cvtColor(bf_out,cv2.COLOR_RGB2BGR)
        jbf_out = cv2.cvtColor(jbf_out,cv2.COLOR_RGB2BGR)

        error = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
        print(error, np.dtype(error))
        if error>high:
            high=error
            jbf_out_highest=jbf_out
            img_gray_highest=img_gray
            
        if error<low:
            low = error
            jbf_out_lowest=jbf_out
            img_gray_lowest=img_gray
            print("now low is:",error)

    img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bf_out = JBF.joint_bilateral_filter(img, img).astype(np.uint8)
    jbf_out = JBF.joint_bilateral_filter(img, img_gray).astype(np.uint8)
    error = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
    print(error)

    cv2.imshow('origin', img)



    cv2.imshow('jbf_out_lowest', jbf_out_lowest)
    cv2.imwrite('C:\\Users\\chengan_huang\\Desktop\\computerview\\hw1_material\\part2\\testdata\\jbf_out_highest.png', jbf_out_highest)
    cv2.imwrite('C:\\Users\\chengan_huang\\Desktop\\computerview\\hw1_material\\part2\\testdata\\gray_highest.png', img_gray_highest)
    cv2.imwrite('C:\\Users\\chengan_huang\\Desktop\\computerview\\hw1_material\\part2\\testdata\\jbf_out_lowest.png', jbf_out_lowest)
    cv2.imwrite('C:\\Users\\chengan_huang\\Desktop\\computerview\\hw1_material\\part2\\testdata\\gray_lowest.png', img_gray_lowest)
    print("h&l:",high,low)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()