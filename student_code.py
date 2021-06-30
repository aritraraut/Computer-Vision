import numpy as np
import cv2 # You must not use cv2.cornerHarris()
# You must not add any other library


### If you need additional helper methods, add those. 
### Write details description of those

"""
  Returns the harris corners,  image derivative in X direction,  and 
  image derivative in Y direction.
  Args
  - image: numpy nd-array of dim (m, n, c)
  - window_size: The shaps of the windows for harris corner is (window_size, window size)
  - alpha: used in calculating corner response function R
  - threshold: For accepting any point as a corner, the R value must be 
   greater then threshold * maximum R value. 
  - nms_size = non maximum suppression window size is (nms_size, nms_size) 
    around the corner
  Returns 
  - corners: the list of detected corners
  - Ix: image derivative in X direction
  - Iy: image derivative in Y direction

"""
def harris_corners(image, window_size=5, alpha=0.04, threshold=1e-2, nms_size=10):

    ### YOUR CODE HERE
    image = cv2.GaussianBlur(image,(5,5),1)
    
    image_x = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    image_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    
    image_xx = cv2.GaussianBlur(image_x * image_x,(window_size,window_size),8)
    image_yy = cv2.GaussianBlur(image_y * image_y,(window_size,window_size),8)
    image_xy = cv2.GaussianBlur(image_x * image_y,(window_size,window_size),8)
    
    det_M = image_xx * image_yy - image_xy**2
    trac_M = image_xx + image_yy
    R = det_M - alpha*(trac_M)**2
    
    R[R<threshold*np.max(R)]=0
    
    
    for i in range(R.shape[0]-nms_size):
        for j in range(R.shape[1]-nms_size):
            nms_window = R[i:i+nms_size,j:j+nms_size]
            nms_window[nms_window!=np.max(nms_window)]=0
    
    corners = R
    Ix = image_x
    Iy = image_y

    return corners, Ix, Iy

"""
  Creates key points form harris corners and returns the list of keypoints. 
  You must use cv2.KeyPoint() method. 
  Args
  - corners:  list of Normalized corners.  
  - Ix: image derivative in X direction
  - Iy: image derivative in Y direction
  - threshold: only select corners whose R value is greater than threshold
  
  Returns 
  - keypoints: list of cv2.KeyPoint
  
  Notes:
  You must use cv2.KeyPoint() method. You should also pass 
  angle of gradient at the corner. You can calculate this from Ix, and Iy 

"""
def get_keypoints(corners, Ix, Iy, threshold):
    
    ### YOUR CODE HERE
    keypoints=[]
    
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if (corners[i,j]>threshold):
                keypoints.append(cv2.KeyPoint(j,i,1,np.degrees(np.arctan(Iy[i,j]/Ix[i,j]))+90,corners[i,j]))

        
    return keypoints

def arctan1(y,x):    
    if x==0:
        return 90.0 
    else:
        return np.degrees(np.arctan(y/x))
arctan2=np.vectorize(arctan1)        

def get_features(image, keypoints, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    #assert len(x) == len(y)
    def get_features(image, keypoints, feature_width, scales=None):
    fw1=feature_width//2
    fw2=feature_width//2+1

    image=np.pad(image,((8,8),(8,8)))


    image_x = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    image_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

    pts =np.int64(cv2.KeyPoint_convert(keypoints))
    for j in range(len(pts)):
        pts[j]=pts[j][::-1]

    kp_orientation=[]
    fp_orientation=[[]]*len(pts)

    for i in range(len(pts)):
        window=image[pts[i,0]-fw1:pts[i,0]+fw2,pts[i,1]-fw1:pts[i,1]+fw2]
        m=np.sqrt(np.add(image_x[pts[i,0]-fw1:pts[i,0]+fw2,pts[i,1]-fw1:pts[i,1]+fw2]**2,image_y[pts[i,0]-fw1:pts[i,0]+fw2,pts[i,1]-fw1:pts[i,1]+fw2]**2))
        #theta=np.degrees(np.arctan(image_y[pts[i,0]-fw1:pts[i,0]+fw2,pts[i,1]-fw1:pts[i,1]+fw2]/image_x[pts[i,0]-fw1:pts[i,0]+fw2,pts[i,1]-fw1:pts[i,1]+fw2]))%360
        theta=arctan2(image_y[pts[i,0]-fw1:pts[i,0]+fw2,pts[i,1]-fw1:pts[i,1]+fw2],image_x[pts[i,0]-fw1:pts[i,0]+fw2,pts[i,1]-fw1:pts[i,1]+fw2])

        #m=m/np.max(m)
        wndw=[]
        M=[]
        t=[]
        for j1 in range(window.shape[0]):
            for j2 in range(window.shape[1]):
                wndw.append(window[j1,j2])
                M.append(m[j1,j2])
                t.append(theta[j1,j2])
        bin=[0]*36

        for j in range(len(wndw)):
            for j1 in range(36):
                if t[j]>=j1*10 and t[j]<(j1+1)*10 :
                    bin[j1]+=M[j]

        idx=bin.index(max(bin))
        kp_orientation.append((idx+0.5)*10) #keypoint_orientation


        window=image[pts[i,0]-8:pts[i,0]+8,pts[i,1]-8:pts[i,1]+8]
        m=np.sqrt(np.add(image_x[pts[i,0]-8:pts[i,0]+8,pts[i,1]-8:pts[i,1]+8]**2,image_y[pts[i,0]-8:pts[i,0]+8,pts[i,1]-8:pts[i,1]+8]**2))
        #theta=np.degrees(np.arctan(image_y[pts[i,0]-8:pts[i,0]+8,pts[i,1]-8:pts[i,1]+8]/image_x[pts[i,0]-8:pts[i,0]+8,pts[i,1]-8:pts[i,1]+8]))%360
        theta=arctan2(image_y[pts[i,0]-8:pts[i,0]+8,pts[i,1]-8:pts[i,1]+8],image_x[pts[i,0]-8:pts[i,0]+8,pts[i,1]-8:pts[i,1]+8])
        #print(window.shape)
        temp=[]
        for h in range(4):
            for w in range(4):
                box=window[h*4:h*4+4,w*4:w*4+4]
                m_box=m[h*4:h*4+4,w*4:w*4+4]
                g=cv2.getGaussianKernel(4,90) @ cv2.getGaussianKernel(4,90).T
                if m_box.shape[0]==0:
                  print(pts[i])
                #print(m_box.shape)
                for k1 in range(m_box.shape[0]):
                    for k2 in range(m_box.shape[1]):
                        m_box[k1,k2]=m_box[k1,k2]*g[k1,k2]
                t_box=theta[h*4:h*4+4,w*4:w*4+4]

                wndw=[]
                M=[]
                t=[]

                for w1 in range(box.shape[0]):
                    for w2 in range(box.shape[1]):
                        M.append(m_box[w1,w2])
                        t.append(t_box[w1,w2])

                bin_box=[0]*8 
                       
                for j in range(len(t)):
                    for j1 in range(8):
                        if t[j]>=j1*45 and t[j]<(j1+1)*45 :
                            bin_box[j1]+=M[j]
                #temp.append(bin_box)

                for j in range(8):
                    temp.append(bin_box[j])        
        fp_orientation[i]=temp
        for j in range(len(fp_orientation[i])):
            fp_orientation[i][j]-=kp_orientation[i]
            if fp_orientation[i][j]<0 :
                fp_orientation[i][j]+=360
    fp_orientation=np.array(fp_orientation)
    
    return fp_orientation                                            

    


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv