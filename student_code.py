import numpy as np
#### DO NOT IMPORT cv2 

def my_imfilter(image, filter):  
    """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that I can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ### 
    m = image.shape[0]
    n = image.shape[1]
    temp = np.zeros((m,n))
    temp_image = np.dstack([image,temp])
    c = temp_image.shape[2]
    
    k1=filter.shape[0]
    k2=filter.shape[1]
    p1=k1//2
    p2=k2//2
    
    #pading
    img=np.zeros((m+2*p1,n+2*p2,c-1))
    for i in range(c-1):
        img[:,:,i]=np.pad(temp_image[:,:,i],((p1,p1),(p2,p2)),mode="reflect")
    
    #filtering
    img1=np.zeros((m,n,c-1))
    m,n,c=img.shape
    for t in range(c):
        for i in range(m-k1+1):
            for j in range(n-k2+1):
                img1[i,j,t]=np.sum(img[i:i+k1,j:j+k2,t]*filter)
    
    filtered_image=img1.copy()
  ### END OF STUDENT CODE ####
  ############################
    return filtered_image

def create_hybrid_image(image1, image2, filter):
    
    """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """
    
    m = image1.shape[0]
    n = image1.shape[1]
    temp = np.zeros((m,n))
    temp_image = np.dstack([image1,temp])
    c = temp_image.shape[2]
    if c==2:
        assert image1.shape[0] == image2.shape[0]
        assert image1.shape[1] == image2.shape[1]
    elif c==4:
        assert image1.shape[0] == image2.shape[0]
        assert image1.shape[1] == image2.shape[1]
        assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###
    low_frequencies=my_imfilter(image1, filter)
     
    high_frequencies1=my_imfilter(image2, filter)
    image2 = image2.reshape(high_frequencies1.shape)
    high_frequencies=image2-high_frequencies1
         
    hybrid_image=low_frequencies+high_frequencies  
    hybrid_image=hybrid_image+abs(hybrid_image.min())
    hybrid_image=hybrid_image/hybrid_image.max()    
  ### END OF STUDENT CODE ####
  ############################

    return low_frequencies, high_frequencies, hybrid_image
