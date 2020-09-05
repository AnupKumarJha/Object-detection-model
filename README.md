# Object-detection-model
---

---

**Image Classification:** Takes an image and predict the object in an image

Like Cat and Dog Classifier predicting the given image has dog or cat

![img](http://cv-tricks.com/wp-content/uploads/2017/12/cat_dog-300x200.jpeg)

if the the image has both dog and cat
we can train the multilabel classifier

![img](http://cv-tricks.com/wp-content/uploads/2017/12/cat_dog-300x200.jpeg)

However we dont have location of cat or dog

The problem of identifing the location of an object(given the class in an image is "localization")

![img](http://cv-tricks.com/wp-content/uploads/2017/12/detection-vs-classification-300x220.png)

However, if the object class is not known, we have to not only determine the location but also predict the class of each object.

**"Predicting the location of the object along with the class is called object Detection"**

SO once we have predicting the image this will have the 5 feature

1. CLass Name
2. Bounding box 4 cordinates



![img](http://cv-tricks.com/wp-content/uploads/2017/12/Multi-class-object-detection.png)



Object detection is modeled as classification problem Where we take windows of fixed size window and feed to image classifier

![img](http://cv-tricks.com/wp-content/uploads/2017/12/Sliding-window.gif)



There can be image of varying size may be present in the image.

**dea is that we resize the image  at multiple scales and we count on the fact that our chosen window size  will completely contain the object in one of these resized images**

![img](http://cv-tricks.com/wp-content/uploads/2017/12/Small-object-300x214.jpg)



For that purpose we need to make image pyrimd.

![img](http://cv-tricks.com/wp-content/uploads/2017/12/pyramid-269x300.png)

There may be different size of image with different aspect ratio. this can be removed using this.



**Some Term**

**Derivative**: It is change of function f(x,y,...) and it is the scaler

**Directional Deriv**ative :: it is change of function f(x,y) in x direction or any direction

**Gradients** : it is maximum change in function on point to its neighbouring point. 



1. Image gradients Vector

2. ![img](https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Blue-green-gradient.jpg/300px-Blue-green-gradient.jpg)

   

   Change of colour on given point.he most common way to approximate the image gradient is to [convolve](https://en.wikipedia.org/wiki/Convolution) an image with a kernel, such as the [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator) or [Prewitt operator](https://en.wikipedia.org/wiki/Prewitt_operator).

Like for edge detection 

![Left: Black and white picture of a cat. Center: The same cat, displayed in a gradient image in the x direction. Appears similar to an embossed image. Right: The same cat, displayed in a gradient image in the y direction. Appears similar to an embossed image.](https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Intensity_image_with_gradient_images.png/400px-Intensity_image_with_gradient_images.png)



```python
import numpy as np
import scipy
import scipy.signal as sig
# With mode="L", we force the image to be parsed in the grayscale, so it is
# actually unnecessary to convert the photo color beforehand.
img = scipy.misc.imread("manu-2004.jpg", mode="L")

# Define the Sobel operator kernels.
kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

G_x = sig.convolve2d(img, kernel_x, mode='same') 
G_y = sig.convolve2d(img, kernel_y, mode='same') 

# Plot them!
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Actually plt.imshow() can handle the value scale well even if I don't do 
# the transformation (G_x + 255) / 2.
ax1.imshow((G_x + 255) / 2, cmap='gray'); ax1.set_xlabel("Gx")
ax2.imshow((G_y + 255) / 2, cmap='gray'); ax2.set_xlabel("Gy")
plt.show()

Sobel operator
```



## Histogram of Oriented Gradients (HOG)

Working.

1. Preprocess the image, including resizing and color normalization.

2. Compute the gradient vector of every pixel, as well as its magnitude and direction.

3. Divide the image into many 8x8 pixel cells. In each cell, the  magnitude values of these 64 cells are binned and cumulatively added  into 9 buckets of unsigned direction 

4. Then we slide a 2x2 cells (thus 16x16 pixels) block across the image. In each block region, 4 histograms of 4 cells are concatenated into  one-dimensional vector of 36 values and then normalized to have an unit  weight. The final HOG feature vector is the concatenation of all the block  vectors. It can be fed into a classifier like SVM for learning object  recognition tasks.



## Image Segmentation (Felzenszwalb’s Algorithm)



When there exist multiple objects in one image (true for almost every real-world photos), we need to identify a region that potentially  contains a target object so that the classification can be executed more efficiently.

Felzenszwalb and Huttenlocher ([2004](http://cvcl.mit.edu/SUNSeminar/Felzenszwalb_IJCV04.pdf)) proposed an algorithm for segmenting an image into similar regions using a graph-based approach

Here we construct a graph from the image then apply the Segmentation algorithms

**GRAPH CONSTRUCTION**

- **Grid Graph**: Each pixel is only connected with  surrounding neighbours (8 other cells in total). The edge weight is the  absolute difference between the intensity values of the pixels.
- **Nearest Neighbor Graph**: Each pixel is a point in  the feature space (x, y, r, g, b), in which (x, y) is the pixel location and (r, g, b) is the color values in RGB. The weight is the Euclidean  distance between two pixels’ feature vectors.
- ![Manu 2013 Image Segmentation](https://lilianweng.github.io/lil-log/assets/images/manu-2013-segmentation.png)

## Selective Search

Selective search is a common algorithm to provide region proposals  that potentially contain objects. It is built on top of the image  segmentation output and use region-based characteristics (NOTE: not just attributes of a single pixel) to do a bottom-up hierarchical grouping.

### How Selective Search Works

1. At the initialization stage, apply Felzenszwalb and Huttenlocher’s graph-based image segmentation algorithm to create regions to start  with.
2. Use a greedy algorithm to iteratively group regions together:    
   - First the similarities between all neighbouring regions are calculated.
   - The two most similar regions are grouped together, and new  similarities are calculated between the resulting region and its  neighbours.
3. The process of grouping the most similar regions (Step 2) is repeated until the whole image becomes a single region.



<img src="https://lilianweng.github.io/lil-log/assets/images/selective-search-algorithm.png" alt="Selective Search Algorithm" style="zoom: 67%;" />



**Summary**

1. the concept of  image gradient vector and how HOG algorithm summarizes the information  across all the gradient vectors in one image;  
2. how the image  segmentation algorithm works to detect regions that potentially contain  objects;
3. how the Selective Search algorithm refines the outcomes of  image segmentation for better region proposal.





## CNN for Image Classification

**Operation**

![Image for post](https://miro.medium.com/max/1255/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg)



### Convolution Operation

![Convolution Operation](https://lilianweng.github.io/lil-log/assets/images/convolution-operation.png)

Here a small kernel is hover on the image this kernel(filter) produces the one output for each kernel

![Convolution Operation](https://lilianweng.github.io/lil-log/assets/images/numerical_no_padding_no_strides.gif)

![Convolution Operation](https://lilianweng.github.io/lil-log/assets/images/numerical_padding_strides.gif)

*Two examples of 2D convolution operation: (top) no padding and 1x1  strides; (bottom) 1x1 border zeros padding and 2x2 strides. (Image  source: [deeplearning.net](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html))

Output of convolution operation is *featured map* Depending upon th filter we may have the different outputs like the edge detection ,bluer,sharp etc...

**STRIDES**

No of pixels shifts over the input matrix for above  example  it is one. for other it be two.

- Pad the picture with zeros (zero-padding) so that it fits
- Drop the part of the image where the filter did not fit. This is called valid padding which keeps only valid part of the image.

**Non Linearity (ReLU)**

ReLU stands for Rectified Linear Unit for a non-linear operation. The output is 


$$
ƒ(x) = max(0,x).
$$
 The purpose the ReLU is to insert the non linearity.

There are many activation function  .

Little intro TO activation function


$$
x = (weight * input) + bias
$$
this is what is given to neuron a linear input in neural network.

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2017/10/17123344/act.png)

This activation is ussed to break the linearity.

> <!--A neural network without an actvcation function is essentially just a linear regression model--> 

Some of the Popular activation function 

1. Binary step function 


   $$
   f(x) = 1 ,x > 0\\
    = 0 ,othrewise
   $$

2. Linear Function 

$$
f(x)=ax
$$

3. Sigmoid (Widly used)   transform the value between the 0 to 1.

   

$$
f(x)=\frac{1}{1+e^{-x}}
$$



4. Tanh:: very similar to sigmoid only difference is it is symmetric towards to origin

   

$$
tanh(x)=2sigmoid(2x)-1\\
\\
or\\
\frac {e^x-e^{-x}}{e^x+e^{-x}}
$$

5. ReLU 

   The main advantage of using the ReLU function over other activation  functions is that it does not activate all the neurons at the same time.

   This means that the neurons will only be deactivated if the output of the linear transformation is less than 0. 
   $$
   f(x)=max(0,x)
   $$

6. 

7. Leaky ReLU:

   Leaky ReLU function is nothing but an improved version of the ReLU  function. As we saw that for the ReLU function, the gradient is 0 for  x<0, which would deactivate the neurons in that region.

   Leaky ReLU is defined to address this problem. Instead of defining  the Relu function as 0 for negative values of x, we define it as an  extremely small linear component of x. Here is the mathematical  expression-


$$
f(x)=0.01x,x<0\\
= x,x>=0
$$

##### 

7. 

7. Parameterised ReLU

This is another variant of ReLU that aims to solve the problem of  gradient’s becoming zero for the left half of the axis.

 The  parameterized ReLU, as the name suggests, introduces a new parameter as a slope of the negative part of the function. Here’s how the ReLU  function is modified to incorporate the slope parameter- 
$$
f(x) = x, x>=0\\
    = ax, x<0
$$

8. Exponential Linear Unit




$$
f(x)=x,x>=0\\
=a(e^x-1) ,x<0
$$

9. Swish

   produces good result for deepar model

   like ReLU

   

$$
f(x)=x*sigmoid(x)\\
f(x) = x/(1-e^{-x})
$$



10. Soft max

    

    Can be considered the combination  of the multiple sigmoid like the sigmoid return the value return 0 to 1 which represent the probability for belonging to the particular class.

    i,e sigmoid is used for binary classification and softmax is used for multi class classification. It return the prob for belonging for a data point to each class.


$$
\sigma(z)_i=\frac{e^{z_i}}{\sum\limits_{k=1}^K e^{z_k}},i=1,...K 
$$

## Choosing the right Activation Function

However depending upon the properties of the problem we might be able to make a better choice for easy and quicker convergence of the  network.

- Sigmoid functions and their combinations generally work better in the case of classifiers
- Sigmoids and tanh functions are sometimes avoided due to the vanishing gradient problem
- ReLU function is a general activation function and is used in most cases these days
- If we encounter a case of dead neurons in our networks the leaky ReLU function is the best choice
- Always keep in mind that ReLU function should only be used in the hidden layers
- As a rule of thumb, you can begin with using ReLU function and then  move over to other activation functions in case ReLU doesn’t provide  with optimum results







[This]: https://arxiv.org/abs/1803.01164

 paper
