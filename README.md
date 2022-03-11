# MLWare-Technex-Project---Stained-cell-counter

This project is about using Machine Learning models for counting the ***number of stained cells*** in the given images . This project was a submission for the event "MLWare" , held under "Technex" , the official techfest of IIT BHU Varanasi . The team members were me ( Ayush Agarwal ) and Anshuman Asauliya . The model files have been attached to this repository .

## Our initial approach towards the problem statment :

First we did a literature survey of the research papers availaible online to get an idea of the methods used for such problems before .

The most commonly used method by researchers for the cell counting problem is using **Image Segmentation** . This can be done using **U-Nets** . Fun fact , U-Nets were invented while solving a similar problem statment to ours . However , the data given to use does not have segmentation and we are not allowed to use external data or pretrained models hence this idea was dropped by our team .

One other way that is commonly used to solve such problems is using **Xception ML model** . It is a state of the art model for this purpose , and is closely related to the Inception model . However , as we couldn't use pretrained models , and training such a computationally expensive model was not possible on google colab as a student , hence we didn't use this idea too .

A less common but still used way is using the **YOLO (You Only Look Once )** models for detection and counting of the cells . However , we could not use pretrained model , and training of yolo models requires coordinates as well as the enclosing box dimensions of the cells in the training dataset , which we didn't have , hence we couldn't use this too .

Then I decided to approach the problem statment using **image processing techniques and blob detection algorithms** . My teammate on the other hand started approaching the problem with Convolutional Neural Networks (CNNs).

## Image Processing and blob detection algorithm approach towards cell counting problem :

### Easy case solved by model :

![image](https://user-images.githubusercontent.com/86561124/157831306-304e5f8e-780a-4514-890e-26c4c577c1ed.png)

### Hard cases solved by model :

![image](https://user-images.githubusercontent.com/86561124/157831538-3d9fef09-ee1a-4226-a557-30a6163f1e40.png)

![image](https://user-images.githubusercontent.com/86561124/157831612-b211b52a-4434-4942-bfe8-531128de36e8.png)

### The places where model goes wrong :
(Last one being the major reason my model error was not single digit I feel )

![image](https://user-images.githubusercontent.com/86561124/157831651-4594359f-2a03-430f-a5b3-1a7b8a570d1c.png)

![image](https://user-images.githubusercontent.com/86561124/157831976-5827324c-a6e8-4e56-a742-9e39dc4b026f.png)




We were required to count the blue stained cells .

One initial and direct way to think is to do **masking** on the image using opencv . So masked the images for blue colour . However the results of masking were unsatisfactory , reason being that even in the brown part of the photos , there was blue part , which left lots of small dots and clouded the mask image with noise .

However , I observed that the **blue layer** of image is a very good representation of where the stained cell is . ( We know that an RGB image is made up of 3 layers , red , green , blue , and I decided to operate on the blue layer only ) . In the blue layer of image , even of the brown part of the image has blue also , then also the blue layer has more blackness at the point where the stained cell is . 

In the blue layer , the spots with the stained cell were black and rest part white . There isnt any thresholding function in opencv which can convert less white to fully white . So I inverted the image using **bitwise inversion** operator . 

Now I could easily use the **cv2.THRESHOLD_TOZERO** to get rid of the low probablity points .

Now , as per the problem statment , we were required to ignore the stained cells at the borders , hence I added another **layer of black border** manually on the image .

Now , we have a black image with the white patches representing stained cells . To further ease up the processing , I added **binary thresholding** using opencv .

Now , for the detection part , there are many **Blob detecting algorithms** , such as **RegionProps ,Laplacian of Gaussian (LoG) , Difference of Gaussian (DoG) , Determinant of Hessian (DoH)** in skimage ( scikit learn tools for images ) . Due to lack of time , I could only apply **RegionProps** . 

Initially it gave too many blobs , however , after **thresholding the detected blobs by area** ( i.e. ignoring the lower area blobs ) , we could achieve an MSE of about 14 , which seems to be pretty good . The model was not always exact , however it was pretty accurate (a difference of 1 or 2 mostly) in the cases of less overlapping cells . This showed that the error was high mostly due to the outlier cases ( overlapping cells , more than 50 cells in an image , whole image being largely covered with blue etc ) . Furthermore , by applying blurring , the accuracy was even more improved .

This was all for the image processing approach . I also tried using the simpleblobdetector library from opencv , however it crashed on google colab notebook . Also tried opencv contour detecting functions but didn't work probably bcoz colab and opencv do not work well together .

## Deep Learning approach :

The images were fed to the **Convolutional Neural Network** . Since the dataset was huge , google colab kept throwing resource exhausion error a lot of times , hence training a huge convnet was not a feasible option without access to GPUs . Hence , we used a **simple structure** of the CNN , which somehow managed to perform slightly better than the image processing method ( almost equal performance ) . Adam optimiser was used , with MSE taken as the loss function . The model was trained with batch gradient descent taking batch  of 1000s . The model was trained for 10 epochs .
