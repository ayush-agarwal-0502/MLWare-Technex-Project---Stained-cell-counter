# MLWare-Technex-Project---Stained-cell-counter

This project is about using Machine Learning models for counting the number of stained cells in the given images . This project was a submission for the event "MLWare" , held under "Technex" , the official techfest of IIT BHU Varanasi . The team members were me ( Ayush Agarwal ) and Anshuman Asauliya . The model files have been attavhed to this repository .

## Our approach towards the problem statment :

First we did a literature survey of the research papers availaible online to get an idea of the methods used for such problems before .

The most commonly used method by researchers for the cell counting problem is using Image Segmentation . This can be done using U-Nets . Fun fact , U-Nets were invented while solving a similar problem statment to ours . However , the data given to use does not have segmentation and we are not allowed to use external data or pretrained models hence this idea was dropped by our team .

One other way that is commonly used to solve such problems is using Xception ML model . It is a state of the art model for this purpose , and is closely related to the Inception model . However , as we couldn't use pretrained models , and training such a computationally expensive model was not possible on google colab as a student , hence we didn't use this idea too .

A less common but still used way is using the YOLO (You Only Look Once ) models for detection and counting of the cells . However , we could not use pretrained model , and training of yolo models requires coordinates as well as the enclosing box dimensions of the cells in the training dataset , which we didn't have , hence we couldn't use this too .

Then I decided to approach the problem statment using image processing techniques . My teammate on the other hand started approaching the problem with Convolutional Neural Networks (CNNs).

## Image Processing approach towards cell counting problem :

We were required to count the blue stained cells .

One initial and direct way to think is to do masking on the image using opencv . So masked the images for blue colour . However the results of masking were unsatisfactory , reason being that even in the brown part of the photos , there was blue part , which left lots of small dots and clouded the mask image with noise .

However , I observed that the blue layer of image is a very good representation of where the stained cell is . ( We know that an RGB image is made up of 3 layers , red , green , blue , and I decided to operate on the blue layer only ) . In the blue layer of image , even of the brown part of the image has blue also , then also the blue layer has more blackness at the point where the stained cell is . 

In the blue layer , the spots with the stained cell were black and rest part white . There isnt any thresholding function in opencv which can convert less white to fully white . So I inverted the image using bitwise inversion operator .
