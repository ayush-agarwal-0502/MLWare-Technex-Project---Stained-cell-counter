# MLWare-Technex-Project---Stained-cell-counter

This project is about using Machine Learning models for counting the number of stained cells in the given images . This project was a submission for the event "MLWare" , held under "Technex" , the official techfest of IIT BHU Varanasi . The team members were me ( Ayush Agarwal ) and Anshuman Asauliya . The model files have been attavhed to this repository .

## Our approach towards the problem statment :

First we did a literature survey of the research papers availaible online to get an idea of the methods used for such problems before .

The most commonly used method by researchers for the cell counting problem is using Image Segmentation . This can be done using U-Nets . Fun fact , U-Nets were invented while solving a similar problem statment to ours . However , the data given to use does not have segmentation and we are not allowed to use external data or pretrained models hence this idea was dropped by our team .

One other way that is commonly used to solve such problems is using Xception ML model . It is a state of the art model for this purpose , and is closely related to the Inception model . However , as we couldn't use pretrained models , and training such a computationally expensive model was not possible on google colab as a student , hence we didn't use this idea too .

A less common but still used way is using the YOLO (You Only Look Once ) models for detection and counting of the cells . However , we could not use pretrained model , and training of yolo models requires coordinates as well as the enclosing box dimensions of the cells in the training dataset , which we didn't have , hence we couldn't use this too .
