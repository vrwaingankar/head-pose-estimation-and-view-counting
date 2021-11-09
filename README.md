# head-pose-estimation-and-view-counting

The following repository is the advanced version of keeping count on number of people looking into the advertisement as mentioned in my other repository.
The updated version keeps a track on the pitch, roll and yaw of a person's head and if it is out of bound in any of the 3 angles then it is considered that he/she is not looking into the ad.
The earlier code didn't consider any specific bounds and checked for full face for updating the view count.
The following code uses facial landmarks and Perspective-n-Point (PnP) for calculating the pitch, roll and yaw of the face with nose as the centre.
It gives a detailed output of the number of faces detected and number of faces looking within the bounds.
The results are stored in Individual_Samples.txt file and output images as jpeg

Note: model file has not been uploaded because of memory limit on github
