# Face Detection - Preliminary steps 

Here we complete the preliminary steps for creating a Face Detection Model

1. Create a Face Dataset by annotating Images with Faces : 

> a. We Downloaded Images with faces looking in different directions - Front, Left, Right, Up , Down , etc

> b. Resized te images to 400 x 400 size.

> c. Annotated the image using VGG Annotator : below are some screenshots of annotations done using VGG annotator 

![Annotation ex1](https://github.com/ravindrabharathi/Project1/blob/master/session19/via-1.png)

![Annotation ex2](https://github.com/ravindrabharathi/Project1/blob/master/session19/via-2.png)

![Annotation ex3](https://github.com/ravindrabharathi/Project1/blob/master/session19/via-3.png)


**The annotation file in json format** :  
  
  Github link : [annotated project file](https://github.com/ravindrabharathi/Project1/blob/master/session19/via_project_faces_eva_ravindra.json)
  Google Drive Link: https://drive.google.com/file/d/18SRff5bYWgCHQAS03FT19y1mxLC_d6mr/view?usp=sharing
  
**The images** are in a zip file located at :https://github.com/ravindrabharathi/Project1/blob/master/session19/EVA-assignment19.zip
    

2. Use KMeans clustering to find the top 4 clusters and the anchor box sizes corresponding to these 4 clusters . 
t-SNE was also tried but it did not give intuitive results , perhaps because of wrong parameters used or maybe because of the very small dataset.

**The KMeans clustering code** can be found in this [Notebook](https://github.com/ravindrabharathi/Project1/blob/master/session19/Anchor_boxes.ipynb)

**Anchor box sizes** : 

The four anchor box sizes found using KMeans clustering are (124 x 183) , (54 x 70) , (196 x 252) and (86 x 114)
