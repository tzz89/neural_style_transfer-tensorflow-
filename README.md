## Neural Style transfer
In this project, I am replicating the original neural style transfer algorithm using VGG19 backbone in tensorflow. I have also created a simple UI using plotly dash (Definitely improvements needed on the apprearance but its not the priority at the moment). Lastly, I will be dockerizing the entire application using a tensorflow docker image with GPU support and uploading to dockerhub.

## Technologies used
1. Tensorflow-gpu
2. Docker/Docker-hub
3. Plotly-dash with dash-bootstrap
    - Responsive 
    - Uploading feature
    - Loading spinner
4. nvidia-docker2


### Original content images
<p align="left">
<img src="sample_data/content_pictures/content1.jpg" height="370px" width="370px">
<img src="sample_data/content_pictures/content3.jpg" height="370px" width="370px">
<p>

### Original style images
<p align="center">
<img src="sample_data/style_pictures/style1.jpg" width="200px">
<img src="sample_data/style_pictures/style2.jpg" width="200px">
<img src="sample_data/style_pictures/style3.jpg" width="200px">
<p><br>

### Example Obama
<p align="center">
<img src="gifs/gif_0_0.gif" width="200px">
<img src="gifs/gif_0_1.gif" width="200px">
<img src="gifs/gif_0_2.gif" width="200px">
<p>

### Example Wild bear
<p align="center">
<img src="gifs/gif_2_0.gif" width="200px">
<img src="gifs/gif_2_1.gif" width="200px">
<img src="gifs/gif_2_2.gif" width="200px">
<p>