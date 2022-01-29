# **Image Captioning**
<img src="examples/model_arch.png" alt="alt text" width="800" height="800">
<br>
<br>
This is the final project from Deep learning school (or DLS), part_2.  
<br>
<br>
Image captioning is a text description of image.  
Model takes raw image (i.e. .jpg), extract CNN-features and use these features in RNN. RNN returns logits for tokens (words). Logits are converted to probabilty. Beam search (or any other) uses probability to get image description (caption).   
<br>
This repository contains jupyter notebook with model training, pretrained model (encoder-decoder), demonstration notebook and brief summary .pdf (on russian).

<br>
Check out model by clicking on icon:   

<a href="https://githubtocolab.com/A-n-d-r-e-w-y/DLS_part_2/blob/main/Final_project_Image_Captioning/demo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>  

or open `demo.ipynb` in Google Collab.
