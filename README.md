# CLIP+MLP Aesthetic Score Predictor

Train, use and visualize an aesthetic score predictor ( how much people like on average an image ) based on a simple neural net that takes CLIP embeddings as inputs.


Link to the AVA training data ( already prepared) :
https://drive.google.com/drive/folders/186XiniJup5Rt9FXsHiAGWhgWz-nmCK_r?usp=sharing


Visualizations of all images from LAION 5B (english subset with 2.37B images) in 40 buckets with the model sac+logos+ava1-l14-linearMSE.pth:
http://captions.christoph-schuhmann.de/aesthetic_viz_laion_sac+logos+ava1-l14-linearMSE-en-2.37B.html


## Usage
`simple_inference.py`: rate a single image
```
python simple_inference.py --model="path/to/model.ckpt" --image='path/to/image.png'
```

`inference_folder.py`: rate all images in a directory and write results to `<outfile>.csv`
```
python inference_folder.py --directory="path/to/image-dir" --model="path/to/model.ckpt" --out="<outfile>"
```

`aesthetic_predictor.py`: standalone python wrapper for using the models in other projects
```python
from PIL import Image
from aesthetic_predictor import AestheticPredictor

model = AestheticPredictor("path/to/model.ckpt")
# predict takes a list of PIL Images
model.predict([Image.open("image1.png"), Image.open("image2.jpg")]))
```
