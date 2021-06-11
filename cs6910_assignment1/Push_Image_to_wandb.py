# CODE to push plot as an image to wandb
import wandb
from matplotlib import pyplot as plt
path_to_img = "<Provide Absolute file Path >"
im = plt.imread(path_to_img)

# Initialize run
wandb.init(project="<Provide project name in wandb>")

# Log image(s)
wandb.log({"img": [wandb.Image(im)]})