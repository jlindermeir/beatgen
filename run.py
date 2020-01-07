# Script to illustrate the usage of the modules
import numpy as np
import matplotlib.pyplot as plt
from models import VAE
from MIDI import *

# Inititalize the model with pretrained weights and show model summary
vae = VAE(16, (64, 22,), weights = 'examples/weights/2020-01-06_18_03_16')
# Load the note dictionary
nd = np.load('notedict.npy', allow_pickle = True).item()
# Define a latent vector
lv = np.array([-1.2200589,  0.309402,    0.99552226,  0.61033183, -1.9666643,   1.2859561,
  -0.76054627,  0.22162047,  0.4384073,   0.19943899, -0.09905571,  1.7563442,
  -0.5543848,  -1.1067173,  -0.97960794, -0.4229455 ])
# Decode the LV into a bar representation
bars = vae.decoder.predict(lv[None, :])
# Display a figure of the bars
plt.figure(figsize=(12,3))
plt.gray()
plt.imshow(1 - np.concatenate(bars).T, aspect = 'auto')
plt.show()
# Write the bars to a midi file
writeBarArray(bars, 'example', nd)
