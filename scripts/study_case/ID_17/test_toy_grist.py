from __future__ import division, print_function

import numpy as np
np.random.seed(98765)

import sys
sys.path.append("/data")
from scripts.study_case.ID_17.tf_unet import image_gen
from scripts.study_case.ID_17.tf_unet import unet_grist
from scripts.study_case.ID_17.tf_unet import util

from datetime import datetime
s1 = datetime.now()
nx = 572
ny = 572
generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)
x_test, y_test = generator(1)
net = unet_grist.Unet(channels=generator.channels, cost='dice_coefficient', n_class=generator.n_class, layers=3, features_root=16)
trainer = unet_grist.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(generator, "./unet_trained", training_iters=32, epochs=30, display_step=2)
# print(f"Time cost {datetime.now() - s1}")
# x_test, y_test = generator(1)
#
# prediction = net.predict("./unet_trained/model.ckpt", x_test)
#
# fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
# ax[0].imshow(x_test[0,...,0], aspect="auto")
# ax[1].imshow(y_test[0,...,1], aspect="auto")
# mask = prediction[0,...,1] > 0.9
# ax[2].imshow(mask, aspect="auto")
# ax[0].set_title("Input")
# ax[1].set_title("Ground truth")
# ax[2].set_title("Prediction")
# fig.tight_layout()
# fig.savefig(".toy_problem.png")



