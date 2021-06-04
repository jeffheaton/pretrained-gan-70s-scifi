import random
from pathlib import Path
import pickle
import tempfile
import sys
import numpy as np
import tensorflow as tf
import PIL
import cog

sys.path.insert(0, "/stylegan2-ada")

import dnnlib.tflib as tflib


class Predictor(cog.Predictor):
    def setup(self):
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        sys.argv = ["cog_infer.py", "--style_imgs", "japanese_waves.png"]
        with self.graph.as_default(), self.sess.as_default():
            tflib.init_tf()
            with open("70s-scifi-gan-2020-12-12.pkl", "rb") as f:
                _, _, self.Gs = pickle.load(f)
            init_random_state(self.Gs, 10)

    @cog.input(
        "seed", default=-1, type=int, help="Random seed. For random images, use -1"
    )
    def predict(self, seed):
        if seed == -1:
            seed = random.randint(0, 100000)

        with self.graph.as_default(), self.sess.as_default():
            z = seed2vec(self.Gs, seed)
            img_data = generate_image(self.Gs, z, 1.0)

        img = PIL.Image.fromarray(img_data, "RGB")
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        img.save(out_path)

        return out_path


def seed2vec(Gs, seed):
    rnd = np.random.RandomState(seed)
    return rnd.randn(1, *Gs.input_shape[1:])


def init_random_state(Gs, seed):
    rnd = np.random.RandomState(seed)
    noise_vars = [
        var
        for name, var in Gs.components.synthesis.vars.items()
        if name.startswith("noise")
    ]
    tflib.set_vars(
        {var: rnd.randn(*var.shape.as_list()) for var in noise_vars}
    )  # [height, width]


def generate_image(Gs, z, truncation_psi):
    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        "output_transform": dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        "randomize_noise": False,
    }
    if truncation_psi is not None:
        Gs_kwargs["truncation_psi"] = truncation_psi

    label = np.zeros([1] + Gs.input_shapes[1][1:])
    images = Gs.run(z, label, **Gs_kwargs)  # [minibatch, height, width, channel]
    return images[0]
