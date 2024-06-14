import equinox as eqx
import numpy as np
import jax.numpy as jnp
import io

from jaxtyping import Array
from typing import Tuple, Union
from jax import vmap

from PIL import Image as PILImage
from IPython.display import display, Image as IPImage

class Patch(eqx.Module):
    """Patch Embedding settings"""

    img_size: Tuple[int] = eqx.field(static=True)
    patch_size: Tuple[int] = eqx.field(static=True)
    grid_size: Tuple[int] = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    in_chans: int = eqx.field(static=True)
    flatten: bool = eqx.field(static=True)
    
    def __init__(
        self,
        img_size: Union[int, Tuple[int]] = 224,
        patch_size: Union[int, Tuple[int]] = 16,
        in_chans: int = 3,
        flatten: bool = True,
    ):
        """
        **Arguments:**

        - `img_size`: The size of the input image. Defaults to `(224, 224)`
        - `patch_size`: Size of the patch to construct from the input image. Defaults to `(16, 16)`
        - `in_chans`: Number of input channels. Defaults to `3`
        - `embed_dim`: The dimension of the resulting embedding of the patch. Defaults to `768`
        - `flatten`: If enabled, the `2d` patches are flattened to `1d`
        """
        super().__init__()

        self.in_chans = in_chans

        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

    def __call__(self, x: Array) -> Array:
        """
        Inputs:
            x - jax.Array representing the image of shape [C, H, W]
        """
        assert (self.in_chans, self.img_size[0], self.img_size[1]) == x.shape
        x = x.reshape(self.in_chans, self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1])
        x = x.transpose(1, 3, 2, 4, 0)    # [H', W', p_H, p_W, C]
        x = x.reshape(-1, *x.shape[2:])   # [H'*W', p_H, p_W, C]
        if self.flatten:
            x = x.reshape(x.shape[0], -1) # [H'*W', p_H*p_W*C]
        
        return x
    
    def inverse(self, x: Array) -> Array:
        """
        Inputs:
            x - jax.Array representing the patches of shape [p, p_H, p_W, C] or [p, p_H * p_W * C]
        """

        x = x.reshape(self.grid_size[0], self.grid_size[1], self.patch_size[0], self.patch_size[0], self.in_chans)
        x = x.transpose(4, 0, 2, 1, 3)  # [C, H', p_H, W', p_W]
        x = x.reshape(self.in_chans, self.grid_size[0] * self.patch_size[0], self.grid_size[1] * self.patch_size[1])

        return x
    

def animate_image_sequence(images, num_steps = None, path = None):
    # remove batch dimension if it exists
    if not isinstance(images, list) and images.shape[0] == 1:
        images = images[0]

    def get_image_data(images, t):
        img = images[t]
        
        if img.shape[-1] == 3:
            # color
            return PILImage.fromarray(img)
        elif len(img.shape) == 3:
            # 2-channel grayscale
            return PILImage.fromarray((np.array(img[0]) * 255).astype(np.uint8)).resize((256, 256), PILImage.NEAREST)
        elif len(img.shape) == 2:
            # grayscale
            return PILImage.fromarray((np.array(img) * 255).astype(np.uint8)).resize((256, 256), PILImage.NEAREST)
        else:
            print(img.shape)
            raise Exception("Unknown image format")

    if num_steps is None:
        if isinstance(images, list):
            num_steps = len(images)
        else:
            num_steps = images.shape[0]

    pil_images = []
    for i in range(num_steps):
        pil_images.append(get_image_data(images, i))

    # Save as GIF
    gif_buffer = io.BytesIO()
    pil_images[0].save(gif_buffer, format='GIF', append_images=pil_images[1:], save_all=True, duration=1000/30, loop=0)
    # Display in Jupyter Notebook
    display(IPImage(data=gif_buffer.getvalue(), width=256, height=256))
    if path is not None:
        pil_images[0].save(path, format='GIF', append_images=pil_images[1:], save_all=True, duration=1000/30, loop=0)


def greyscale_convert(batched_imgs, quantize):
    gs_imgs = []
    for imgs in batched_imgs:
        gs_imgs.append([])
        for img in imgs:
            gs_imgs[-1].append( jnp.array(PILImage.fromarray(img).convert('L')) // quantize )
        gs_imgs[-1]  = jnp.stack(gs_imgs[-1])

    return jnp.stack(gs_imgs)