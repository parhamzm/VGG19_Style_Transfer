# VGG19_Style_Transfer


---------------------------------------------------
Transfer Learning is a powerful teachnique that can save a lot of time and can provide amazing results in a variaty of Applications. One of these applications is commonly reffered to as: <b>Style Transfer<b>
  
  
- Style Transfer: Deals with the process of differentiating between the image content and the image style. Where image content relates more heavily to the objects in the image and the specific layout of these objects in the given image. While image style refers to the specific style within an image. At a general level, this can refer to the basic components of an image, such as color and texture. This can also include more specific features such as the style of paint strokes or the style of the painting technique used.
  
- The game of Style Transfer is the combined contents of an image with the style of a completely different image. Effectively transferring the style of the second image to the first. The resulting image of the combined content and style is referred to as the target image. This technique can have fascinating results.


<img src="images/style_transfer.png" style="width:500px;height:250;" align="center">
  
<caption><center> <u><b>Figure 1</u></b>: Examples of Style Transfer <br> </center></caption>










# VGG19 Style Transfer

![Style Transfer](examples/output_example.jpg)

## Overview

**VGG19 Style Transfer** is a PyTorch implementation of the Neural Style Transfer algorithm based on the pretrained **VGG19** network. The method combines the **content** of one image with the **style** of another to generate a new image that preserves the content while adopting the artistic style.

This repository is inspired by the pioneering work by Gatys et al. ([A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)) and follows the standard approach of optimizing an image to minimize a weighted combination of content and style losses.

---

## Features

- Utilizes pretrained VGG19 to extract hierarchical features for style and content.
- Computes content loss and style loss using feature maps and Gram matrices.
- Supports user-defined content and style images.
- Adjustable weights for content and style to control the output.
- Saves intermediate results for visualization.
- Easy-to-use command-line interface for style transfer execution.

---

## Sample Images

| Content Image                | Style Image                   | Stylized Output               |
|-----------------------------|------------------------------|------------------------------|
| ![content](examples/content.jpg) | ![style](examples/style.jpg) | ![output](examples/output.jpg) |

---

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch
- torchvision
- PIL (Pillow)
- matplotlib
- numpy

### Steps

```bash
git clone https://github.com/parhamzm/VGG19_Style_Transfer.git
cd VGG19_Style_Transfer

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

---

## Usage

Run the style transfer script with your content and style images:

```bash
python style_transfer.py --content_path path/to/content.jpg --style_path path/to/style.jpg --output_path output.jpg --content_weight 1e5 --style_weight 1e10 --num_steps 300
```

### Arguments

| Argument         | Description                            | Default   |
|------------------|------------------------------------|-----------|
| `--content_path` | Path to the content image            | Required  |
| `--style_path`   | Path to the style image              | Required  |
| `--output_path`  | Path where the output image is saved | Required  |
| `--content_weight` | Weight for content loss              | 1e5       |
| `--style_weight` | Weight for style loss                | 1e10      |
| `--num_steps`    | Number of optimization iterations    | 300       |

---

## How It Works

1. **Feature Extraction**: Extracts features from both content and style images using specific convolutional layers of VGG19.
2. **Content Loss**: Measures how much the generated image deviates from the content image’s features.
3. **Style Loss**: Uses Gram matrices of style features to capture texture and style information.
4. **Optimization**: Updates the generated image by minimizing the combined loss using gradient descent.

---

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style*. [Paper](https://arxiv.org/abs/1508.06576)
- [PyTorch Neural Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Thanks to the original authors of the Neural Style Transfer method.
- Thanks to the PyTorch community for excellent tutorials and documentation.
- Inspired by and adapted from official PyTorch examples and academic papers.

---

If you want, I can help you generate a `requirements.txt` or assist with running examples!
