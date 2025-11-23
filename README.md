# Computer Vision Tricks with OpenCV and Other Computer Vision Libraries

## Projects

- `qrcode_science`:  Detect QR codes with OpenCV erosion and dilations
- `polygon_science`:  Mask regions in a certain HSV color range and calculate polygons with OpenCV and Shapely

## Color Space

To find a color in HSV see the colorspace plot below.  In OpenCV hue (H) is in the range 0-179, saturation (S) 0-255 and value (V) 0-255.  In the plot below, V is set to a constant value of 255.

![HSV color space chart](assets/hsv_chart_constant_v.png)

See `hsv_colorspace_plotter.py` to see how this image was made with the `matplotlib` library.

## Setup

Create a Python environment (conda, `venv`, `uv`, etc.) and install the libraries with pip as follows.

```
pip install -r requirements.txt
```

Use scripts in the `utils` folder to help you choose color ranges.

## Helper scripts in `utils`

To explore interactively upper and lower values for the HSV values (the hue, saturation and value), the app/tool `interactive_color_thresholder.py`, in the `utils` folder, for any image Open CV can read.

To determine upper and lower values for colors using RGB values, the `find_hsv_ranges.py` will provide this (check the script to set input RGB values).

To explore HSV masks and image results try out `explore_hsv_masks.py` (check the script to set the color ranges).
