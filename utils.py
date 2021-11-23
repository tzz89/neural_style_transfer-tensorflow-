from io import BytesIO
from PIL import Image
import base64


def image_read_and_resize(image, target_size=(300, 240), input_type='filepath'):
    """
    Takes in an imagefile path, resize the image and returns a base64 encoded string to be used with html.Img component
    Args:
        image ([str]): Can be either a filepath or a base64 encoded image (example upload component content_string)
        target_size (tuple): Targeted image size. Defaults to (300, 240).
        input_type (str): [filepath | base64]. Defaults to 'filepath'.
    """
    target_height, target_width = target_size

    if input_type == "filepath":
        img_array = Image.open(image)

    elif input_type == "base64":
        binary_img = BytesIO(base64.b64decode(image))
        binary_img.seek(0)
        img_array = Image.open(binary_img)

    image_height, image_width = img_array.size
    rescale_factor = max(image_height/target_height,
                         image_width/target_width)
    img_array = img_array.resize((int(image_height/rescale_factor),
                                  int(image_width/rescale_factor)), Image.ANTIALIAS)
    binary_img = BytesIO()
    img_array.save(binary_img, format='png')
    binary_img.seek(0)  # reset cursor
    encoded_img = base64.b64encode(binary_img.read())

    return encoded_img
