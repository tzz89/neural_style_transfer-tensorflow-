# tensorflow lib
import tensorflow as tf

# python lib
import os
import random
import numpy as np

# image library
import PIL


def seed_python(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_tf(seed):
    tf.random.set_seed(seed)


def image_clip(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def loss_function(model_output, target_content, target_style, config):
    style_tensors = model_output['style']
    content_tensors = model_output['content']
    num_style_tensor = len(style_tensors)
    num_target_tensor = len(content_tensors)

    style_loss = tf.add_n([
        tf.reduce_mean((style_tensors[layer_name]-target_style[layer_name])**2)
        for layer_name in style_tensors.keys()
    ])
    style_loss *= config['style_weight']/num_style_tensor

    content_loss = tf.add_n([
        tf.reduce_mean(
            (content_tensors[layer_name]-target_content[layer_name])**2)
        for layer_name in content_tensors.keys()
    ])

    content_loss *= config['content_weight']/num_target_tensor
    return style_loss + content_loss


def train_step(image, target_content, target_style, extractor, optimizer, config):
    # extract the target content and style matrix
    with tf.GradientTape() as tape:
        model_output = extractor(image)  # b,w,h,c
        loss = loss_function(model_output, target_content,
                             target_style, config)
        loss += config["total_variation_weight"] * \
            tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(image_clip(image))
    return image


def cal_gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    return result/tf.cast(input_shape[1]*input_shape[2], dtype=tf.float32)


def get_vgg19():
    required_layer_names = ['block1_conv1',
                            'block2_conv1',
                            'block3_conv1',
                            'block4_conv1',
                            'block5_conv1'] + ['block5_conv2']

    model_filepath = os.path.join("models", "vgg19.h5")
    if os.path.isfile(model_filepath):
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights=model_filepath)
    else:
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

    vgg_input = vgg.input
    outputs = [vgg.get_layer(name).output for name in required_layer_names]

    return tf.keras.Model(inputs=[vgg_input], outputs=outputs)


class StyleContentExtractor(tf.keras.models.Model):
    def __init__(self):
        super(StyleContentExtractor, self).__init__()
        self.vgg19_backbone = get_vgg19()
        self.vgg19_backbone.trainable = False
        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']
        self.content_layers = ['block5_conv2']

    def call(self, image):
        image = image * 255.0
        model_inputs = tf.keras.applications.vgg19.preprocess_input(image)
        model_outputs = self.vgg19_backbone(model_inputs)

        style_outputs = model_outputs[:5]
        content_outputs = model_outputs[5:]

        extracted_style = {layer_name: cal_gram_matrix(
            tensor) for layer_name, tensor in zip(self.style_layers, style_outputs)}
        extracted_content = {layer_name: tensor for layer_name,
                             tensor in zip(self.content_layers, content_outputs)}

        return {
            'content': extracted_content,
            'style': extracted_style
        }


def intialize_models_target(content_image, style_image, config):
    seed_python(42)
    seed_tf(42)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # content_image = tf.image.resize(content_image, config['image_size'])
    # style_image = tf.image.resize(style_image, config['image_size'])

    content_image_variable = tf.Variable(content_image)
    extractor = StyleContentExtractor()
    target_content = extractor(content_image)['content']
    target_style = extractor(style_image)['style']

    return content_image_variable, target_content, target_style, extractor, optimizer


def tf_loadimg(filepath, max_image_dim=512):
    """
    We will not be normalizing the image as we will be usinng vgg19.preprocess function
    """
    bin_img = tf.io.read_file(filepath)
    image_tensor = tf.io.decode_jpeg(bin_img, channels=3)
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    max_dim = tf.reduce_max(image_tensor.shape)
    scale = tf.cast(max_image_dim/max_dim, tf.float32)
    new_size = scale*tf.cast(image_tensor.shape[:-1], tf.float32)
    new_size = tf.cast(tf.round(new_size), tf.int32)

    image_tensor = tf.image.resize(image_tensor, new_size)
    return tf.expand_dims(image_tensor, 0)


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


if __name__ == "__main__":
    config = {
        "total_variation_weight": 30,
        "style_weight": 1e-2,
        "content_weight": 1e4,
        "image_size": (256, 256)
    }

    content_images = [
        "sample_data\content_pictures\content1.jpg",
        "sample_data\content_pictures\content2.jpg",
        "sample_data\content_pictures\content3.jpg",
    ]
    style_images = [
        "sample_data\style_pictures\style1.jpg",
        "sample_data\style_pictures\style2.jpg",
        "sample_data\style_pictures\style3.jpg",
    ]

    for i, content_img in enumerate(content_images):
        for j, style_img in enumerate(style_images):

            dir = os.path.join("generated_images", f"content_{i}_ style_{j}")
            os.makedirs(dir, exist_ok=True)

            content_image = tf_loadimg(content_img)
            style_image = tf_loadimg(style_img)

            content_image_variable, target_content, target_style, extractor, optimizer = intialize_models_target(
                content_image, style_image, config)

            epochs = 1000 
            for epoch in range(epochs):
                print(f"Training step {epoch+1} / {epochs}")
                train_step(content_image_variable, target_content,
                           target_style, extractor, optimizer, config)

                if (epoch+1) % 25 == 0:
                    print("Saving")
                    trained_image = tensor_to_image(content_image_variable)
                    image_path = os.path.join(dir, f"epoch_{epoch+1}.jpg")
                    trained_image.save(image_path)

            trained_image = tensor_to_image(content_image_variable)
            image_path = os.path.join(dir, f"final.jpg")
            trained_image.save(image_path)
