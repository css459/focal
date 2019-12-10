#
# Cloud for ML Final Project
# Cole Smith
# load_model.py
#

import base64
import os
import uuid

from model.alexnet import load
from model.alexnet import predict as predict_alexnet
from model.focus_blur import focus_blur


# -----------------------------------------------------
# -- Inference
# -----------------------------------------------------

def predict(img):
    """
    Returns in JSON format the class, probability of class, the
    blurred output image, the mask of the image, blurred mask of image

    @param img:
    @return:
    """
    # Write down to tmp file
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("tmp", filename)
    img.save(filepath)

    # Infer class
    (cls, conf, idx) = sorted(predict_alexnet(img, load()), reverse=True)[0]

    # Write to output file
    output = os.path.join("output", filename)
    mask_output = os.path.join("output", 'mask_' + filename)
    saliency_output = os.path.join("output", 'sal_' + filename)

    focus_blur(filepath, imagenet_class_number=idx, output_file=output,
               mask_output_file=mask_output, saliency_map_output_file=saliency_output)

    # Output file to base64
    with open(output, "rb") as fp:
        encoded_img = base64.b64encode(fp.read())

    # Output file to base64
    with open(mask_output, "rb") as fp:
        encoded_img_mask = base64.b64encode(fp.read())

    # Output file to base64
    with open(saliency_output, "rb") as fp:
        encoded_img_saliency = base64.b64encode(fp.read())

    # Remove tmp and output files
    os.remove(filepath)
    os.remove(output)
    os.remove(mask_output)
    os.remove(saliency_output)

    response = {'class': str(cls),
                'probability': float(conf),
                'blurredImage': encoded_img.decode('utf-8'),
                'maskImage': encoded_img_mask.decode('utf-8'),
                'unblurredMask': encoded_img_saliency.decode('utf-8')}

    return response
