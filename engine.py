import base64
import cv2
from baml_py import ClientRegistry
from baml_client.sync_client import b
from baml_py import Image
import json
import os
import traceback, sys
from baml_py import Collector
import uuid


def init_cr():
    # Initialize baml client registry.
    cr = ClientRegistry()
    cr.set_primary("Gemini_2_0_pro")
    return cr


def to_json(results):
    """Save extracted results to a JSON file inside a unique folder for each image."""

    def serialize(obj):
        if hasattr(obj, "dict"):
            return obj.dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)
    return json.loads(json.dumps(results, indent=4, ensure_ascii=False, default=serialize))


def llm_extract_image(images_base64, client_registry):
    extraction_type = "EKYB"
    collector_llm_image = Collector(name="collector_llm_image")
    extract_functions = {"EKYB": b.ExtractInvoices}
    
    image_result = extract_functions[extraction_type](
        images_base64, {"client_registry": client_registry, "collector": collector_llm_image}
    )
    
    tokens = [collector_llm_image.usage.input_tokens, collector_llm_image.usage.output_tokens]
    return to_json(image_result), tokens


def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)  # Change to '.jpg' if needed
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return Image.from_base64("image/png", base64_string)


def llm_predict_images(images):
    cr = init_cr()
    images_base64 = [image_to_base64(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) for image in images]
    llm_results, tokens = llm_extract_image(images_base64=images_base64, client_registry=cr)
    return llm_results, tokens


def llm_predict(uuid, files_name, images=None):

    results, llm_results, tokens = [], None, [0, 0]
    try:
        llm_results, tokens = llm_predict_images(images)

        results.append({
            "file_name": os.path.basename(files_name),
            "extract_data": llm_results,
            "tokens": tokens
        })
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        tb_info = traceback.extract_tb(exc_tb)
        print("ID {} >>> ERROR LLM inference, Message Error: {}, exc_type: {}, exc_obj: {}, \
                        exc_tb: {}, tb_info: {}". format(str(uuid), str(e), exc_type, exc_obj, exc_tb, tb_info))
        results.append({
            "file_name": os.path.basename(files_name),
            "extract_data": None,
            "tokens": tokens
        })
    return results


if __name__ == "__main__":
    
    uid = uuid.uuid1()
    file_name = "images/3.png"
    image = cv2.imread(file_name)
    
    results = llm_predict(uuid=str(uid), files_name=file_name, images=[image])
    print("results >>> ", results)
    pass