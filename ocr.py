import torch
import cv2 as cv
import numpy as np
from typing import List

from craft_text_detector import Craft
from utils.file import read_media
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TextRecognizer:
    def __init__(
        self,
        detection_weights: str,
        refine_weights,
        ocr_weights: str,
        device: str = "cuda",
    ):
        self.device = device
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available")
            is_cuda = True
        else:
            is_cuda = False

        self.processor = TrOCRProcessor.from_pretrained(ocr_weights)
        self.recognizor = VisionEncoderDecoderModel.from_pretrained(ocr_weights)
        self.craft = Craft(
            text_threshold=0.7,
            link_threshold=0.4,
            refiner=True,
            crop_type="box",
            weight_path_craft_net=detection_weights,
            weight_path_refine_net=refine_weights,
            cuda=is_cuda,
        )

        if is_cuda:
            self.recognizor = self.recognizor.to(device)

    def merge_boxes(self, bbox: List, line_overlap_threshold=0.6) -> List:
        bbox = sorted(bbox, key=lambda x: x[1])
        merged_bbox = []
        for i in range(len(bbox)):
            x_min, y_min, x_max, y_max, text = bbox[i]
            if i == 0:
                merged_bbox.append([x_min, y_min, x_max, y_max, text])
            else:
                x_min_prev, y_min_prev, x_max_prev, y_max_prev, text_prev = merged_bbox[
                    -1
                ]
                y_intersect = (min(y_max, y_max_prev) - max(y_min, y_min_prev)) / (
                    max(y_max, y_max_prev) - min(y_min, y_min_prev)
                )
                if y_intersect > line_overlap_threshold:
                    if x_min < x_min_prev:
                        new_text = text + " " + text_prev
                    else:
                        new_text = text_prev + " " + text

                    new_x_min = min(x_min, x_min_prev)
                    new_y_min = min(y_min, y_min_prev)
                    new_x_max = max(x_max, x_max_prev)
                    new_y_max = max(y_max, y_max_prev)
                    merged_bbox[-1] = [
                        new_x_min,
                        new_y_min,
                        new_x_max,
                        new_y_max,
                        new_text,
                    ]
                else:
                    merged_bbox.append([x_min, y_min, x_max, y_max, text])
        return merged_bbox

    def resutls_to_json(self, results: List) -> List[dict]:
        json_output = []
        for result in results:
            json_output.append(
                {
                    "text": result[-1],
                    "bbox": result[:-1],
                }
            )
        return json_output

    def recognize(self, img_list: List, pad=0.1) -> List:
        print("Text detection")
        text_results = []
        for img in img_list:
            print(img.shape)
            results = self.craft.detect_text(img)
            result = np.array(results["boxes"])
            result = result.astype(int)
            # text_results.append([])
            bbox = []
            for res in result:
                x_min, y_min = res[0, :].tolist()
                x_max, y_max = res[2, :].tolist()
                h = y_max - y_min
                pad_size = int(h * pad)
                x_min = max(0, x_min - pad_size)
                y_min = max(0, y_min - pad_size)
                x_max = min(img.shape[1], x_max + pad_size)
                y_max = min(img.shape[0], y_max + pad_size)

                roi = img[y_min:y_max, x_min:x_max]
                pixel_values = self.processor(roi, return_tensors="pt").pixel_values
                if self.device == "cuda":
                    pixel_values = pixel_values.to(self.device)
                with torch.no_grad():
                    generated_ids = self.recognizor.generate(pixel_values)
                text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                bbox.append([x_min, y_min, x_max, y_max, text])
                # cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
            bbox = self.merge_boxes(bbox)
            bbox_json = self.resutls_to_json(bbox)
            text_results.append(bbox_json)
            # cv.imshow("img", img)
            # cv.waitKey(0)
        return text_results


def main():
    ocr_weights = "./weights/trocr-base-handwritten/"
    # ocr_weights = "./weights/thai-trocr/"
    craft_weight = "./weights/craft_mlt_25k.pth"
    refine_weight = "./weights/craft_refiner_CTW1500.pth"

    img_list = read_media("./data/images/test_00.png")
    recognizer = TextRecognizer(craft_weight, refine_weight, ocr_weights)
    text_results = recognizer.recognize(img_list)
    print(text_results)


if __name__ == "__main__":
    main()
