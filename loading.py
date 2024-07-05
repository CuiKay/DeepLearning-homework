
from common.params import args, args_numcode, args_seg
from seg_system import Segment
from ppocr.infer import predict_rec, predict_cls
from ocr_system_base import OCR, load_model
import ppocr.infer.predict_det as predict_det



rec = predict_rec.TextRecognizer(args_numcode)

e2e_algorithm, text_sys = load_model(args, e2e_algorithm = False)

text_detection = predict_det.TextDetector(args)
segment = Segment(args_seg)