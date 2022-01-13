from mmocr.utils.ocr import MMOCR

# Load models into memory
ocr = MMOCR(det='FCE_CTW_DCNv2', recog=None)

# Inference
results = ocr.readtext(
    'demo/demo_text_det.jpg', output='demo/det_out.jpg', export='demo/')

print(results)
