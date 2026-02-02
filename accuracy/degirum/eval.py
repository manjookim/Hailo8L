from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 1. COCO 유효성 검증(Ground Truth) 정보 로드
coco_gt = COCO('labels_coco.json')

# 2. 모델 추론 결과(Prediction) 정보 로드
coco_dt = coco_gt.loadRes('predictions.json')

# 3. COCO 평가 객체 생성
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

# 4. 평가 실행
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
