# Todo Convert Output To JSON
# Integrate with Main Inception

import argparse
import time
from time import gmtime, strftime
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument( '--image', help='File path of the image.', required=True)
  args = parser.parse_args()

  output_name = '/opt/demo/images/objectdetection_{0}.jpg'.format(strftime("%Y%m%d%H%M%S",gmtime()))

  # Initialize engine.
  engine = DetectionEngine('/opt/demo/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')

  # Open image.
  img = Image.open(args.image)
  draw = ImageDraw.Draw(img)

  # Run inference.
  ans = engine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True,
                               relative_coord=False, top_k=10)

  # Display result.
  if ans:
    for obj in ans:
      print ('score = ', obj.score)
      box = obj.bounding_box.flatten().tolist()
      print ('box = ', box)
      draw.rectangle(box, outline='red')
    img.save(output_name)
  else:
    print ('No object detected!')

if __name__ == '__main__':
  main()
