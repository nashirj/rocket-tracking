# Project overview

Link to YouTube demo of rocket object detection

[![Watch the video](https://img.youtube.com/vi/_ECUBA9DCYI/maxresdefault.jpg)](https://youtu.be/_ECUBA9DCYI)

See https://docs.google.com/document/d/1TMhAC7ZwyCgVrUk31nMRxyOAeYL_1aFDZaZbZZmNZ58/edit#

## Object detection

[Rocket launch video](https://www.youtube.com/watch?v=BfjFsI3BFPI), 1:16 - 2:12

## References
- https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d
- https://heartbeat.comet.ml/a-2019-guide-to-object-detection-9509987954c3
- https://pytorch.org/vision/stable/models.html
- https://github.com/pytorch/vision/tree/main/torchvision/models/detection

Title of the project: Real-time rocket object tracking/detection in video.

Problem definition: Apply object detection/tracking to a video of a rocket launch (e.g. https://www.youtube.com/watch?v=BfjFsI3BFPI). If I am able to successfully apply object detection/tracking to unoccluded video, I'll try applying object detection/tracking with occlusion.

What dataset will be used: I will curate two datasets; 1) only images of the SpaceX Falcon 9 rocket and 2) images of rockets in general (including SpaceX Falcon 9).

How will you solve this problem: I will fine-tune an object detection model pretrained on another dataset (e.g. https://github.com/pytorch/vision/tree/main/torchvision/models/detection) with images of rockets. Then, I will develop code to apply the trained model to video. I will evaluate the model using object detection metrics such as mAP and IoU.

What will be your contribution: I am working alone, so I will 1) curate the dataset(s), 2) fine tune an existing model to detect images of rockets, 3) apply the trained model to a video and evaluate performance using object detection metrics such as mean average precision (mAP).
