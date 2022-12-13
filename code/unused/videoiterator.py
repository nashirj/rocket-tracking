from timeit import default_timer as timer

import cv2 as cv
import numpy as np

import detect


def run_detection_model_on_video(path, detect_fn=None, category_index=None, show_time=True):
    t0 = timer()
    cap = cv.VideoCapture(path)
    t1 = timer()

    print(f"took {t1-t0} to load vid")

    while cap.isOpened():
        ret, image = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # TODO: Make this resize to the correct expected dimension for the model
        new_shape = (640, 640)
        image = cv.resize(image, new_shape, interpolation=cv.INTER_AREA)

        t0 = timer()
        if detect_fn:
            image_np, input_tensor = inference.convert_img_to_tensor(image)
            detections = detect_fn(input_tensor)[0]
            print(detections)
            image_w_annot = inference.annotate_img_w_preds(image_np, detections,
                category_index, min_thresh=0.01)

            cv.imshow('frame', image_w_annot)
        
            if cv.waitKey(1) == ord('q'):
                break
        else:
            cv.imshow('frame', image)

            if cv.waitKey(25) == ord('q'):
                break
        t1 = timer()
        print(f"processing time: {t1-t0}")

    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    video_path = '../data/spacex-rocket-trimmed.mp4'
    
    model = inference.load_model(
        model_dir="../od_models/rocket_model/",
        pipeline_config="ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config",
        ckpt_name="ckpt-1")
    detect_fn = inference.get_model_detection_function(model)
    
    # This is for coco
    # category_index = inference.get_category_index()
    category_index = {1: {'id': 1, 'name': 'rocket'}}

    run_detection_model_on_video(video_path, detect_fn, category_index)

