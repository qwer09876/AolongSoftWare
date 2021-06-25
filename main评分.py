import os
import cv2
import argparse

from deepsort import DeepSort


class VideoCapture(object):

    def __init__(self, img_path):
        self.name = img_path
        self.base = 'D:/pythonProject3/MOT20/images/test/{}'
        self.img_path = self.base.format(img_path)
        self.num = len(os.listdir(self.img_path))
        self.count = 0

    def read(self):
        self.count += 1
        img = os.path.join(self.img_path, '{:05}.jpg'.format(self.count))
        image = cv2.imread(img)
        return not image is None, image


def MOTresult(filename):
    font = cv2.FONT_HERSHEY_SIMPLEX
    deepsort = DeepSort(
        det_model_dir=args.det_model_dir,
        emb_model_dir=args.emb_model_dir,
        use_gpu=args.use_gpu,
        run_mode='fluid',
        threshold=args.threshold,
        max_cosine_distance=args.max_cosine_distance,
        nn_budget=args.nn_budget,
        max_iou_distance=args.max_iou_distance,
        max_age=args.max_age,
        n_init=args.n_init
    )
    cap = VideoCapture(str(filename))
    bboxes = []
    name = str(int(cap.name) + 1)
    # if args.save_dir:
    #     if not os.path.exists(args.save_dir):
    #         os.mkdir(args.save_dir)
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #     h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #     print(fps, w, h)
    #     save_video_path = os.path.join(args.save_dir, 'output.avi')
    #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #     writer = cv2.VideoWriter(save_video_path, fourcc, fps, (int(w), int(h)))

    for fid in range(cap.num):
        success, frame = cap.read()
        if not success:
            break

        num = 1
        outputs = deepsort.update(frame)
        if outputs is not None:
            for output in outputs:
                conf_ = 1
                bboxes.append([fid + 1, num, output[0], output[1], output[2] - output[0],
                               output[3] - output[1], conf_, -1, -1, -1])
                # print([fid + 1, num, output[0], output[1], output[2],
                #                output[3], conf_, -1, -1, -1])

                num += 1
                cv2.rectangle(frame, (output[0], output[1]), (output[2], output[3]), (0, 0, 255), 2)
                cv2.putText(frame, str(output[-1]), (output[0], output[1]), font, 1.2, (255, 255, 255), 2)
                cv2.imshow("img", frame)
                # cv2.imwrite(str(name)+"test"+str(num)+".png",frame)
                cv2.waitKey(1)


    with open("SoftwareCup-0" + name + '.txt', 'w') as f:
        for box in bboxes:
            line = ''
            for v in box:
                line += ',{}'.format(v)
            line = line[1:] + '\n'
            f.write(line)


def main(args):
    for i in range(0, 4):
        MOTresult(i)

    # if args.save_dir:
    #     writer.write(frame)
    # if args.display:
    #     cv2.imshow('test', frame)
    #     k = cv2.waitKey(1)
    #     if k==27:
    #         cap.release()
    #         break

    # if args.save_dir:
    #     writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='''you can set the video_path or camera_id to start the program, 
        and also can set the display or save_dir to display the results or save the output video.''',
        description="this is the help of this script."
    )

    parser.add_argument("--det_model_dir", type=str, default='model/detection', help="the detection model dir.")
    parser.add_argument("--emb_model_dir", type=str, default='model/embedding', help="the embedding model dir.")
    parser.add_argument("--run_mode", type=str, default='fluid', help="the run mode of detection model.")
    parser.add_argument("--use_gpu", action="store_true", help="do you want to use gpu.")

    parser.add_argument("--threshold", type=float, default=0.1, help="the threshold of detection model.")
    parser.add_argument("--max_cosine_distance", type=float, default=0.2, help="the max cosine distance.")
    parser.add_argument("--nn_budget", type=int, default=100, help="the nn budget.")
    parser.add_argument("--max_iou_distance", type=float, default=0.7, help="the max iou distance.")
    parser.add_argument("--max_age", type=int, default=70, help="the max age.")
    parser.add_argument("--n_init", type=int, default=3, help="the number of init.")

    parser.add_argument("--video_path", type=str, default=None, help="the input video path or the camera id.")
    parser.add_argument("--camera_id", type=int, default=0, help="do you want to use the camera and set the camera id.")
    parser.add_argument("--display", action="store_true", help="do you want to display the results.")
    parser.add_argument("--save_dir", type=str, default=None, help="the save dir for the output video.")

    args = parser.parse_args()
    main(args)
