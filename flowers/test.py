import cv2 as cv
import argparse
import numpy as np
import sys

np.set_printoptions(precision=3)

def get_args():
        
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV)
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD)

    parser = argparse.ArgumentParser(description='Use this script to run classification deep learning networks using OpenCV.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--model', required=True,
                        help='Path to a binary file of model contains trained weights. '
                            'It could be a file with extensions .caffemodel (Caffe), '
                            '.pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet)')
    parser.add_argument('--config',
                        help='Path to a text file of model contains network configuration. '
                            'It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet)')
    parser.add_argument('--framework', choices=['caffe', 'tensorflow', 'torch', 'darknet'],
                        help='Optional name of an origin framework of the model. '
                            'Detect it automatically if it does not set.')
    parser.add_argument('--classes', help='Optional path to a text file with names of classes.')
    parser.add_argument('--mean', nargs='+', type=float, default=[0, 0, 0],
                        help='Preprocess input image by subtracting mean values. '
                            'Mean values should be in BGR order.')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Preprocess input image by multiplying on a scale factor.')
    parser.add_argument('--width', type=int, required=True,
                        help='Preprocess input image by resizing to a specific width.')
    parser.add_argument('--height', type=int, required=True,
                        help='Preprocess input image by resizing to a specific height.')
    parser.add_argument('--rgb', action='store_true',
                        help='Indicate that model works with RGB input images instead BGR ones.')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                            "%d: automatically (by default), "
                            "%d: Halide language (http://halide-lang.org/), "
                            "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "%d: OpenCV implementation" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                            '%d: CPU target (by default), '
                            '%d: OpenCL, '
                            '%d: OpenCL fp16 (half-float precision), '
                            '%d: VPU' % targets)
    args = parser.parse_args()
    return args

def test(args, net, classes, image):
    frame = cv.imread(image)
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, args.scale, (args.width, args.height), args.mean, args.rgb, crop=False)

    # Run a model
    net.setInput(blob)
    out = net.forward()

    # Get a class with a highest score.
    out = out.flatten()
    classId = np.argmax(out)
    confidence = out[classId]

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    print(label)
    # Print predicted class.
    label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
    # cv.putText(frame, label, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    print(label)
    # cv.imwrite('result.jpg', frame)
    # cv.imshow(winName, frame)
    return classId, confidence

def test_batch(args, net, classes):
    cls_num = len(classes)
    result = np.zeros((cls_num+3, cls_num+3))
    with open(args.input, 'r') as fi:
        for line in fi:
            arr = line.strip().split(' ')
            if len(arr) < 2:
                continue
            cls_id, conf = test(args, net, classes, arr[0])
            label = int(arr[1])
            result[label][-3] += 1
            result[label][cls_id] += 1
            result[-3][cls_id] += 1
            if cls_id == int(arr[1]):
                result[label][-2] += 1
                result[-2][label] += 1

    for i in range(cls_num):
        if result[i][-3] > 0:
            result[i][-1] = result[i][-2] / result[i][-3]
        if result[-3][i] > 0:
            result[-1][i] = result[-2][i] / result[-3][i]
        result[-1][-3] += result[i][-3]
        result[-1][-2] += result[i][-2]
    result[-1][-1] = result[-1][-2] / result[-1][-3]
    print(result)




def read_label_name(classes_file):
    classes = None
    with open(classes_file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

def main(args):
    # args = get_args()
    # Load names of classes
    classes = None
    if args.classes:
        classes = read_label_name(args.classes)

    # Load a network
    net = cv.dnn.readNet(args.model, args.config, args.framework)
    net.setPreferableBackend(args.backend)
    net.setPreferableTarget(args.target)
    if args.input.endswith('.txt'):
        test_batch(args, net, classes)
    elif os.path.isdir(args.input):
        pass
    else:
        test(args, net, classes, args.input)

if __name__ == '__main__':
    main(get_args())