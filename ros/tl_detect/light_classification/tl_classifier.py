from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np

import rospy
import yaml
import datetime

# this modul was accomplished in team work; while my model wasn't performing well,
# I decided to use the trained one from my team co-worker https://github.com/YimengZhu/CarND-Capstone
# He used more ressources (GPUs) than me, and so the classification is much better
# if requested, i always can show with my trained models
# basically i followed https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# and used as my team member ssd_mobilenet_v1_coco

class TLClassifier(object):
    def __init__(self):

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.safe_load(config_string)

        model_name = 'frozen_inference_graph.pb'

        if self.config['is_site']:
            model_path = 'light_classification/model/site/' + model_name
        else:
            model_path = 'light_classification/model/sim/' + model_name

        self.frozen_graph = tf.Graph()

        with self.frozen_graph.as_default():
            graph_defintion = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                graph_defintion.ParseFromString(fid.read())
                tf.import_graph_def(graph_defintion, name='')

            self.graph_num_detections = self.frozen_graph.get_tensor_by_name('num_detections:0')
            self.graph_image_tensor = self.frozen_graph.get_tensor_by_name('image_tensor:0')
            self.graph_boxes = self.frozen_graph.get_tensor_by_name('detection_boxes:0')
            self.graph_scores = self.frozen_graph.get_tensor_by_name('detection_scores:0')
            self.graph_classes = self.frozen_graph.get_tensor_by_name('detection_classes:0')

        self.sess = tf.Session(graph=self.frozen_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: traffic light color ID (specified in styx_msgs/TrafficLight)
        """
        with self.frozen_graph.as_default():

            input_image = np.expand_dims(image, axis=0)

            predict = self.sess.run([self.graph_boxes, self.graph_scores, self.graph_classes, self.graph_num_detections],
                                         feed_dict={self.graph_image_tensor: input_image})

        probility = np.squeeze(predict[1])
        predicted_classes = np.squeeze(predict[2]).astype(np.int32)

        if probility[0] > .6:
            print(probility[0])
            print(predicted_classes[0])
            if predicted_classes[0] == 1:
                return TrafficLight.GREEN

            elif predicted_classes[0] == 2:
                return TrafficLight.RED

            elif predicted_classes[0] == 3:
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN