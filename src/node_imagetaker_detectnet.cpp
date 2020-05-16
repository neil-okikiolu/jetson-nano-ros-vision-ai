/**
* This node takes images from the jetson camera (using a ros driver node)
* and saves it to disk, for now...
*/

#include <iostream>
#include <ctime>
#include <sstream>
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>

#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>

#include <jetson-inference/detectNet.h>
#include <jetson-utils/cudaMappedMemory.h>

#include <jetson-utils/loadImage.h>

#include "image_converter.h"

#include <unordered_map>

using namespace cv;
using namespace std;

// globals
detectNet* net = NULL;
imageConverter* cvt = NULL;

ros::Publisher* result_pub    = NULL;

void saveImage(cv::Mat *cvImage) {
    // create date time string for captures
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H-%M-%S");
    auto str = oss.str();
    
    // combine date time string with path for final image path
    std::string base_path = "/home/dlinano/Test_Data/CameraCaptures/";
    std::string out_string = base_path + "capture_" + str + ".png";
   
    bool result = false;

    try
    {
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        // try saving the cv::Mat as a png
        result = imwrite(out_string, *cvImage, compression_params);
    }
    catch (const cv::Exception& ex)
    {
        ROS_ERROR(
            "Exception converting image to PNG format: %s\n",
            ex.what()
        );
    }

    if (result) {
        ROS_INFO(
            "Saved %s",
            out_string.c_str()
        );
    }
    else {
        ROS_ERROR("ERROR: Can't save PNG file.");
        return;
    }
}

// input image subscriber callback
void img_callback( const sensor_msgs::ImageConstPtr& input )
{
    ROS_INFO ("Received Image");
    
    // convert sensor_msgs[rgb] to opencv[brg]
    cv_bridge::CvImagePtr cv_ptr;
    cv_bridge::CvImagePtr cv_ptr_flip; // pointer for flipped image
    try
    {
      cv_ptr = cv_bridge::toCvCopy(input, sensor_msgs::image_encodings::BGR8);
      cv_ptr_flip = cv_bridge::toCvCopy(input, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    
    // we are doing a 180 deg flip since
    // my camera is upside down
    const int img_flip_mode_ = -1;
    // flip the image
    cv::flip(cv_ptr->image, cv_ptr_flip->image, img_flip_mode_);
    
    // convert converted image back to a sensor_msgs::ImagePtr
    // for use with nvidia / other ML algorithms
    sensor_msgs::ImagePtr flippedImage = cv_ptr_flip->toImageMsg();

    // convert the image TO reside on GPU
    // the converting TO and converting FROM are the SAME funtion name
    // with different signatures
    if( !cvt || !cvt->Convert(flippedImage) )
    {
        ROS_ERROR (
            "failed to convert %ux%u %s image",
            flippedImage->width,
            flippedImage->height,
            flippedImage->encoding.c_str()
        );
        return;
    }
    else {
        ROS_INFO (
            "Converted %ux%u %s image",
            flippedImage->width,
            flippedImage->height,
            flippedImage->encoding.c_str()
        );
    }
    
    // classify the image
    detectNet::Detection* detections = NULL;

    /*
    enum OverlayFlags
    {
        OVERLAY_NONE       = 0, // < No overlay.
        OVERLAY_BOX        = (1 << 0),  //< Overlay the object bounding boxes
        OVERLAY_LABEL      = (1 << 1), //< Overlay the class description labels
        OVERLAY_CONFIDENCE = (1 << 2), //< Overlay the detection confidence values
    };
    */
    // detect AND render the all possible info i.e
    // bounding box, label, confidence
    const uint32_t flags = detectNet::OVERLAY_BOX | detectNet::OVERLAY_LABEL | detectNet::OVERLAY_CONFIDENCE;
    const int numDetections = net->Detect(
        cvt->ImageGPU(),
        cvt->GetWidth(),
        cvt->GetHeight(),
        &detections,
        flags
    );

    // if an error occured
    if( numDetections < 0 )
    {
        ROS_ERROR(
            "failed to run object detection on %ux%u image",
            flippedImage->width,
            flippedImage->height
        );
        return;
    }
    
    /*
    // if no detections
    if( numDetections == 0 )
    {
        ROS_INFO(
            "detected %i objects in %ux%u image",
            numDetections,
            flippedImage->width,
            flippedImage->height
        );
        return;
    }
    */
    
    
    // if objects were detected, send out message
    if( numDetections >= 0 )
    {
        ROS_INFO(
            "detected %i objects in %ux%u image",
            numDetections,
            flippedImage->width,
            flippedImage->height
        );
        
        
        // get our image with overlays back from the GPU
        // the converting TO and converting FROM are the SAME funtion name
        // with different signatures
        sensor_msgs::Image outputImage;
        // outputImage.header.stamp = ros::Time::now();
        if(!cvt || !cvt->Convert(outputImage, sensor_msgs::image_encodings::BGR8)) {
            ROS_ERROR ("failed to convert outputImage image back");
            return;
        }
        else {
            ROS_INFO (
                "Converted %ux%u %s image back",
                outputImage.width,
                outputImage.height,
                outputImage.encoding.c_str()
            );
        }
        
        // save the output image to png
        sensor_msgs::ImagePtr out_img_ptr;
        cv_bridge::CvImagePtr out_cv_img_ptr;
        try
        {
            out_img_ptr = boost::make_shared<sensor_msgs::Image>(outputImage);

            out_cv_img_ptr = cv_bridge::toCvCopy(
                out_img_ptr,
                sensor_msgs::image_encodings::BGR8
            );

            // publish image result
            result_pub->publish(outputImage);
            // save the image with the bounding box
            // saveImage(&out_cv_img_ptr->image);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}

int main (int argc, char **argv) {
    ros::init(argc, argv, "imagetaker_detectnet");
    
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    std::string class_labels_path;
    std::string prototxt_path;
    std::string model_path;
    // can change this to another model you have
    // e.g rosrun jetsoncam imagetaker_detectnet _model_name:=pednet
    // "ssd-mobilenet-v2" is better than "pednet"!
    std::string model_name = "ssd-mobilenet-v2";
    float mean_pixel = 0.0f;
    // If the desired objects aren't being detected in the video feed or
    // you're getting spurious detections, try decreasing or increasing
    // the detection threshold (default value is 0.5)
    float threshold  = 0.5f;
    
    private_nh.param<std::string>(
        "model_name",
        model_name,
        "ssd-mobilenet-v2"
    );
    
    // determine which built-in model was requested
    detectNet::NetworkType model = detectNet::NetworkTypeFromStr(model_name.c_str());

    if( model == detectNet::CUSTOM )
    {
        ROS_ERROR(
            "invalid built-in pretrained model name '%s', defaulting to pednet", 
            model_name.c_str()
        );
        model = detectNet::PEDNET;
    }

    // create network using the built-in model
    net = detectNet::Create(model);
    
    if( !net )
    {
        ROS_ERROR("failed to load detectNet model");
        return 0;
    }
    
    
    /*
    * create the class labels parameter vector
    */
    std::hash<std::string> model_hasher;  // hash the model path to avoid collisions on the param server
    std::string model_hash_str = std::string(
        net->GetModelPath()) + std::string(net->GetClassPath()
    );
    const size_t model_hash = model_hasher(model_hash_str);

    ROS_INFO("model hash => %zu", model_hash);
    ROS_INFO("hash string => %s", model_hash_str.c_str());

    // obtain the list of class descriptions
    std::vector<std::string> class_descriptions;
    const uint32_t num_classes = net->GetNumClasses();

    for( uint32_t n=0; n < num_classes; n++ ) {
        class_descriptions.push_back(net->GetClassDesc(n));
    }

    // create the key on the param server
    std::string class_key = std::string("class_labels_") + std::to_string(model_hash);
    // private_nh.setParam(class_key, class_descriptions);

    // populate the vision info msg
    std::string node_namespace = private_nh.getNamespace();
    ROS_INFO("node namespace => %s", node_namespace.c_str());
    
    /*
    * create an image converter object
    */
    cvt = new imageConverter();

    if(!cvt)
    {
        ROS_ERROR("failed to create imageConverter object");
        return 0;
    }
    
    
    /*
    * advertise publisher topics
    */
    ros::Publisher result_publsh = private_nh.advertise<sensor_msgs::Image>("image_detections", 2);
    result_pub = &result_publsh;
    
    /*
    * subscribe to image topic
    */
    ros::Subscriber img_sub = nh.subscribe(
        "/csi_cam_0/image_raw",
        5,
        img_callback
    );

    /*
    * wait for messages
    */
    ROS_INFO("imagetaker node initialized, waiting for messages");

    ros::spin();

    return 0;
    
}