/**
* This node takes images from the jetson camera (using a ros driver node)
* and saves it to disk, for now...
********************** IMPORTANT **********************
READ https://github.com/dusty-nv/jetson-inference for notes on Model performance
and image size parameters, etc
********************** IMPORTANT **********************

********************** IN-PROGRESS ********************
NEED 2 Camera Images at a time to
construct a 3D representation
********************** IN-PROGRESS ********************
*/


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

#include <jetson-inference/homographyNet.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/cudaWarp.h>
#include <jetson-utils/mat33.h>

#include <jetson-utils/loadImage.h>

#include "image_converter.h"
#include "image_ops.h"

#include <unordered_map>

using namespace cv;
using namespace std;

// globals
homographyNet* net = NULL;

imageConverter* cvt = NULL;

ros::Publisher* homography_pub = NULL;

void save_msg_to_image (sensor_msgs::Image *img, std::string *suffix) {
    // save the output image to png
    sensor_msgs::ImagePtr out_img_ptr;
    cv_bridge::CvImagePtr out_cv_img_ptr;
    try
    {
        out_img_ptr = boost::make_shared<sensor_msgs::Image>(*img);

        out_cv_img_ptr = cv_bridge::toCvCopy(
            out_img_ptr,
            sensor_msgs::image_encodings::BGR8
        );

        std::string base_path = "/home/dlinano/Test_Data/CameraCaptures/";
        // save the image with the bounding box
        imgops::saveImage(&out_cv_img_ptr->image, &base_path, suffix);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}


// publish overlay image
bool save_homography( uint32_t width, uint32_t height )
{
    ROS_INFO ("Starting to save image overlay");
    // assure correct image size
    if( !overlay_cvt->Resize(width, height) ) {
        return false;
    }

    // generate the overlay
    if( !net->Overlay(overlay_cvt->ImageGPU(), width, height, overlay_filter) ) {
        return false;
    }

    // populate the message
    sensor_msgs::Image msg;

    if( !overlay_cvt->Convert(msg, sensor_msgs::image_encodings::BGR8) ) {
        return false;
    }
    
    // publish image result
    homography_pub->publish(msg);

    // save the image result
    //std::string suffix = "_homography";
    //save_msg_to_image(&msg, &suffix);
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
        cv_ptr_flip = cv_bridge::toCvCopy(
            input,
            sensor_msgs::image_encodings::BGR8
        );
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
    
    const bool processed = net->Process(
        cvt->ImageGPU(),
        cvt->GetWidth(),
        cvt->GetHeight()
    );

    // process the segmentation network
    if (!processed)
    {
        ROS_ERROR(
            "failed to process segmentation on %ux%u image",
            flippedImage->width,
            flippedImage->height
        );
        return;
    }

    // color overlay
    save_homography(input->width, input->height);
}



int main (int argc, char **argv) {
    ros::init(argc, argv, "imagetaker_homographynet");
    
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    // can change this to another model you have
    // Below are some models we have downloaded
    std::string model_name = "fcn-resnet18-cityscapes-512x256";
    // std::string model_name = "fcn-resnet18-cityscapes-1024x512";
    // std::string model_name = "fcn-resnet18-deepscene-576x320";
    // std::string model_name = "fcn-resnet18-mhp-512x320";
    // std::string model_name = "fcn-alexnet-pascal-voc";
    // std::string model_name = "fcn-resnet18-sun-512x400";
    
    // use this so we can pass parameters via command line e.g
    // rosrun jetsoncam imagetaker_segnet _model_name:=fcn-resnet18-cityscapes-512x256
    private_nh.param<std::string>(
        "model_name",
        model_name,
        "fcn-resnet18-cityscapes-512x256"
    );

    // retrieve filter mode settings
    std::string overlay_filter_str = "linear";
    std::string mask_filter_str    = "linear";
    
    overlay_filter = segNet::FilterModeFromStr(
        overlay_filter_str.c_str(),
        segNet::FILTER_LINEAR
    );
    mask_filter = segNet::FilterModeFromStr(
        mask_filter_str.c_str(),
        segNet::FILTER_LINEAR
    );
    
    // determine which built-in model was requested
    segNet::NetworkType model = segNet::NetworkTypeFromStr(model_name.c_str());

    if(model == segNet::SEGNET_CUSTOM)
    {
        ROS_ERROR(
            "invalid built-in pretrained model name '%s', defaulting to cityscapes",
            model_name.c_str()
        );
        model = segNet::FCN_ALEXNET_CITYSCAPES_HD;
    }

    // create network using the built-in model
    net = segNet::Create(model);
    
    if(!net)
    {
        ROS_ERROR("failed to load segNet model");
        return 0;
    }
    
    
    /*
     * create the class labels parameter vector
     */
    // hash the model path to avoid collisions on the param server
    std::hash<std::string> model_hasher;
    std::string model_hash_str =
        std::string(net->GetModelPath())+ std::string(net->GetClassPath());

    const size_t model_hash = model_hasher(model_hash_str);

    ROS_INFO("model hash => %zu", model_hash);
    ROS_INFO("hash string => %s", model_hash_str.c_str());

    // obtain the list of class descriptions
    std::vector<std::string> class_descriptions;
    const uint32_t num_classes = net->GetNumClasses();

    for( uint32_t n=0; n < num_classes; n++ )
    {
        const char* label = net->GetClassDesc(n);

        if( label != NULL ) {
            class_descriptions.push_back(label);
        }
    }

    /*
    * create image converters
    */
    cvt      = new imageConverter();
    overlay_cvt    = new imageConverter();
    mask_color_cvt = new imageConverter();
    mask_class_cvt = new imageConverter();

    if( !cvt || !overlay_cvt || !mask_color_cvt || !mask_class_cvt )
    {
        ROS_ERROR("failed to create imageConverter objects");
        return 0;
    }
    
    /*
    * advertise publisher topics
    */
    ros::Publisher overlay_publsh = private_nh.advertise<sensor_msgs::Image>("overlay", 2);
    overlay_pub = &overlay_publsh;
    
    ros::Publisher mask_color_publsh = private_nh.advertise<sensor_msgs::Image>("mask_color", 2);
    mask_color_pub = &mask_color_publsh;
    
    ros::Publisher mask_class_publsh = private_nh.advertise<sensor_msgs::Image>("mask_class", 2);
    mask_class_pub = &mask_class_publsh;


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
    ROS_INFO("segnet node initialized, waiting for messages");

    ros::spin();

    return 0;
}