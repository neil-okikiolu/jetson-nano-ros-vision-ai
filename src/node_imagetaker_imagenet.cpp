/**
* This node takes images from the jetson camera (using a ros driver node)
* and saves it to disk, for now...
********************** IMPORTANT **********************
READ https://github.com/dusty-nv/jetson-inference for notes on Model performance
and image size parameters, etc
********************** IMPORTANT **********************
*/


#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>

#include <vision_msgs/Classification2D.h>
#include <vision_msgs/VisionInfo.h>

#include <jetson-inference/imageNet.h>

#include <jetson-utils/cudaFont.h>
#include <jetson-utils/cudaMappedMemory.h>

#include "image_converter.h"
#include "image_ops.h"

#include <unordered_map>

using namespace cv;
using namespace std;

// globals
imageNet* net = NULL;
imageConverter* cvt = NULL;

ros::Publisher* result_pub    = NULL;

// input image subscriber callback
void img_callback( const sensor_msgs::ImageConstPtr& input )
{
    ROS_INFO ("Received Image");
    
    // convert sensor_msgs[rgb] to opencv[brg]
    cv_bridge::CvImagePtr cv_ptr;
    cv_bridge::CvImagePtr cv_ptr_flip; // pointer for flipped image
    try
    {
        cv_ptr = cv_bridge::toCvCopy(
            input,
            sensor_msgs::image_encodings::BGR8
        );
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
    
    // classify the image
    float confidence = 0.0f;
    const int img_class = net->Classify(
        cvt->ImageGPU(),
        cvt->GetWidth(),
        cvt->GetHeight(),
        &confidence
    );
    
    
    // verify the output
    if( img_class >= 0 )
    {
        ROS_INFO(
            "classified image, %2.5f%% %s (class=%i)",
            confidence * 100.0f,
            net->GetClassDesc(img_class),
            img_class
        );
        
        
        // use font to draw the class description
        cudaFont* font = cudaFont::Create(adaptFontSize(flippedImage->width));

        if( font != NULL )
        {
            char overlay_str[512];
            sprintf(
                overlay_str,
                "%2.3f%% %s",
                confidence * 100.0f,
                net->GetClassDesc(img_class)
            );

            font->OverlayText(
                (float4*)cvt->ImageGPU(),
                cvt->GetWidth(),
                cvt->GetHeight(),
                (const char*)overlay_str,
                10,
                10,
                make_float4(255, 255, 255, 255),
                make_float4(0, 0, 0, 100)
            );
        }

        // wait for GPU to complete work
        CUDA(cudaDeviceSynchronize());
    }
    else
    {
        // an error occurred if the output class is < 0
        ROS_ERROR(
            "failed to classify %ux%u image",
            flippedImage->width,
            flippedImage->height
        );
    }
    
    
    // populate the message
    sensor_msgs::Image msg;

    if( !cvt->Convert(msg, sensor_msgs::image_encodings::BGR8) ) {
        return;
    }
    
    // save the output image to png
    sensor_msgs::ImagePtr out_img_ptr;
    cv_bridge::CvImagePtr out_cv_img_ptr;
    try
    {
        out_img_ptr = boost::make_shared<sensor_msgs::Image>(msg);

        out_cv_img_ptr = cv_bridge::toCvCopy(
            out_img_ptr,
            sensor_msgs::image_encodings::BGR8
        );

        // publish image result
        result_pub->publish(msg);
        // std::string base_path = "/home/dlinano/Test_Data/CameraCaptures/";
        // std::string suffix = "_classification";
        // save the image with the bounding box
        // imgops::saveImage(&out_cv_img_ptr->image, &base_path, &suffix);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}



int main (int argc, char **argv) {
    ros::init(argc, argv, "imagetaker_imagenet");
    
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    // can change this to another model you have
    // Below are some models we have downloaded
    std::string model_name = "googlenet";
    // std::string model_name = "resnet-18";
    
    // use this so we can pass parameters via command line e.g
    // rosrun jetsoncam imagetaker_imagenet _model_name:=googlenet
    private_nh.param<std::string>(
        "model_name",
        model_name,
        "googlenet"
    );
    
    // determine which built-in model was requested
    imageNet::NetworkType model = imageNet::NetworkTypeFromStr(
        model_name.c_str()
    );

    if(model == imageNet::CUSTOM)
    {
        ROS_ERROR(
            "invalid built-in pretrained model name '%s', defaulting to googlenet",
            model_name.c_str()
        );
        model = imageNet::GOOGLENET;
    }

    // create network using the built-in model
    net = imageNet::Create(model);
    
    if(!net)
    {
        ROS_ERROR("failed to load imageNet model");
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
        class_descriptions.push_back(net->GetClassDesc(n));
    }

    /*
    * create image converters
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
    ros::Publisher result_publsh = private_nh.advertise<sensor_msgs::Image>("image_classifications", 2);
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
    ROS_INFO("imagenet node initialized, waiting for messages");

    ros::spin();

    return 0;
}