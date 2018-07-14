#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/openni2_grabber.h>
#include <pcl/common/time.h>
#include <pcl/features/integral_image_normal.h>
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>// use timer1->see boost tutorial:basic skill
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/exception/to_string.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/filters/extract_indices.h>


#include <Eigen/Dense>
#include <cmath>    
#include <math.h>
#include <string.h>
#include <cstring>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>

#include <boost/asio/serial_port.hpp>
#include <boost/asio.hpp>
#include "TimeoutSerial.h"
#include "TimeoutSerial.cpp"

#include <pcl/features/pfh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/mls.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/keypoints/iss_3d.h>

#include <stdio.h>
#include <sstream>
#include <stdlib.h> 

#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\video.hpp>
#include <opencv2\videostab\videostab.hpp>
#include <stdio.h>
#include "opencv2/opencv.hpp"

# include <opencv2/core/core.hpp>
# include <opencv2/features2d/features2d.hpp>
# include <opencv2/nonfree/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
//#include "wsc.h"
using namespace boost::accumulators;
using namespace std;
using namespace boost;
using namespace cv;

#define FPS_CALC(_WHAT_)\
do\
{\
    static unsigned count = 0;\
    static double last = pcl::getTime();\
    double now = pcl::getTime (); \
	++count;\
    if(now – last >= 1.0) \
			    {\
	std::cout << “Average framerate(“<< _WHAT_ << “): “ << double(count)/double(now – last) << ” Hz” <<std::endl;\
	count = 0;\
	last = now;\
			    }while(false) 

#define Zfilter 0.5 // filtering Z cordinate inside 0.5m

unsigned int text_id = 0;
#define PI 3.14159265
string Tdata;
string Rdata;

cv::Mat redOutput;
int Rlow = 0, Rhigh = 255, Glow = 0, Ghigh = 134, Blow = 140, Bhigh = 255;
TimeoutSerial port("COM1", 38400);
Eigen::Matrix3d in_intrinsic;
Eigen::Matrix3d in_R;//cam to board
Eigen::Vector3d t(-216.1163, -52.9724, 716.9573);//cam to board
//Eigen::Vector3d tt(-111.0900, -311.6000, 77.0900);//from board to arm
Eigen::Vector3d tt(-111.0900, -310.6000, 77.0900);
Eigen::Matrix3d A;//from board to arm
void serial_sendd(string command);
string serial_reieve();
int check_ACK();
pcl::PointCloud<pcl::PointXYZ>::Ptr Pass_cloud_filtered_X(new pcl::PointCloud<pcl::PointXYZ>);
int ch = 0;
class SimpleOpenNIProcessor
{
public:
	void cloud_cb_(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
	{

		oke = false;
		con = 0;
		pcl::PointCloud<pcl::PointXYZ>::Ptr Voxel_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr Pass_cloud_filtered_Z(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr Pass_cloud_filtered_Y(new pcl::PointCloud<pcl::PointXYZ>);

		//voxel filter
		pcl::VoxelGrid <pcl::PointXYZ> sor;

		//Create the filtering voxel object
		//sor.setInputCloud (cloud);
		//sor.setLeafSize (0.005f, 0.005f, 0.005f);//leaf size 1cm
		//sor.filter (*Voxel_cloud_filtered);

		//Passthrough filter
		pcl::PassThrough<pcl::PointXYZ> passZ;
		//passZ.setInputCloud (Voxel_cloud_filtered);
		passZ.setInputCloud(cloud);
		passZ.setFilterFieldName("z");
		passZ.setFilterLimits(0.0, 0.67);//0.67
		//pass.setFilterLimitsNegative (true);
		passZ.filter(*Pass_cloud_filtered_Z);

		pcl::PassThrough<pcl::PointXYZ> passY;
		passY.setInputCloud(Pass_cloud_filtered_Z);
		passY.setFilterFieldName("y");
		passY.setFilterLimits(-0.15, 0.25);
		passY.filter(*Pass_cloud_filtered_Y);

		pcl::PassThrough<pcl::PointXYZ> passX;
		passX.setInputCloud(Pass_cloud_filtered_Y);
		passX.setFilterFieldName("x");
		passX.setFilterLimits(-0.3, 0.3);
		passX.filter(*Pass_cloud_filtered_X);


	}

	void run()
	{
		emty = false;
		con = 0;
		oke == false;
		// create a new grabber for OpenNI devices
		pcl::io::OpenNI2Grabber *interface = new pcl::io::OpenNI2Grabber();

		//interface->setDepthCameraIntrinsics(0, 0, 320, 240);
		// make callback function from member function
		boost::function<void(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> f = boost::bind(&SimpleOpenNIProcessor::cloud_cb_, this, _1);
		
		boost::signals2::connection c = interface->registerCallback(f);
		// start receiving point clouds
		interface->start();

		//while (true)
		//{
		boost::this_thread::sleep(boost::posix_time::seconds(4));
		//if (con = 1) break;
		//	}

		// stop the grabber
		interface->stop();
		if (Pass_cloud_filtered_X->empty()) con = 0;
		else if (pcl::io::savePCDFileBinary("ddpcd.pcd", *Pass_cloud_filtered_X) != -1)
		{
			con = 1;
			cout << "!";
			cout << "done";
			oke = true;
		}


	}

	int con;
	bool oke;
	bool emty;
};


void extractEuclideanClusters(pcl::PointCloud<pcl::PointXYZ> &cloud, pcl::PointCloud<pcl::Normal> &normals, float tolerance, const boost::shared_ptr<pcl::KdTree<pcl::PointXYZ> > &tree,
	std::vector<pcl::PointIndices> &clusters, double eps_angle, unsigned int min_cluster, unsigned int max_cluster)
{
	std::vector<bool> processed(cloud.points.size(), false);// list check if points have been processed yet
	std::vector<int> nn_indices;
	std::vector<float> nn_distances;
	//Eigen::Vector4f p = normals->points[].getNormalVector4fMap();
	// all points in cloud
	for (size_t i = 0; i < cloud.points.size(); ++i)
	{
		if (processed[i])  //points have been processed yet?
			continue;

		std::vector<unsigned int> query_point; // list of L have been processed
		int sq_idx = 0;
		query_point.push_back(i); // store point i to L;

		processed[i] = true; //

		//for every point in L
		while (sq_idx < static_cast<int> (query_point.size()))
		{
			// Search for neighboors of query point
			if (!tree->radiusSearch(query_point[sq_idx], tolerance, nn_indices, nn_distances))//if point have no neighbor->continue
			{
				sq_idx++;
				continue;
			}

			for (unsigned int j = 1; j < nn_indices.size(); ++j)             // nn_indices[0] should be sq_idx
			{
				if (processed[nn_indices[j]])                         // Has this point been processed before ?
					continue;

				//calculate angle between query_point and it's neighboor
				double dot_p = normals.points[i].normal[0] * normals.points[nn_indices[j]].normal[0] +
					normals.points[i].normal[1] * normals.points[nn_indices[j]].normal[1] +
					normals.points[i].normal[2] * normals.points[nn_indices[j]].normal[2];
				if (fabs(acos(dot_p)) < eps_angle)
				{
					processed[nn_indices[j]] = true;
					query_point.push_back(nn_indices[j]);
				}
			}

			sq_idx++;
		}
		// If this queue236* is satisfactory, add to the clusters
		if (query_point.size() >= min_cluster && query_point.size() <= max_cluster)
		{
			pcl::PointIndices r;


			r.indices.resize(query_point.size());

			for (size_t j = 0; j < query_point.size(); ++j)
			{//size_t
				r.indices[j] = query_point[j];

			}
			std::sort(r.indices.begin(), r.indices.end());
			r.indices.erase(std::unique(r.indices.begin(), r.indices.end()), r.indices.end());

			r.header = cloud.header;
			clusters.push_back(r);   // We could avoid a copy by working directly in the vector
		}
		if (query_point.size() <= min_cluster && query_point.size() >= 1300)
		{
			pcl::PointIndices r;

			r.indices.resize(query_point.size());
			int num = 0;
			for (size_t j = 0; j < query_point.size(); ++j)
			{//size_t
				r.indices[j] = query_point[j];
				if (cloud.points[query_point[j]].z >= 0.68 && cloud.points[query_point[j]].z <= 0.7) num++;
			}
			if (num >= 1000)
			{
				std::sort(r.indices.begin(), r.indices.end());
				r.indices.erase(std::unique(r.indices.begin(), r.indices.end()), r.indices.end());

				r.header = cloud.header;
				clusters.push_back(r);   // We could avoid a copy by working directly in the vector
			}
		}
	}
}

void specicalcase(pcl::PointCloud<pcl::PointXYZ> &cloud,float tolerance, const boost::shared_ptr<pcl::KdTree<pcl::PointXYZ> > &tree,
std::vector<pcl::PointIndices> &clusters,int fpoint)
{
	std::vector<int> nn_indices;
	std::vector<float> nn_distances;
	std::vector<unsigned int> query_point;
	 tree->radiusSearch(fpoint, tolerance, nn_indices, nn_distances);
	 for (unsigned int j = 0; j < nn_indices.size(); ++j)             // nn_indices[0] should be sq_idx
	 {
		 query_point.push_back(nn_indices[j]);
	 }

	 pcl::PointIndices r;
	 r.indices.resize(query_point.size());

	 for (size_t j = 0; j < query_point.size(); ++j)
	 {//size_t
		 r.indices[j] = query_point[j];

	 }
	 std::sort(r.indices.begin(), r.indices.end());
	 r.indices.erase(std::unique(r.indices.begin(), r.indices.end()), r.indices.end());

	 r.header = cloud.header;
	 clusters.push_back(r);   // We could avoid a copy by working directly in the vector

}

int check_ACK()
{
	string ACK = Rdata.substr(0, 3);// 
	if (ACK == "QeR")
	{
		//cout << "Failed" << endl;
		return 0;
	}
	else if (ACK == "QoK")
	{
		//cout << "Successed" << endl;
		return 1;
	}
	else
	{
		cout << "Error Communication" << endl;
		return 2;
	}
}


void permutation(int a[], int i, int j)
{
	int dem = a[i];
	a[i] = a[j];
	a[j] = dem;
}

void fpermutation(float a[], int i, int j)
{
	float dem = a[i];
	a[i] = a[j];
	a[j] = dem;
}

void command(string exp)
{
	bool k = false;
	port.writeString(exp);
	Rdata = port.readStringUntil("\r");
	if (check_ACK() == 2) {}
	boost::this_thread::sleep(boost::posix_time::seconds(0.5));
	port.writeString(exp);
	Rdata = port.readStringUntil("\r");
	if (check_ACK() == 2) {}
	boost::this_thread::sleep(boost::posix_time::seconds(0.5));
	//cout << Rdata << endl;
	do{
		boost::this_thread::sleep(boost::posix_time::seconds(0.5));
		port.writeString(exp);
		Rdata = port.readStringUntil("\r");
		if (check_ACK() == 2)
		{
			cout << Rdata << endl;
			k = true;
			break;
		}
	} while (check_ACK() != 1);
	while (k == true)
	{
	}
}

bool th2D=false;
boost::timed_mutex mtex;
cv::VideoCapture cap(0);
Mat frame1;
SimpleOpenNIProcessor v;
void thread3()
{
	std::string move = "1;1;EXECMOV PJOG\r";
	std::string pick = "1;1;OUT=0;0100\r";
	std::string place = "1;1;OUT=0;0000\r";
	while (true)
	{
		if (th2D == true)
		{
			
			v.run();
			boost::unique_lock<boost::timed_mutex> lock{ mtex,boost::try_to_lock };
			if (v.oke ==true)
			{
				th2D = false;
				cout << "thread3" << endl;
			}
		}
	}
}
void thread1()
{

	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded

	Mat edges;
	namedWindow("edges", 1);
	int i = 1;
	//for (;;)
	while (true)
	{
			//Mat frame;
			cap >> frame1; // get a new frame from camera
			imshow("edges", frame1);
			if (cv::waitKey(30) >= 0)
			{

				//break;
			}
		}
	}
	//mtex.unlock();

void thread2()
{	
	while (true)
	{
		
		if (v.oke = true)
		{
			
			ch++;
			//if (v.emty = true) break;
			
			//return 0;
			pcl::PCDWriter writer;
			pcl::PointCloud<pcl::PointXYZ>::Ptr extract_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
			const clock_t begin_time = clock();

			//reading point from file
			if (pcl::io::loadPCDFile<pcl::PointXYZ>("ddpcd.pcd", *cloud1) == -1) //* load the file
			{
				PCL_ERROR("Couldn't read file ddpcd.pcd \n");
			}
			//cout << "ADasd" << endl;
			boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("3D Viewer"));
			//viewer1->addCoordinateSystem();
			viewer1->addPointCloud(cloud1);


			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());


			// normals estimtion
			//-------------------------------------------------------------------------
			//--normal estimation with unorganized dataset----------------------------
			pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

			pcl::NormalEstimation <pcl::PointXYZ, pcl::Normal> ne;
			ne.setInputCloud(cloud1);

			ne.setSearchMethod(tree);
			// Use all neighbors in a sphere of radius 3cm
			ne.setRadiusSearch(0.01);//0.01
			// Compute the features
			ne.compute(*normals);
			//---------------------------------------------------------
			if(ch !=5) std::cout << "Size cloud1 :" << cloud1->points.size() << endl;
			else {
				std::cout << "Size cloud1 :0" << endl; break;
			}
			//------------------------------------------------------------------------
			
			//euclidean clusters extraction base on normals angle fifference
			//------------------------------------------------------------------------
			//kdtree searching
			boost::shared_ptr<pcl::KdTree<pcl::PointXYZ> > kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>());
			kdtree->setInputCloud(cloud1);
			std::vector<int> indices;

			std::vector <float> distance;
			//emty list of cluster
			std::vector <pcl::PointIndices> cluster_indices;
			std::vector <pcl::PointNormal> dir;
			float tolerance = 0.003f;//0.003
			double angle_error = 0.5;//0.4
			unsigned int min_cluster = 3000;//2700
			unsigned int max_cluster = 10000;

			//pcl::extractEuclideanClusters<pcl::PointXYZ,pcl::Normal>(*cloud1,*normals,0.03f,kdtree,cluster_indices,0.3);
			extractEuclideanClusters(*cloud1, *normals, tolerance, kdtree, cluster_indices, angle_error, min_cluster, max_cluster);


			std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster;
			std::vector<pcl::PointCloud<pcl::Normal>::Ptr> e_normal;
			int j = 0;

			for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::Normal>::Ptr ee_normal(new pcl::PointCloud<pcl::Normal>);
				for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
				{
					cloud_cluster->points.push_back(cloud1->points[*pit]);
					ee_normal->points.push_back(normals->points[*pit]);

				}

				ee_normal->width = static_cast<uint32_t>(cloud_cluster->points.size());
				ee_normal->height = 1;
				ee_normal->is_dense = true;
				e_normal.push_back(ee_normal);
				cloud_cluster->width = static_cast<uint32_t>(cloud_cluster->points.size());
				cloud_cluster->height = 1;
				cloud_cluster->is_dense = true;
				cluster.push_back(cloud_cluster);
				std::cout << "PointCloud representing the Cluster_" << j << " using xyzn: " << cloud_cluster->points.size() << " data points." << std::endl;
				j++;
			}
			if (cluster.size() == 0) break;

			int centroidx[10];
			int centroidy[10];
			int centroidz[10];
			int pre_centroidz[10];
			int vecZ[10];
			float cam_centroidx[10];
			float cam_centroidy[10];
			float cam_centroidz[10];
			int cluster_size[10];

			for (int i = 0; i < cluster.size(); i++)
			{

				//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> b(cluster[i], std::rand() % 255, std::rand() % 255, 255);
				//viewer1->addPointCloud(cluster[i], b, id);
				Eigen::Vector4f xyz_centroid;
				compute3DCentroid(*cluster[i], xyz_centroid);

				double dot_X = e_normal[i]->points[e_normal[i]->points.size() / 2].normal_x * 1 + e_normal[i]->points[e_normal[i]->points.size() / 2].normal_y * 0 + e_normal[i]->points[e_normal[i]->points.size() / 2].normal_z * 0;
				double dot_Y = e_normal[i]->points[e_normal[i]->points.size() / 2].normal_x * 0 + e_normal[i]->points[e_normal[i]->points.size() / 2].normal_y * 1 + e_normal[i]->points[e_normal[i]->points.size() / 2].normal_z * 0;
				double dot_Z = e_normal[i]->points[e_normal[i]->points.size() / 2].normal_x * 0 + e_normal[i]->points[e_normal[i]->points.size() / 2].normal_y * 0 + e_normal[i]->points[e_normal[i]->points.size() / 2].normal_z * 1;

				//cout << "Sad " << e_normal[i]->points[e_normal[i]->points.size() / 2].normal_y;
				//cout << "Sad " << dot_Z << endl;
				//std::cout << xyz_centroid.x() << " " << xyz_centroid.y() << " " << xyz_centroid.z() << endl;

				//std::cout << "a_x" << acos(dot_X) * 180 / PI << " a_y" << acos(dot_Y) * 180 / PI << " a_z" << acos(dot_Z) * 180 / PI << endl;
				int XX = acos(dot_X) * 180 / PI;
				//float  rXX = XX * 100 / 100;
				int YY = acos(dot_Y) * 180 / PI;
				int ZZ;
				if (e_normal[i]->points[e_normal[i]->points.size() / 2].normal_y < 0)  ZZ = ((acos(dot_Z)) * 180 / PI);
				else ZZ = -((acos(dot_Z)) * 180 / PI);
				
				int cenX = xyz_centroid.y() * 1000 + 255;//220.7
				int cenY = -xyz_centroid.x() * 1000 +29 ;//-0.67
				int cenZ = -xyz_centroid.z() * 1000 + 803.19 - 13;
				int pre_cenZ = cenZ + 80;
				std::cout << "x" << cenX << " y" << cenY << " z" << cenZ << endl;

				//
				centroidx[i] = cenX;
				centroidy[i] = cenY;
				centroidz[i] = cenZ;
				cam_centroidx[i] = xyz_centroid.x();
				cam_centroidy[i] = xyz_centroid.y();
				cam_centroidz[i] = xyz_centroid.z();
				pre_centroidz[i] = pre_cenZ;
				vecZ[i] = ZZ;
				cluster_size[i] = cluster[i]->points.size();
			}


			float minX;
			float maxX;
			float minXYY;
			float minXZZ;
			float maxXYY;
			float maxXZZ;
			float distances;
			float x1;
			float x2;
			float y1;
			float y2;
			float maxXLineX[10];
			float maxXLineY[10];
			float maxXLineZ[10];
			unsigned int k = 0;
			unsigned int h = 0;
			
			float maxdistances1 = 0;
			float maxdistances2 = 0;

			int pos_cluster[10];
			std::vector <pcl::PointIndices> spe_indices;
			std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> special;
			std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> spe_cluster;
			std::vector<pcl::PointCloud<pcl::Normal>::Ptr> spe_e_normal;
			boost::shared_ptr<pcl::KdTree<pcl::PointXYZ> > spe_kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>());
			unsigned int n = 0;
			for (int i = 0; i < cluster.size(); i++)
			{

				if (cluster[i]->points.size()>5000)
				{
					
					spe_kdtree->setInputCloud(cluster[i]);
					for (int j = 0; j < cluster[i]->points.size(); j++)
					{
						//cout << j << endl;

						if (cluster[i]->points[j].x > cam_centroidx[i])
						{
							h++;
							if (h == 1)
							{
								minX = cluster[i]->points[j].x;
								maxX = cluster[i]->points[j].x;
							}

							if (cluster[i]->points[j].x > maxX)
							{
								maxX = cluster[i]->points[j].x;
								maxXYY = cluster[i]->points[j].y;
								maxXZZ = cluster[i]->points[j].z;
								k = j;
								
							}

						}

						if (j == cluster[i]->points.size() - 1)
						{
							pos_cluster[n] = i;
							
							maxXLineX[i] = maxX;
							maxXLineY[i] = maxXYY;
							maxXLineZ[i] = maxXZZ;
							specicalcase(*cluster[i],0.05, spe_kdtree, spe_indices, k);
							for (std::vector<pcl::PointIndices>::const_iterator it = spe_indices.begin()+n; it != spe_indices.end(); ++it)
							{
								pcl::PointCloud<pcl::PointXYZ>::Ptr spe_cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
								pcl::PointCloud<pcl::Normal>::Ptr spe_ee_normal(new pcl::PointCloud<pcl::Normal>);
								for (std::vector<int>::const_iterator pit = it->indices.begin()+n; pit != it->indices.end(); ++pit)
								{
									spe_cloud_cluster->points.push_back(cluster[i]->points[*pit]);
									spe_ee_normal->points.push_back(e_normal[i]->points[*pit]);

								}
								spe_ee_normal->width = static_cast<uint32_t>(spe_ee_normal->points.size());
								spe_ee_normal->height = 1;
								spe_ee_normal->is_dense = true;
								spe_e_normal.push_back(spe_ee_normal);
								spe_cloud_cluster->width = static_cast<uint32_t>(spe_cloud_cluster->points.size());
								spe_cloud_cluster->height = 1;
								spe_cloud_cluster->is_dense = true;
								spe_cluster.push_back(spe_cloud_cluster);
							}
							n++;
						}
					}
					k = 0;
					h = 0;
				}
			}

			cout << "spe_cluster.size()" << spe_cluster.size() << endl;
			for (int i = 0; i < spe_cluster.size(); i++)
			{
				int n = 0;
				string Result;          // string which will contain the result
				ostringstream convert;   // stream used for the conversion
				convert << i;      // insert the textual representation of 'Number' in the characters in the stream
				Result = convert.str();
				string id = "cloud" + Result;
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> nn(cluster[i], std::rand() % 255, std::rand() % 255, 255);
				viewer1->addPointCloud(spe_cluster[i], nn, id);
				Eigen::Vector4f xyz_centroid;
				compute3DCentroid(*spe_cluster[i], xyz_centroid);
				double dot_Z = e_normal[i]->points[e_normal[i]->points.size() / 2].normal_x * 0 + e_normal[i]->points[e_normal[i]->points.size() / 2].normal_y * 0 + e_normal[i]->points[e_normal[i]->points.size() / 2].normal_z * 1;
				int ZZ;
				if (e_normal[i]->points[e_normal[i]->points.size() / 2].normal_y < 0)  ZZ = ((acos(dot_Z)) * 180 / PI);
				else ZZ = -((acos(dot_Z)) * 180 / PI);

				int cenX = xyz_centroid.y() * 1000 + 255;//220.7
				int cenY = -xyz_centroid.x() * 1000 + 29;//-0.67
				int cenZ = -xyz_centroid.z() * 1000 + 803.19 - 10;
				//cout << "cenX " << cenX << " cenY " << cenY << " cenZ " << cenZ << endl;
				int pre_cenZ = cenZ + 80;
				centroidx[pos_cluster[n]] = cenX;
				centroidy[pos_cluster[n]] = cenY;
				centroidz[pos_cluster[n]] = cenZ;
				cam_centroidx[pos_cluster[n]] = xyz_centroid.x();
				cam_centroidy[pos_cluster[n]] = xyz_centroid.y();
				cam_centroidz[pos_cluster[n]] = xyz_centroid.z();
				pre_centroidz[pos_cluster[n]] = pre_cenZ;
				vecZ[pos_cluster[n]] = ZZ;
				//cluster_size[pos_cluster[n]] = spe_cluster[i]->points.size();
				n++;
			}


			

			for (unsigned int i = 0; i < cluster.size(); i++)
			{
				for (unsigned int j = i + 1; j < cluster.size(); j++)
				{
					if (centroidz[j] > centroidz[i])
					{
						permutation(centroidz, i, j);
						permutation(centroidx, i, j);
						permutation(centroidy, i, j);
						permutation(pre_centroidz, i, j);
						permutation(vecZ, i, j);
						permutation(cluster_size, i, j);
						fpermutation(maxXLineX, i, j);
						fpermutation(maxXLineY, i, j);
						fpermutation(maxXLineZ, i, j);
						fpermutation(cam_centroidx, i, j);
						fpermutation(cam_centroidy, i, j);
						fpermutation(cam_centroidx, i, j);
					}
				}
			}

			while (true)
			{
				if (th2D == false) break;
			}

			while (!viewer1->wasStopped())
			{
					viewer1->spinOnce();
			}


			//cout << " Set speed: ";
			//port.writeString("1;1;EXECSPD 5\r");
			//Rdata = port.readStringUntil("\r");
			//if (check_ACK() != 1) {};

			for (int i = 0; i < cluster.size(); i++)
			{

				std::string move = "1;1;EXECMOV PJOG\r";
				std::string pick = "1;1;OUT=0;0100\r";
				std::string place = "1;1;OUT=0;0000\r";
				std::string exp;
				std::string sXX = boost::lexical_cast<std::string>(centroidx[i]);
				std::string sYY = boost::lexical_cast<std::string>(centroidy[i]);
				std::string sZZ = boost::lexical_cast<std::string>(centroidz[i]);
				std::string s_aZZ = boost::lexical_cast<std::string>(vecZ[i]);
				std::string pre_sZZ = boost::lexical_cast<std::string>(pre_centroidz[i]);

				if (cluster_size[i] < 2500)
				{

					int push_cenZZ = centroidz[i] - 1;
					int push_cenXX = centroidx[i] + 50;
					std::string sXX = boost::lexical_cast<std::string>(centroidx[i]);
					std::string sYY = boost::lexical_cast<std::string>(centroidy[i]);
					std::string sZZ = boost::lexical_cast<std::string>(push_cenZZ);
					std::string s_aZZ = boost::lexical_cast<std::string>(vecZ[i]);
					std::string pre_sZZ = boost::lexical_cast<std::string>(pre_centroidz[i]);
					std::string pushXX = boost::lexical_cast<std::string>(push_cenXX);
					exp = "1;1;EXECPJOG=(" + sXX + ".00," + sYY + ".00," + pre_sZZ + ".00,0.00," + s_aZZ + ".00,0.00)(6,0)\r";
					command(exp);
					command(move);
					exp = "1;1;EXECPJOG=(" + sXX + ".00," + sYY + ".00," + sZZ + ".00,0.00," + s_aZZ + ".00,0.00)(6,0)\r";
					command(exp);
					command(move);
					exp = "1;1;EXECPJOG=(" + pushXX + ".00," + sYY + ".00," + sZZ + ".00,0.00," + s_aZZ + ".00,0.00)(6,0)\r";
					command(exp);
					command(move);
					exp = "1;1;EXECPJOG=(" + pushXX + ".00," + sYY + ".00," + pre_sZZ + ".00,0.00," + s_aZZ + ".00,0.00)(6,0)\r";
					command(exp);
					command(move);
					exp = ("1;1;EXECPJOG=(28.54,-318.48,166.40,0.00,180.00,0.00)(6,0)\r");
					command(exp);
					command(move);
					break;
				}

				else 
				{
				exp = "1;1;EXECPJOG=(200.00,-300.00,200.00,0.00,180.00,0.00)(6,0)\r";
				command(exp);
				command(move);
				exp = "1;1;EXECPJOG=(" + sXX + ".00," + sYY + ".00," + pre_sZZ + ".00,0.00,180.00,0.00)(6,0)\r";
				command(exp);
				command(move);
				exp = "1;1;EXECPJOG=(" + sXX + ".00," + sYY + ".00," + sZZ + ".00,0.00," + s_aZZ + ".00,0.00)(6,0)\r";
				command(exp);
				command(move);
				//std::system("pause");
				command(pick);
				exp = "1;1;EXECPJOG=(" + sXX + ".00," + sYY + ".00," + pre_sZZ + ".00,0.00," + "180" + ".00,0.00)(6,0)\r";
				command(exp);
				command(move);
				exp = "1;1;EXECPJOG=(200.00,-300.00,200.00,0.00,180.00,0.00)(6,0)\r";
				command(exp);
				command(move);
				exp = "1;1;EXECPJOG=(10.00,-350.00,110.00,0.00,180.00,0.00)(6,0)\r";
				command(exp);
				command(move);
				if (i == (cluster.size() - 1)) th2D = true;
				exp = "1;1;EXECPJOG=(10.00,-350.00,80.00,0.00,180.00,0.00)(6,0)\r";
				command(exp);
				command(move);
				command(place);
				exp = "1;1;EXECPJOG=(10.00,-350.00,110.00,0.00,180.00,0.00)(6,0)\r";
				command(exp);
				command(move);
				exp = "1;1;EXECPJOG=(250.00,-300.00,110.00,0.00,180.00,0.00)(6,0)\r";
				command(exp);
				command(move);
				command(pick);
				boost::this_thread::sleep(boost::posix_time::seconds(0.5));
				command(place);
				boost::this_thread::sleep(boost::posix_time::seconds(0.5));
				}
				//-----------------------------------------------------------------------------------------------------------//
				Mat frame;
				boost::unique_lock<boost::timed_mutex> lock{ mtex, boost::try_to_lock };
				if (lock.owns_lock() || lock.try_lock_for(boost::chrono::seconds{ 1 }))
				{
					frame = frame1.clone();
				}
				cv::inRange(frame, cv::Scalar(Rlow, Glow, Blow), cv::Scalar(Rhigh, Ghigh, Bhigh), redOutput);
				Mat element = getStructuringElement(0, Size(3, 3));
				erode(redOutput, redOutput, element);
				dilate(redOutput, redOutput, element);
				erode(redOutput, redOutput, element);
				dilate(redOutput, redOutput, element);
				std::vector<std::vector<cv::Point> > contours;
				cv::findContours(redOutput, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
				for (int i = 0; i < contours.size(); ++i)
				{
					if (contours[i].size()>250)
					{
						// fit bounding rectangle around contour
						cv::RotatedRect rotatedRect = cv::minAreaRect(contours[i]);

						// read points and angle
						cv::Point2f rect_points[4];
						rotatedRect.points(rect_points);

						float  angle = rotatedRect.angle; // angle

						Point2f pts1[4];
						rotatedRect.points(pts1);
						// read center of rotated rect
						cv::Point2f center = rotatedRect.center; // center
						for (unsigned int j = 0; j < 4; ++j)
						{
							cv::line(frame, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 255, 0));
							// draw center and print text
							std::stringstream ss;   ss << angle; // convert float to string
							cv::circle(frame, center, 5, cv::Scalar(0, 255, 0)); // draw center
							cv::putText(frame, ss.str(), center + cv::Point2f(-25, 25), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 255)); // print angle
						}
						//cv::imshow("aa", frame);
					//	std::cout << "center: " << center << std::endl;
						cv::circle(frame, pts1[0], 10, cv::Scalar(255, 255, 255));//trang
						cv::circle(frame, pts1[1], 10, cv::Scalar(0, 0, 0));//den
						cv::circle(frame, pts1[2], 10, cv::Scalar(100, 100, 100));
						Eigen::Vector3d p1, p2, p3, co_center;
						co_center << center.x, center.y, 1;
						p1 << pts1[0].x, pts1[0].y, 1;
						p2 << pts1[1].x, pts1[1].y, 1;
						p3 << pts1[2].x, pts1[2].y, 1;
						co_center *= 716.9573;
						p1 *= 716.9573;
						p2 *= 716.9573;
						p3 *= 716.9573;
						//corner
						p1 = in_intrinsic*p1;
						p1 -= t;
						p1 = in_R*p1;
						p1 = A*p1;
						p1 += tt;

						p2 = in_intrinsic*p2;
						p2 -= t;
						p2 = in_R*p2;
						p2 = A*p2;
						p2 += tt;

						p3 = in_intrinsic*p3;
						p3 -= t;
						p3 = in_R*p3;
						p3 = A*p3;
						p3 += tt;
						//Center
						co_center = in_intrinsic*co_center;
						co_center -= t;
						co_center = in_R*co_center;
						co_center = A*co_center;
						co_center += tt;

						//cout << co_center << endl;
						//cout << co_center(0) << endl;
						//cout << co_center(1) << endl;
						double L1 = sqrt(pow(p1(0) - p2(0), 2) + pow(p1(1) - p2(1), 2));
						double L2 = sqrt(pow(p2(0) - p3(0), 2) + pow(p2(1) - p3(1), 2));
						double vec_corner[2];
						double err_angle;
						if (L1 < L2)
						{

							if (p2(0) > p3(0))
							{
								vec_corner[0] = p3(0) - p2(0);
								vec_corner[1] = p3(1) - p2(1);
							}
							else{
								vec_corner[0] = p2(0) - p3(0);
								vec_corner[1] = p2(1) - p3(1);
							}
							err_angle = acos((vec_corner[1] / L2)) * 180 / PI;
						//	cout << "1 " << vec_corner[1] << endl;
						}
						else
						{
							if (p1(0) > p2(0))
							{
								vec_corner[0] = p2(0) - p1(0);
								vec_corner[1] = p2(1) - p1(1);
							}
							else{
								vec_corner[0] = p1(0) - p2(0);
								vec_corner[1] = p1(1) - p2(1);
							}
							err_angle = acos((vec_corner[1] / L1)) * 180 / PI;
							//cout << "2 " << vec_corner[1] << endl;
						}
					//	cout << "err_angle " << err_angle << endl;
						//calculate error angle
						if (err_angle <= 90) err_angle = -(90 - err_angle);
						else err_angle = (err_angle - 90);
						double ss_co_X = co_center(0) +35;
						std::string recX = boost::lexical_cast<std::string>(ss_co_X);
						recX = recX.substr(0, 6);
						//cout << "recX: " << recX << endl;
						double ss_co_Y = co_center(1)+44;
						std::string recY = boost::lexical_cast<std::string>(ss_co_Y);
						recY = recY.substr(0, 6);
						//cout << "recY: " << recY << endl;
						std::string e_angle = boost::lexical_cast<std::string>(err_angle);
						e_angle = e_angle.substr(0, 6);
						cout << "angle: " << e_angle << endl;
						std::string rec_Z = "76.00 ";
						std::string pre_rec_Z = "100.00 ";
						cout << "err_angle " << err_angle << endl;
					//	cout << "L1 " << L1 << endl;
					//	cout << "L2 " << L2 << endl;


						std::string move2D = "1;1;EXECPJOG=(" + recX + "," + recY + "," + pre_rec_Z + ",0.00, 180.00,0.00)(6,0)\r";
						command(move2D);
						command(move);
						move2D = "1;1;EXECPJOG=(" + recX + "," + recY + "," + rec_Z + ",0.00, 180.00,0.00)(6,0)\r";
						command(move2D);
						command(move);
						command(pick);
						boost::this_thread::sleep(boost::posix_time::seconds(0.5));
						move2D = "1;1;EXECPJOG=(" + recX + "," + recY + "," + pre_rec_Z + "," + e_angle + ",180.00,0.00)(6,0)\r";
						command(move2D);
						command(move);
						boost::this_thread::sleep(boost::posix_time::seconds(0.5));
						//move2D = "1;1;EXECPJOG=(-10.00," + recY + "," + pre_rec_Z + "," + e_angle + ",180.00,0.00)(6,0)\r";
						move2D = "1;1;EXECPJOG=(-101.82,-309.12," + pre_rec_Z + "," + e_angle + ", 180.00, 0.00)(6, 0)\r";
						command(move2D);
						command(move);
						boost::this_thread::sleep(boost::posix_time::seconds(0.5));
						move2D = "1;1;EXECPJOG=(-101.82,-309.12," + rec_Z + "," + e_angle + ",180.00,0.00)(6,0)\r";
						command(move2D);
						command(move);
						command(place);
						cout << e_angle << endl;
						//std::system("pause");
						move2D = "1;1;EXECPJOG=(-101.82,-309.12," + pre_rec_Z + "," + e_angle + ",180.00,0.00)(6,0)\r";
						command(move2D);
						command(move);
						
					}
				}
			}//for....
			
		}
	}
	
}

int main()
{
	port.setTimeout(boost::posix_time::seconds(2));
	cout << "Initialize the communication: ";
	port.writeString("1;1;OPEN=NARCUSR\r");
	Rdata = port.readStringUntil("\r");
	if (check_ACK() != 1){};
	cout << Rdata << endl;
	cout << "Reset alarm to reactive robot: ";
	port.writeString("1;1;RSTALRM\r");
	Rdata = port.readStringUntil("\r");
	//Rdata = port.readString(3);
	if (check_ACK() != 1) {};
	cout << Rdata << endl;
	cout << "Turn ON remote control over serial interface: ";
	port.writeString("1;1;CNTLON\r");
	if (check_ACK() != 1) {};
	cout << Rdata << endl;
	cout << "Turn ON the servo motors: ";
	port.writeString("1;1;SRVON\r");
	Rdata = port.readStringUntil("\r");
	if (check_ACK() != 1) {};
	cout << Rdata << endl;

	//Eigen::Matrix3d intrinsic;
	in_intrinsic << 0.00182326, 0, -0.6072542259,
		0, 0.00183672, -0.4508807281,
		0, 0, 1.0000;

	in_R << 0.9994, -0.0230, -0.0236,
		0.0219, 0.9990, -0.0405,
		0.0245, 0.0401, 0.9989;
	A << 1, 0, 0,
		0, -1, 0,
		0, 0, -1;
	v.run();
	boost::thread t1{ thread1 };
	boost::thread t2{ thread2 };
	boost::thread t3{ thread3 };
	
	//t1.join();
	//t2.join();
	bool run = true;
	th2D = false;
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	while (true)
	{

	}

}