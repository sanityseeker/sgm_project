
#include "stdafx.h"

#pragma once

#include "MultiViewDepthCalculator.h"

#include <sgm.h>

#ifdef USE_GPU
#include <sgm_gpu.h>
#endif

#ifdef USE_GPU
#pragma comment(lib, "sfm_gpu.lib")
#endif

using namespace std;
//using namespace openMVG;
using namespace boost;
using namespace cv;
using namespace xlslib_core;

#define ENABLE_RECTIFICATION
//#define USE_FULLIMAGES
#define USE_FLOAT_DISPARITY
//#define DEBUG_OUTPUT
//#define TRUNCATE_DISP_MAP
//#define FILTER_DISP_MAP

#define DEPTH_MAP_NODATA_VALUE -10000
//#define SGM_NULL_VALUE 10001


void MultiViewDepthCalculator::LoadImages(string images_dir)
{
#ifndef ENABLE_RECTIFICATION
	
	Images.resize( Doc->_vec_imageNames.size() );
	GradientMaps.resize( Doc->_vec_imageNames.size() );
	cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,cv::Size(5,5) );

	for( unsigned int i(0); i < Doc->_vec_imageNames.size(); ++i )
	{
		//Images[i] = LoadGrayScaleImageCV( images_dir + Doc->_vec_imageNames[i], pyr_lvl );
		//Images[i] = LoadRaster( images_dir + Doc->_vec_imageNames[i], pyr_lvl, false );
		cv::Mat InputImage = LoadGrayScaleImageCV( images_dir + Doc->_vec_imageNames[i], pyr_lvl );
		Images[i] = CensusTransform5x5( InputImage );
		//Images[i] = CensusTransform7x7( InputImage ); don't work(yjhv

		//cvtColor( Images[i], Images[i], CV_BGR2HSV );

		if( Images[i].empty() ) 
		{
			cout << "Failed to load " << Doc->_vec_imageNames[i] << endl;
			continue;
		}
		if( !Images[i].empty() ) cout << Doc->_vec_imageNames[i] << " Image Loaded. Size: " << Images[i].cols << "x" << Images[i].rows << endl;

		cv::Mat ImageBW = LoadGrayScaleImageCV( images_dir + Doc->_vec_imageNames[i], pyr_lvl );
		bool UseL2gradient = false;
		double LowThreshold = 1900;
		double HighThreshod = 2400;
		Canny( ImageBW, GradientMaps[i], LowThreshold, HighThreshod, 5, UseL2gradient );
		//cv::dilate( GradientMaps[i], GradientMaps[i], element);

		imwrite( images_dir + "canny\\" + Doc->_vec_imageNames[i], GradientMaps[i] );

		cv::Mat Test = LoadRaster( images_dir + Doc->_vec_imageNames[i], pyr_lvl, false );//test to view images
		imwrite( images_dir + "pyr\\" + Doc->_vec_imageNames[i], Test );
	}
		//Image<unsigned char> LoadGrayScaleImage(std::string filename, int pyr_lvl = 0, cv::Rect image_rect = cv::Rect());
#endif
}

void MultiViewDepthCalculator::UpdateCosts( std::tuple<int, int, int> xyz, float IndividualCost )
{
	UpdateCosts( std::get<0>(xyz), std::get<1>(xyz), std::get<2>(xyz), IndividualCost );
}

void MultiViewDepthCalculator::UpdateCosts( int x, int y, int z, float IndividualCost )
{
	//cout << "MultiViewDepthCalculator::UpdateCosts on address = " << x << " "<< y << " " << z << " IndividualCost = " << IndividualCost << endl;

	float TrimmedCost = IndividualCost < CostThreshold ? IndividualCost : CostThreshold;
	int n = NumberOfProjections[x][y][z];
	if( n == 0 )
		LocalCosts[x][y][z] = TrimmedCost;
	else
	{
		float old_value = LocalCosts[x][y][z];
		float new_value = ( n*old_value + TrimmedCost ) / float(n+1);
		LocalCosts[x][y][z] = new_value;
	}
	NumberOfProjections[x][y][z] += 1;
}


float L1Distance( const Vec3b & a, const Vec3b & b )
{
	return abs( a[0]-b[0] )+abs( a[1]-b[1] )+abs( a[2]-b[2] );
}

float L2Distance( const Vec3b & a, const Vec3b & b )
{
	float d0 = abs( a[0]-b[0] );
	float d1 = abs( a[1]-b[1] );
	float d2 = abs( a[2]-b[2] );
	return sqrt( d0*d0 + d1*d1 + d2*d2 );
}

float GetChannelDifference( const Vec3b & a, const Vec3b & b, int channelNumber )
{
	if( channelNumber < 0 ) channelNumber = 0;
	if( channelNumber > 2 ) channelNumber = 2;
	return abs( a[channelNumber]-b[channelNumber] );
}

float GetNCC( SimpleFloatPoint a, SimpleFloatPoint b, cv::Mat aImage, cv::Mat bImage, int ApertureRadius )
{
	if( a.x < ApertureRadius || a.y < ApertureRadius || b.x < ApertureRadius || b.y < ApertureRadius ) return 0;
	if( a.x > aImage.cols - ApertureRadius - 1 || a.y > aImage.rows - ApertureRadius - 1 ) return 0;
	if( b.x > bImage.cols - ApertureRadius - 1 || b.y > bImage.rows - ApertureRadius - 1 ) return 0;

	cv::Mat result;
	cv::Rect Aperture( -ApertureRadius, -ApertureRadius, ApertureRadius+ApertureRadius+1, ApertureRadius+ApertureRadius+1 );
	cv::Rect aRect = Aperture + cv::Point(a.x,a.y);
	cv::Rect bRect = Aperture + cv::Point(b.x,b.y);

	matchTemplate( aImage(aRect), bImage(bRect), result, CV_TM_CCOEFF_NORMED );
	float ncc = result.at<float>(0,0);
	return ( 1.0 - ncc ) / 2.0;
}

float MultiViewDepthCalculator::CalculatePixelSimilarity( SimpleFloatPoint a, SimpleFloatPoint b, int ImageIda, int ImageIdb )
{
	/*Vec3b aValue = Images[ImageIda].at<Vec3b>(a.y-0.5,a.x-0.5);
	Vec3b bValue = Images[ImageIdb].at<Vec3b>(b.y-0.5,b.x-0.5);
	return L1Distance(aValue, bValue);*/
	//return GetNCC( a, b, Images[ImageIda], Images[ImageIdb], 2 );
	//return GetChannelDifference( aValue, bValue, 0 );
	//return abs( Images[ImageIda].at<uchar>(a.y,a.x) - Images[ImageIdb].at<uchar>(b.y,b.x) );
	//return HammingDistance( Images[ImageIda].at<uchar>(a.y,a.x), Images[ImageIdb].at<uchar>(b.y,b.x) );
	return HammingDistance( Images[ImageIda].at<int>(a.y-0.5,a.x-0.5), Images[ImageIdb].at<int>(b.y-0.5,b.x-0.5) );
	//return HammingDistance( Images[ImageIda].at<long int>(a.y,a.x), Images[ImageIdb].at<long int>(b.y,b.x) );
}

float MultiViewDepthCalculator::CalculatePixelSimilarity( const Point &a, const Point &b, const cv::Mat &img_a, const cv::Mat &img_b )
{
	return HammingDistance( img_a.at<int>(a.y,a.x), img_b.at<int>(b.y,b.x) );
}

float MultiViewDepthCalculator::CalculatePixelSimilarity( const SimpleFloatPoint &a, const SimpleFloatPoint &b, const cv::Mat &img_a, const cv::Mat &img_b )
{
	return HammingDistance( img_a.at<int>(a.y - 0.5,a.x - 0.5), img_b.at<int>(b.y - 0.5,b.x - 0.5) );
}

//#define OUTPUT_VISIBLE_IMAGES_DATA

void MultiViewDepthCalculator::CalculateLocalCosts(string cache_folder)
{
		cout << endl << "MultiViewDepthCalculator::CalculateLocalCosts started" << endl;

		Timer timer;
		timer.reset();

		int dim1 = std::get<0>(GridSize);
		int dim2 = std::get<1>(GridSize);
		int dim3 = std::get<2>(GridSize);

		//LocalCosts = Array3f( boost::extents[std::get<0>(GridSize)][std::get<1>(GridSize)][std::get<2>(GridSize)] );
		//NumberOfProjections = Array3b( boost::extents[std::get<0>(GridSize)][std::get<1>(GridSize)][std::get<2>(GridSize)] );

		LocalCosts = vector< vector< vector < float > > > ( dim1, vector< vector< float > > ( dim2, vector< float > ( dim3, SGM_NULL_VALUE ) ) );//10001
		NumberOfProjections = vector< vector< vector < char > > > ( dim1, vector< vector< char > > ( dim2, vector< char > ( dim3, 0 )  ) );
		GradientMask = vector< vector < unsigned char > >( dim1, vector< unsigned char > ( dim2, 0 ) );
		GradientMaskMat = cv::Mat::zeros( dim2, dim1, CV_8U );

		cout << "Arrays initialized" << endl;	
		cout << "LocalCosts sizes = " << LocalCosts.size() << " " << LocalCosts.front().size() << " " << LocalCosts.front().front().size() << endl;
		cout << "NumberOfProjections sizes = " << NumberOfProjections.size() << " " << NumberOfProjections.front().size() << " " << NumberOfProjections.front().front().size() << endl;

		unsigned int ncores = std::thread::hardware_concurrency();

		cout << "ncores = " << ncores << endl;

		sgmProgress = std::shared_ptr<C_Progress_display>(new C_Progress_display(GetTotalNumberOfVoxels()));

		//load rectified images
#ifdef ENABLE_RECTIFICATION
		map<pair<int, int>, std::tuple<FusielloRectification, cv::Mat, cv::Mat>> recitification_data;
		vector<pair<int, int>> existed_stereo_pairs;

		Rect rectification_rect;

		for(int i = 0; i < Doc->_vec_imageNames.size(); i++)
			for(int j = 0; j < Doc->_vec_imageNames.size(); j++)
				if(i != j)
				{
					string cur_rectification_folder = stlplus::folder_append_separator(cache_folder) + to_string(i) + "_" + to_string(j) + "\\";

					if(stlplus::folder_exists(cur_rectification_folder))
					{
						string left_image_name = cur_rectification_folder + "left.tiff", right_image_name = cur_rectification_folder + "right.tiff";
						string rectification_data_name = cur_rectification_folder + "rectification.dat";

						FusielloRectification rectification;
						rectification.Load(rectification_data_name);

						cv::Mat rect_image_1 = CensusTransform5x5(LoadGrayScaleImageCV(left_image_name, pyr_lvl)), rect_image_2 = CensusTransform5x5(LoadGrayScaleImageCV(right_image_name, pyr_lvl));

						recitification_data[make_pair(i, j)] = std::make_tuple(rectification, rect_image_1, rect_image_2);
						existed_stereo_pairs.push_back(make_pair(i, j));

						rectification_rect = Rect(0, 0, rect_image_1.cols, rect_image_1.rows);
					}
				}

		for each(pair<int, int> spair in existed_stereo_pairs)
		{
			auto initial_pair = spair;
			
			FusielloRectification buf_rect = std::get<0>(recitification_data[spair]);
			buf_rect.swap();

			swap(spair.first, spair.second);

			recitification_data[spair] = std::make_tuple(buf_rect, std::get<2>(recitification_data[initial_pair]), std::get<1>(recitification_data[initial_pair]));
		}

		
#endif

#ifdef OUTPUT_VISIBLE_IMAGES_DATA
		boost::numeric::ublas::matrix<vector<int>> cells(dim1, dim2, vector<int>(dim3, 0));
#endif

		
		//��������� ���������� ������
		int blocks_count = p_vx_manager->GetBlocksCount();
		for(int i = 0; i < blocks_count; i++)
		{
			//��������� ����� �� �������
			auto local_block = p_vx_manager->GetBlockByIndex(i);

			//��������� ���������� ���������
			for(int j = 0; j < local_block->GetSubBlocksCount(); j++)
			{
				char* local_data = NULL;
				std::tuple<int, int, int> begin_voxel_addr;
				int elements_count;

				//��������� ��������� �� ������ ������ �������� �� ������� 
				if(local_block->GetSubBlockByIndex(j, &local_data, begin_voxel_addr, elements_count))
				{
					//cout << "initialize VoxelDataShema" << endl;
					//cout << "elements_count = " << elements_count << endl;	

					vector<std::tuple<int, int, int>> vox_addr_list(ncores);
					vector<VoxelDataShema> vox_shema_list(ncores);

					vector<vector<Point2f>> original_pts1(ncores), original_pts2(ncores), rect_pts1(ncores), rect_pts2(ncores);
					for(int k = 0; k < ncores; k++)
					{
						original_pts1[k].resize(1); original_pts2[k].resize(1);
						rect_pts1[k].resize(1); rect_pts2[k].resize(1);
					}

					//cout << "initialize VoxelDataShema succeeded" << endl;

#pragma omp parallel for
					for(int k = 0; k < ncores; k++)
					{
						//����� ������������ ��� ������ ���������� ����� (����� �������)
						vox_shema_list[k] = VoxelDataShema( Doc->_vec_imageNames.size());
						
						//cout << "vox_shema_list[k] = VoxelDataShema( Doc->_vec_imageNames.size());" << endl;

						int counter = k;
						
						while(counter < elements_count)
						{
							//�������������� ��������� �������� ������� � ������� � ����� ������� � �������
							if(local_block->ConvertElementsPos2VoxelAddress(j, counter, vox_addr_list[k]))
							{
								//get coords
								//Vec3 pos = working_aabb.first + Vec3(std::get<0>(vox_addr_list[k])*voxel_step + voxel_step/2, std::get<1>(vox_addr_list[k])*voxel_step + voxel_step/2, std::get<2>(vox_addr_list[k])*voxel_step + voxel_step/2);
								//cout << "Voxel address = " << std::get<0>(vox_addr_list[k]) << " x " << std::get<1>(vox_addr_list[k]) << " x " << std::get<2>(vox_addr_list[k]) << endl << endl;
								
								//cout << "�������� � ������������ ����� �������� �������" << endl;
								//������������ ����� �������� �������
								vox_shema_list[k].Init(local_data + counter*vox_shema_list[k].total_data_size);

								//cout << "vox_shema_list[k].Init(local_data + counter*vox_shema_list[k].total_data_size);" << endl;

								int base_img_id;
								bool success = vox_shema_list[k].get_base_img_id( base_img_id );
								SimpleFloatPoint BaseImageProjection;
								success &= vox_shema_list[k].get_projection( base_img_id, BaseImageProjection );

								bool VoxelIsVisibleOnBaseImage;
								success &= vox_shema_list[k].get_visibility( base_img_id, VoxelIsVisibleOnBaseImage );

								/*if( success && VoxelIsVisibleOnBaseImage )
								{
									int xx = std::get<0>(vox_addr_list[k]);
									int yy = std::get<1>(vox_addr_list[k]);
									int zz = std::get<2>(vox_addr_list[k]);

									GradientMask[xx][yy] = max( GradientMaps[base_img_id].at<unsigned char>( BaseImageProjection.y-0.5, BaseImageProjection.x-0.5 ), GradientMask[xx][yy] );
									GradientMaskMat.at<unsigned char>( yy, xx ) = max( GradientMaps[base_img_id].at<unsigned char>( BaseImageProjection.y-0.5, BaseImageProjection.x-0.5 ), GradientMaskMat.at<unsigned char>( yy, xx ) );
								}*/

								int local_vis_imgs_count = 1;

								for( int imid(0); imid < NumberOfImages; ++imid )
								{
									if( imid == base_img_id ) continue;
									bool VoxelIsVisibleOnImage;
									bool success = vox_shema_list[k].get_visibility( imid, VoxelIsVisibleOnImage );
									if( !VoxelIsVisibleOnImage ) continue;
									SimpleFloatPoint ImageProjection;
									success &= vox_shema_list[k].get_projection( imid, ImageProjection );

#ifdef ENABLE_RECTIFICATION
									//make rectification
									if(recitification_data.find(make_pair(base_img_id, imid)) == recitification_data.end()) continue;

									auto &rect_data = recitification_data[make_pair(base_img_id, imid)];

									original_pts1[k][0] = Point2f(BaseImageProjection.x*pow(2.0, pyr_lvl), BaseImageProjection.y*pow(2.0, pyr_lvl));
									original_pts2[k][0] = Point2f(ImageProjection.x*pow(2.0, pyr_lvl), ImageProjection.y*pow(2.0, pyr_lvl));

									std::get<0>(rect_data).convert_2_transform_left(original_pts1[k], rect_pts1[k]);
									std::get<0>(rect_data).convert_2_transform_right(original_pts2[k], rect_pts2[k]);

									Point rect_base_pt(rect_pts1[k][0].x/pow(2.0, pyr_lvl), rect_pts1[k][0].y/pow(2.0, pyr_lvl));
									Point rect_neigh_pt(rect_pts2[k][0].x/pow(2.0, pyr_lvl), rect_pts2[k][0].y/pow(2.0, pyr_lvl));

									if(!rectification_rect.contains(rect_base_pt) || !rectification_rect.contains(rect_neigh_pt))
									{
										/*float IndividualCost = CalculatePixelSimilarity( BaseImageProjection, ImageProjection, base_img_id, imid );
										UpdateCosts( vox_addr_list[k], IndividualCost );*/
									}
									else
									{
										float IndividualCost = CalculatePixelSimilarity( rect_base_pt, rect_neigh_pt, std::get<1>(rect_data), std::get<2>(rect_data) );
										UpdateCosts( vox_addr_list[k], IndividualCost );

										local_vis_imgs_count++;
									}

									

									//float MultiViewDepthCalculator::CalculatePixelSimilarity( const Point &a, const Point &b, const cv::Mat &img_a, const cv::Mat &img_b )
#else
									float IndividualCost = CalculatePixelSimilarity( BaseImageProjection, ImageProjection, base_img_id, imid );
									UpdateCosts( vox_addr_list[k], IndividualCost );

									local_vis_imgs_count++;
#endif
								}

#ifdef OUTPUT_VISIBLE_IMAGES_DATA
								cells(std::get<0>(vox_addr_list[k]), std::get<1>(vox_addr_list[k]))[std::get<2>(vox_addr_list[k])] = local_vis_imgs_count;
#endif
								/*	//TEST
								int xx = std::get<0>(vox_addr_list[k]);
								int yy = std::get<1>(vox_addr_list[k]);
								int zz = std::get<2>(vox_addr_list[k]);

								//if( xx == std::get<0>(GridSize)/2 && yy == std::get<1>(GridSize)/2 && zz == std::get<2>(GridSize)/2 )
								if( xx == 512 && yy == 413 && zz == 5 )
								{
									for( int imid(0); imid < NumberOfImages; ++imid )
									{
										bool VoxelIsVisibleOnImage;
										bool success = vox_shema_list[k].get_visibility( imid, VoxelIsVisibleOnImage );
										if( !VoxelIsVisibleOnImage ) continue;
										SimpleFloatPoint ImageProjection;
										success &= vox_shema_list[k].get_projection( imid, ImageProjection );
										cv::Mat Test;
										Images[imid].copyTo( Test );
										cv::circle( Test, Point(ImageProjection.x,ImageProjection.y),3,3);
										stringstream ss;
										ss << "d:\\projtest1_" <<  imid << ".jpg";
										cv::imwrite( ss.str(), Test );
									}
								}
								if( xx == 512 && yy == 413 && zz == 150 )
								{
									for( int imid(0); imid < NumberOfImages; ++imid )
									{
										bool VoxelIsVisibleOnImage;
										bool success = vox_shema_list[k].get_visibility( imid, VoxelIsVisibleOnImage );
										if( !VoxelIsVisibleOnImage ) continue;
										SimpleFloatPoint ImageProjection;
										success &= vox_shema_list[k].get_projection( imid, ImageProjection );
										cv::Mat Test;
										Images[imid].copyTo( Test );
										cv::circle( Test, Point(ImageProjection.x,ImageProjection.y),3,3);
										stringstream ss;
										ss << "d:\\projtest2_" <<  imid << ".jpg";
										cv::imwrite( ss.str(), Test );
									}
								}
								if( xx == 512 && yy == 413 && zz == 121 )
								{
									for( int imid(0); imid < NumberOfImages; ++imid )
									{
										bool VoxelIsVisibleOnImage;
										bool success = vox_shema_list[k].get_visibility( imid, VoxelIsVisibleOnImage );
										if( !VoxelIsVisibleOnImage ) continue;
										SimpleFloatPoint ImageProjection;
										success &= vox_shema_list[k].get_projection( imid, ImageProjection );
										cv::Mat Test;
										Images[imid].copyTo( Test );
										cv::circle( Test, Point(ImageProjection.x,ImageProjection.y),3,3);
										stringstream ss;
										ss << "d:\\projtest2_" <<  imid << ".jpg";
										cv::imwrite( ss.str(), Test );
									}
								}
								if( xx == 700 && yy == 620 && zz == 80 )
								{
									for( int imid(0); imid < NumberOfImages; ++imid )
									{
										bool VoxelIsVisibleOnImage;
										bool success = vox_shema_list[k].get_visibility( imid, VoxelIsVisibleOnImage );
										if( !VoxelIsVisibleOnImage ) cout << "Projection on image " << imid << " : NONE Projection " << endl;
										if( !VoxelIsVisibleOnImage ) continue;
										SimpleFloatPoint ImageProjection;
										success &= vox_shema_list[k].get_projection( imid, ImageProjection );

										cout << "Projection on image " << imid << " : " << ImageProjection.x << " , " << ImageProjection.y << endl;
										//cv::Mat Test;
										//Images[imid].copyTo( Test );
										//cv::circle( Test, Point(ImageProjection.x,ImageProjection.y),3,3);
										//stringstream ss;
										//ss << "d:\\projtest2_" <<  imid << ".jpg";
										//cv::imwrite( ss.str(), Test );
									}
								}*/

								(*sgmProgress)++;
							}
							counter+= ncores;
						}
						
					}

				}
				
			}

			
		}

#ifdef OUTPUT_VISIBLE_IMAGES_DATA
		workbook wb1;
		auto sh1 = wb1.sheet("SGM");

		for(int i = 0; i < dim1; i++)
			for(int j = 0; j < dim2; j++)
			{
				float mean_val = 0;
				for(int k = 0; k < dim3; k++)
					mean_val+= cells(i, j)[k];

				mean_val/= dim3;

				sh1->number(j, i, mean_val);
			}

		wb1.Dump("J:\\delme\\vis_data.xls");
#endif
		
		/*workbook wb;
		auto sh = wb.sheet("SGM");

		int Width = LocalCosts.size();
		int Height = LocalCosts.front().size();
		int Depth = LocalCosts.front().front().size();
		for(int i = 0; i < Width; i++)
			for(int j = 0; j < Height; j++)
			{
				float max_val = 0;
				for(int k = 0; k < Depth; k++)
					if(LocalCosts[i][j][k] > max_val)
						max_val = LocalCosts[i][j][k];

				sh->number(j, i, max_val);
			}

		wb.Dump("J:\\delme\\full_sgm.xls");

		std::system("PAUSE");*/
		

		cout << endl << "It takes " << timer.elapsedMs() << " ms" << endl;

		/*std::system("PAUSE");*/
}

void MultiViewDepthCalculator::CalculateAggregatedCosts()
{
	//DUMMY
	//AggregatedCosts = LocalCosts;

	cout << endl << "MultiViewDepthCalculator::CalculateAggregatedCosts started" << endl;
	Timer timer;
	timer.reset();

	AggregatedCosts = CalculateSGMCosts( LocalCosts );

	/*workbook wb;
	auto sh = wb.sheet("SGM");

	int Width = AggregatedCosts.size();
	int Height = AggregatedCosts.front().size();
	int Depth = AggregatedCosts.front().front().size();
	for(int i = 0; i < Width; i++)
		for(int j = 0; j < Height; j++)
		{
			float max_val = 0;
			for(int k = 0; k < Depth; k++)
				if(AggregatedCosts[i][j][k] > max_val)
					max_val = AggregatedCosts[i][j][k];

			sh->number(j, i, max_val);
		}

		wb.Dump("J:\\delme\\full_sgm.xls");

		std::system("PAUSE");*/
	
	cout << endl << "It takes " << timer.elapsedMs() << " ms" << endl;
}

void MultiViewDepthCalculator::GetDepthMap()
{
/*	int Width = AggregatedCosts.shape()[0];
	int Height = AggregatedCosts.shape()[1];
	int Depth = AggregatedCosts.shape()[2];*/
	int Width = AggregatedCosts.size();
	int Height = AggregatedCosts.front().size();
	int Depth = AggregatedCosts.front().front().size();

	
	DepthMap = cv::Mat( Height, Width, CV_32F );

	cout << "DepthMap sizes = " << DepthMap.cols << " x " << DepthMap.rows << endl;

//#pragma omp parallel for
	for( size_t y(0); y < Height; ++y)
    {
        float *const scanLine( DepthMap.ptr<float>(y) );
        for( size_t x(0); x < Width; ++x)
		{
			float MinCost = 10000000000;
			int BestZPosition = 0;
			for( size_t z(0); z < Depth; ++z )
				if( AggregatedCosts[x][y][z] < MinCost )
				{
					MinCost = AggregatedCosts[x][y][z];
					BestZPosition = z;
				}
			scanLine[x] = BestZPosition;


			//if( MinCost>1000)scanLine[x] = 0;
		}
	}


	/*for(size_t x = 0; x < Width; x++) 
	{
		float *
		for(size_t y = 0; y < Heigth; y++) 
		{

		}
	}*/
}

void MultiViewDepthCalculator::GetDepthMapSubpix()
{
	//cout << endl << "MultiViewDepthCalculator::GetDepthMapSubpix started" << endl;
	Timer timer;
	timer.reset();

	int Width = AggregatedCosts.size();
	int Height = AggregatedCosts.front().size();
	int Depth = AggregatedCosts.front().front().size();
	
	DepthMap = cv::Mat( Height, Width, CV_32FC1 );

	//cout << "DepthMap sizes = " << DepthMap.cols << " x " << DepthMap.rows << endl;

//#pragma omp parallel for
	for( size_t y(0); y < Height; ++y)
    {
        float *const scanLine( DepthMap.ptr<float>(y) );
        for( size_t x(0); x < Width; ++x)
		{
			float BestZPosition = GetInterpolatedArgmin( AggregatedCosts[x][y] );
			scanLine[x] = BestZPosition;

			//if( MinCost>1000)scanLine[x] = 0;
		}
	}
	//cout << endl << "It takes " << timer.elapsedMs() << " ms" << endl;
}


void MultiViewDepthCalculator::SaveDepthMapToBitmap( string FileName )
{
	cv::Mat NormalizedGradientMask;
	normalize( GradientMaskMat, NormalizedGradientMask, 0, 255, CV_MINMAX );
	imwrite( "d:\\delme\\GradientMask.bmp", NormalizedGradientMask );

	cv::Mat NormalizedDepthMap, NormalizedDepthMap255;
	normalize( DepthMap, NormalizedDepthMap, 0, 1, CV_MINMAX );
	//imwrite( FileName, NormalizedDepthMap );
	imshow( "NormalizedDepthMap", NormalizedDepthMap );
	normalize( DepthMap, NormalizedDepthMap255, 0, 255, CV_MINMAX );
	imwrite( FileName, NormalizedDepthMap255 );
	imshow( "NormalizedDepthMap255", NormalizedDepthMap );
}


void MultiViewDepthCalculator::SaveDepthMapToGeoTiff( string FileName, bool fill_empty_cells )
{
	int dimx = DepthMap.cols;
	int dimy = DepthMap.rows;

	if( fill_empty_cells )
	{
		const int FILL_EMPTY_CELL_VALUE = -1;
		for( size_t y(0); y < dimy; ++y)
		{
			float *const scanLine( DepthMap.ptr<float>(y) );
			for( size_t x(0); x < dimx; ++x)
				if( NumberOfProjections[x][y][scanLine[x]] == 0 )
					scanLine[x] = FILL_EMPTY_CELL_VALUE;
		}
	}

	bool success = CreateFloatTiffAndFillWithNoData( FileName, dimx, dimy, -1);

	success &= WriteBlock2FloatTiff( FileName, 0, 0, dimx, dimy, (float*)DepthMap.data );
}

int MultiViewDepthCalculator::CountEmtpyCostCells()
{
	int Count = 0;
	/*int Width = NumberOfProjections.shape()[0];
	int Height = NumberOfProjections.shape()[1];
	int Depth = NumberOfProjections.shape()[2];*/
	int Width = NumberOfProjections.size();
	int Height = NumberOfProjections.front().size();
	int Depth = NumberOfProjections.front().front().size();

//#pragma omp parallel for
	for( size_t x(0); x < Width; ++x)
        for( size_t y(0); y < Height; ++y)
			for( size_t z(0); z < Depth; ++z )
				if( NumberOfProjections[x][y][z] == 0 )	
					Count++;
	return Count;
}

int MultiViewDepthCalculator::GetTotalNumberOfVoxels()
{
	return std::get<0>(GridSize) * std::get<1>(GridSize) * std::get<2>(GridSize);
}

void MultiViewDepthCalculator::ProcessFullSceneST( string images_dir, string cache_dir, string depth_map_name )
{
	cache_dir = stlplus::folder_append_separator(cache_dir);

	const string rectification_name = "rectification.dat", left_img_name = "left.tiff", right_img_name = "right.tiff";
	
	int dim_x = std::get<0>(GridSize), dim_y = std::get<1>(GridSize), dim_z = std::get<2>(GridSize);

	CreateFloatTiffAndFillWithNoData(depth_map_name, dim_x, dim_y, -1);
	
	vector<Rect> blocks_list;
	int full_im_size_y = dim_y / max_virtual_block_size, full_im_size_x = dim_x / max_virtual_block_size;
	int last_y_size = dim_y - full_im_size_y*max_virtual_block_size, last_x_size = dim_x - full_im_size_x*max_virtual_block_size;

	for(int i = 0; i < full_im_size_y; i++)
		for(int j = 0; j < full_im_size_x; j++)
			blocks_list.push_back(Rect(j*max_virtual_block_size, i*max_virtual_block_size, max_virtual_block_size, max_virtual_block_size));

	if(last_y_size)
		for(int i = 0; i < full_im_size_x; i++)
			blocks_list.push_back(Rect(i*max_virtual_block_size, full_im_size_y*max_virtual_block_size, max_virtual_block_size, last_y_size));

	if(last_x_size)
		for(int i = 0; i < full_im_size_y; i++)
			blocks_list.push_back(Rect(full_im_size_x*max_virtual_block_size, i*max_virtual_block_size, last_x_size, max_virtual_block_size));

	if(last_y_size&&last_x_size)
		blocks_list.push_back(Rect(full_im_size_x*max_virtual_block_size, full_im_size_y*max_virtual_block_size, last_x_size, last_y_size));

	VoxelDataShema vds(Doc->_vec_imageNames.size());

	auto blocks = p_vx_manager->GetAllBlocks();


	cout << "Single threaded SGM..." << endl;
	C_Progress_display sgmProgress(blocks_list.size());

	for(int i = 0; i < blocks_list.size(); i++)
	//for each(auto &block_r in blocks_list)
	{
		auto block_r = blocks_list[i];
		
		Rect extended_rect = Rect(block_r.x - blocks_offset, block_r.y - blocks_offset, block_r.width + 2*blocks_offset, block_r.height + 2*blocks_offset) & Rect(0, 0, dim_x, dim_y);
		//cout << Rect(block_r.x - extended_rect.x, block_r.y - extended_rect.y, block_r.width, block_r.height) << endl;
		int local_dim_x = extended_rect.width, local_dim_y = extended_rect.height, local_dim_z = dim_z;
		
		//get existing base images
		map<pair<int, int>, int> base_search_container;

		VirtualVoxelBlock vvb(extended_rect.x, extended_rect.y, 0, extended_rect.width, extended_rect.height, dim_z, blocks, vds.total_data_size, p_vx_manager->GetFileHandler());

		//VoxelAdress buf_vox_addr;
		//vvb.ConvertElementsPos2VoxelAddress(11, 52377650, buf_vox_addr);

		int non_empty_voxels_counter = 0;

		//iterate over vb elements
		for(int j = 0; j < vvb.GetSubBlocksCount(); j++)
		{
			char* local_data = NULL;
			VoxelAdress begin_voxel_addr;
			int elements_count;

			//��������� ��������� �� ������ ������ �������� �� ������� 
			if(vvb.GetSubBlockByIndex(j, &local_data, begin_voxel_addr, elements_count))
			{
				for(int k = 0; k < elements_count; k++)
				{
					vds.Init(local_data + k*vds.total_data_size);

					bool non_empty = false;


					/*VoxelAdress cur_vox_addr;
					vvb.ConvertElementsPos2VoxelAddress(j, k, cur_vox_addr);

					auto obj_space_pt = Vcore->Get3DCoordByVoxelPos(std::get<0>(cur_vox_addr), std::get<1>(cur_vox_addr), std::get<2>(cur_vox_addr));*/



					int base_img_id;
					vds.get_base_img_id(base_img_id);

					if(base_img_id < 0) continue;

					//delme
					/*VoxelAdress cur_vox_addr;
					vvb.ConvertElementsPos2VoxelAddress(j, k, cur_vox_addr);

					auto obj_space_pt = Vcore->Get3DCoordByVoxelPos(std::get<0>(cur_vox_addr), std::get<1>(cur_vox_addr), std::get<2>(cur_vox_addr));

					SimpleFloatPoint a, b;

					vds.get_projection(base_img_id, a);

					auto real_projection = Doc->_map_camera[base_img_id].Project(obj_space_pt);
					b.x = real_projection.x()/pow(2.0, pyr_lvl); b.y = real_projection.y()/pow(2.0, pyr_lvl);

					cout << "a = " << a.x << " , " << a.y << endl;
					cout << "b = " << b.x << " , " << b.y <<  endl;*/

					for(int cam_counter = 0; cam_counter < Doc->_vec_imageNames.size(); cam_counter++)
					if(cam_counter != base_img_id)
					{
						bool vis;
						vds.get_visibility(cam_counter, vis);
						
						if(vis) 
						{
							//vvb.ConvertElementsPos2VoxelAddress(j, k, cur_vox_addr);

							//

							/*VoxelAdress cur_vox_addr;
							vvb.ConvertElementsPos2VoxelAddress(j, k, cur_vox_addr);

							auto obj_space_pt = Vcore->Get3DCoordByVoxelPos(std::get<0>(cur_vox_addr), std::get<1>(cur_vox_addr), std::get<2>(cur_vox_addr));

							SimpleFloatPoint a, b;

							vds.get_projection(base_img_id, a);

							auto real_projection = Doc->_map_camera[base_img_id].Project(obj_space_pt);
							b.x = real_projection.x()/pow(2.0, pyr_lvl); b.y = real_projection.y()/pow(2.0, pyr_lvl);

							cout << "a = " << a.x << " , " << a.y << endl;
							cout << "b = " << b.x << " , " << b.y <<  endl;*/

							//
							
							base_search_container[make_pair(base_img_id, cam_counter)] = 0;
							non_empty = true;
						}
					}

					if(non_empty) non_empty_voxels_counter++;
				}
			}

		}

		if(non_empty_voxels_counter <= 100) continue;

		//VirtualVoxelBlock vvb(extended_rect.x, extended_rect.y, 0, extended_rect.width, extended_rect.height, dimz, blocks, vds.total_data_size, p_vx_manager->GetFileHandler());
		vector<Vec3> block_corners_list;
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x, extended_rect.y, 0));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x + extended_rect.width - 1, extended_rect.y, 0));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x, extended_rect.y + extended_rect.height - 1, 0));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x + extended_rect.width - 1, extended_rect.y + extended_rect.height - 1, 0));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x, extended_rect.y, dim_z - 1));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x + extended_rect.width - 1, extended_rect.y, dim_z - 1));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x, extended_rect.y + extended_rect.height - 1, dim_z - 1));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x + extended_rect.width - 1, extended_rect.y + extended_rect.height - 1, dim_z - 1));

		//erase repeated pairs
		for(auto it = base_search_container.begin(); it != base_search_container.end();)
			if(base_search_container.find(make_pair(it->first.second, it->first.first)) != base_search_container.end()) base_search_container.erase(it++);
			else ++it;
		
		//for each elm in base_search_container make rectification & save into bin rect data
		//if exists load from file
		vector<Rect_<float>> image_rects_f_list;
		vector<Rect> image_rects_list;
		for each(auto img_name in Doc->_vec_imageNames)
		{
			auto full_img_name = images_dir + img_name;
			auto full_size = GetImageSize(full_img_name);
			image_rects_f_list.push_back(Rect_<float>(0, 0, full_size.width, full_size.height));
			image_rects_list.push_back(Rect(0, 0, full_size.width, full_size.height));
		}

		map<pair<int, int>, std::tuple<cv::Mat, Rect, cv::Mat, Rect, FusielloRectification>> rectified_pairs;

#ifdef ENABLE_RECTIFICATION
		for(auto it = base_search_container.begin(); it != base_search_container.end(); it++)
		{
			int id0 = it->first.first, id1 = it->first.second;
			string dir1 = cache_dir + to_string(id0) + "_" + to_string(id1) + "\\", dir2 = cache_dir + to_string(id1) + "_" + to_string(id0) + "\\";

			if(stlplus::folder_exists(dir1) || stlplus::folder_exists(dir2))
			{
				//load data
				//swap if necessary
				bool swap = stlplus::folder_exists(dir2);
				string main_dir = swap ? dir2 : dir1;
				string rectification_data_file_name = main_dir + rectification_name;

				FusielloRectification rectification;
				rectification.Load(rectification_data_file_name);
				

				auto rect_patch_data = LoadBlocksFormRectifiedImages(block_corners_list, rectification, main_dir + left_img_name, main_dir + right_img_name, Doc, id0, id1, pyr_lvl);

				if(swap) 
				{
					rectification.swap();
					auto buf_img = std::get<0>(rect_patch_data);
					std::get<0>(rect_patch_data) = std::get<2>(rect_patch_data);
					std::get<2>(rect_patch_data)= buf_img;

					auto buf_rect = std::get<1>(rect_patch_data);
					std::get<1>(rect_patch_data) = std::get<3>(rect_patch_data);
					std::get<3>(rect_patch_data)= buf_rect;
				}

				rectified_pairs[make_pair(id0, id1)] = make_tuple(std::get<0>(rect_patch_data), std::get<1>(rect_patch_data), std::get<2>(rect_patch_data), std::get<3>(rect_patch_data), rectification);

				auto reverse_rectification = rectification;
				reverse_rectification.swap();

				rectified_pairs[make_pair(id1, id0)] = make_tuple(std::get<2>(rect_patch_data), std::get<3>(rect_patch_data), std::get<0>(rect_patch_data), std::get<1>(rect_patch_data), reverse_rectification);
			}
			else
			{
				stlplus::folder_create(dir1);
				string inp_img_name_1 = images_dir + Doc->_vec_imageNames[id0], inp_img_name_2 = images_dir + Doc->_vec_imageNames[id1];
				string out_img_name_1 = dir1 + left_img_name, out_img_name_2 = dir1 + right_img_name;
				string rectification_data_name = dir1 + rectification_name;

				//create rectification
				cv::Mat Po1, Po2;
				eigen2cv(Doc->_map_camera[id0]._P, Po1);
				eigen2cv(Doc->_map_camera[id1]._P, Po2);

				FusielloRectification rectification(Po1, Po2);
				rectification.wrap_left(inp_img_name_1, out_img_name_1);
				rectification.wrap_right(inp_img_name_2, out_img_name_2);

				rectification.Write(rectification_data_name);

				//load data into memory
				auto rect_patch_data = LoadBlocksFormRectifiedImages(block_corners_list, rectification, dir1 + left_img_name, dir1 + right_img_name, Doc, id0, id1, pyr_lvl);

				rectified_pairs[make_pair(id0, id1)] = make_tuple(std::get<0>(rect_patch_data), std::get<1>(rect_patch_data), std::get<2>(rect_patch_data), std::get<3>(rect_patch_data), rectification);
			
				auto reverse_rectification = rectification;
				reverse_rectification.swap();

				rectified_pairs[make_pair(id1, id0)] = make_tuple(std::get<2>(rect_patch_data), std::get<3>(rect_patch_data), std::get<0>(rect_patch_data), std::get<1>(rect_patch_data), reverse_rectification);
			}

		}
#else
		for(auto it = base_search_container.begin(); it != base_search_container.end(); it++)
		{
			const int local_offset = 5;
			
			//load rects from appropriate pyr lvls
			//get appropriate rects
			int id0 = it->first.first, id1 = it->first.second;
			string inp_img_name_1 = images_dir + Doc->_vec_imageNames[id0], inp_img_name_2 = images_dir + Doc->_vec_imageNames[id1];

#ifdef USE_FULLIMAGES
			cv::Mat sub_img1 = CensusTransform5x5(LoadGrayScaleImageCV(inp_img_name_1, pyr_lvl)), sub_img2 = CensusTransform5x5(LoadGrayScaleImageCV(inp_img_name_2, pyr_lvl));

			Rect sub_rect1(image_rects_list[id0].x/pow(2.0, pyr_lvl), image_rects_list[id0].y/pow(2.0, pyr_lvl), image_rects_list[id0].width/pow(2.0, pyr_lvl), image_rects_list[id0].height/pow(2.0, pyr_lvl));
			Rect sub_rect2(image_rects_list[id1].x/pow(2.0, pyr_lvl), image_rects_list[id1].y/pow(2.0, pyr_lvl), image_rects_list[id1].width/pow(2.0, pyr_lvl), image_rects_list[id1].height/pow(2.0, pyr_lvl));

			rectified_pairs[make_pair(id0, id1)] = make_tuple(sub_img1, sub_rect1, sub_img2, sub_rect2, FusielloRectification());
			rectified_pairs[make_pair(id1, id0)] = make_tuple(sub_img2, sub_rect2, sub_img1, sub_rect1, FusielloRectification());
#else
			vector<Vec2> original_img_projection_list_1, original_img_projection_list_2;
			for each(auto &pt in block_corners_list)
			{
				original_img_projection_list_1.push_back(Doc->_map_camera[id0].Project(pt));
				original_img_projection_list_2.push_back(Doc->_map_camera[id1].Project(pt));
			}

			Vec2 min_pt_1(original_img_projection_list_1[0]), max_pt_1(original_img_projection_list_1[0]), min_pt_2(original_img_projection_list_2[0]), max_pt_2(original_img_projection_list_2[0]);

			for each(auto &pt in original_img_projection_list_1)
			{
				if(pt.x() < min_pt_1.x()) min_pt_1.x() = pt.x();
				if(pt.y() < min_pt_1.y()) min_pt_1.y() = pt.y();

				if(pt.x() > max_pt_1.x()) max_pt_1.x() = pt.x();
				if(pt.y() > max_pt_1.y()) max_pt_1.y() = pt.y();
			}

			for each(auto &pt in original_img_projection_list_2)
			{
				if(pt.x() < min_pt_2.x()) min_pt_2.x() = pt.x();
				if(pt.y() < min_pt_2.y()) min_pt_2.y() = pt.y();

				if(pt.x() > max_pt_2.x()) max_pt_2.x() = pt.x();
				if(pt.y() > max_pt_2.y()) max_pt_2.y() = pt.y();
			}

			Rect rect_1 = Rect(cvFloor(min_pt_1.x()) - local_offset, cvFloor(min_pt_1.y()) - local_offset, cvCeil(max_pt_1.x()) - cvFloor(min_pt_1.x()) + 2*local_offset + 1, cvCeil(max_pt_1.y()) - cvFloor(min_pt_1.y()) + 2*local_offset + 1) & image_rects_list[id0];
			Rect rect_2 = Rect(cvFloor(min_pt_2.x()) - local_offset, cvFloor(min_pt_2.y()) - local_offset, cvCeil(max_pt_2.x()) - cvFloor(min_pt_2.x()) + 2*local_offset + 1, cvCeil(max_pt_2.y()) - cvFloor(min_pt_2.y())+ 2*local_offset + 1) & image_rects_list[id1];

			cv::Mat sub_img1 = CensusTransform5x5(LoadGrayScaleImageCV(inp_img_name_1, pyr_lvl, rect_1)), sub_img2 = CensusTransform5x5(LoadGrayScaleImageCV(inp_img_name_2, pyr_lvl, rect_2));
			rect_1.x/= pow(2.0, pyr_lvl); rect_1.y/= pow(2.0, pyr_lvl); rect_1.width/= pow(2.0, pyr_lvl); rect_1.height/= pow(2.0, pyr_lvl);
			rect_2.x/= pow(2.0, pyr_lvl); rect_2.y/= pow(2.0, pyr_lvl); rect_2.width/= pow(2.0, pyr_lvl); rect_2.height/= pow(2.0, pyr_lvl);

			rectified_pairs[make_pair(id0, id1)] = make_tuple(sub_img1, rect_1, sub_img2, rect_2, FusielloRectification());
			rectified_pairs[make_pair(id1, id0)] = make_tuple(sub_img2, rect_2, sub_img1, rect_1, FusielloRectification());
#endif
		}
#endif

		/*for(auto it = rectified_pairs.begin(); it != rectified_pairs.end(); it++)
		{
			//cout << "patch img size = " << std::get<0>(it->second).size() << endl;
			//cout << "patch img size = " << std::get<2>(it->second).size() << endl;
		}*/

		//std::system("PAUSE");
		
		//iterate all voxels, recalc each visivle projection into new plane, check if its inside appropriate rect, get distance, update costs
		CostsCalculator local_costs_calculator(local_dim_x, local_dim_y, local_dim_z, SGM_NULL_VALUE );

		/*cout << "Arrays initialized" << endl;	
		cout << "LocalCosts sizes = " << LocalCosts.size() << " " << LocalCosts.front().size() << " " << LocalCosts.front().front().size() << endl;
		cout << "NumberOfProjections sizes = " << NumberOfProjections.size() << " " << NumberOfProjections.front().size() << " " << NumberOfProjections.front().front().size() << endl;*/

		unsigned int ncores = std::thread::hardware_concurrency();

		//cout << "ncores = " << ncores << endl;

		//cv::Mat local_mask = cv::Mat::zeros(local_dim_y, local_dim_x, CV_8UC1);

		//iterate over vb elements
#pragma omp parallel for
		for(int j = 0; j < vvb.GetSubBlocksCount(); j++)
		{
			char* local_data = NULL;
			VoxelAdress begin_voxel_addr, cur_vox_addr;
			int elements_count;
			VoxelDataShema cur_vds(Doc->_vec_imageNames.size());

			vector<Point2f> base_pts_list(1), neighbor_pts_list(1), rect_base_pts_list(1), rect_neighbor_pts_list(1);
			Point buf_base_pt, buf_neighbor_pt;
			float local_cost;

			if(vvb.GetSubBlockByIndex(j, &local_data, begin_voxel_addr, elements_count))
			{
				for(int k = 0; k < elements_count; k++)
				{
					vvb.ConvertElementsPos2VoxelAddress(j, k, cur_vox_addr);
					std::get<0>(cur_vox_addr)-= extended_rect.x;
					std::get<1>(cur_vox_addr)-= extended_rect.y;
					
					cur_vds.Init(local_data + k*cur_vds.total_data_size);

					SimpleFloatPoint BaseImageProjection;
					int base_img_id;
					cur_vds.get_base_img_id(base_img_id);

					if(base_img_id < 0) continue;

					cur_vds.get_projection(base_img_id, BaseImageProjection);

#ifdef ENABLE_RECTIFICATION
					base_pts_list[0] = Point2f(BaseImageProjection.x*pow(2.0, pyr_lvl), BaseImageProjection.y*pow(2.0, pyr_lvl));

					if(!image_rects_f_list[base_img_id].contains(base_pts_list[0])) continue;
#else
					base_pts_list[0].x = BaseImageProjection.x;
					base_pts_list[0].y = BaseImageProjection.y;

					if(!image_rects_list[base_img_id].contains(base_pts_list[0])) continue;
#endif

					//������ �� ��� ������� �������
					for(int cam_counter = 0; cam_counter < Doc->_vec_imageNames.size(); cam_counter++)
						if(cam_counter != base_img_id)
						{
							bool vis;
							cur_vds.get_visibility(cam_counter, vis);

							if(vis) 
							{
								auto &rect_patch = rectified_pairs[make_pair(base_img_id, cam_counter)];

								SimpleFloatPoint NeighborImageProjection;
								cur_vds.get_projection(cam_counter, NeighborImageProjection);

#ifdef ENABLE_RECTIFICATION	
								neighbor_pts_list[0] = Point2f(NeighborImageProjection.x*pow(2.0, pyr_lvl), NeighborImageProjection.y*pow(2.0, pyr_lvl));

								if(!image_rects_f_list[cam_counter].contains(neighbor_pts_list[0])) continue;

								//convert 2 rectified images
								std::get<4>(rect_patch).convert_2_transform_left(base_pts_list, rect_base_pts_list);
								std::get<4>(rect_patch).convert_2_transform_right(neighbor_pts_list, rect_neighbor_pts_list);

								rect_base_pts_list[0].x/= pow(2.0, pyr_lvl); rect_base_pts_list[0].y/= pow(2.0, pyr_lvl);
								rect_neighbor_pts_list[0].x/= pow(2.0, pyr_lvl); rect_neighbor_pts_list[0].y/= pow(2.0, pyr_lvl);

								//check if pts inside rect

								buf_base_pt = rect_base_pts_list[0]; buf_neighbor_pt = rect_neighbor_pts_list[0];
								if(std::get<1>(rect_patch).contains(buf_base_pt) && std::get<3>(rect_patch).contains(buf_neighbor_pt))
								{
									buf_base_pt.x-= std::get<1>(rect_patch).x; buf_base_pt.y-= std::get<1>(rect_patch).y;

									buf_neighbor_pt.x-= std::get<3>(rect_patch).x; buf_neighbor_pt.y-= std::get<3>(rect_patch).y;

									local_cost = CalculatePixelSimilarity(buf_base_pt, buf_neighbor_pt, std::get<0>(rect_patch), std::get<2>(rect_patch));

									local_costs_calculator.UpdateCosts( cur_vox_addr, local_cost );
								}
#else
								neighbor_pts_list[0].x = NeighborImageProjection.x;
								neighbor_pts_list[0].y = NeighborImageProjection.y;
								if(!image_rects_list[cam_counter].contains(neighbor_pts_list[0])) continue;

								buf_base_pt = base_pts_list[0]; buf_neighbor_pt = neighbor_pts_list[0];
#ifdef USE_FULLIMAGES

#else					
								buf_base_pt.x-= std::get<1>(rect_patch).x; buf_base_pt.y-= std::get<1>(rect_patch).y;

								buf_neighbor_pt.x-= std::get<3>(rect_patch).x; buf_neighbor_pt.y-= std::get<3>(rect_patch).y;
#endif
								
								//local_cost = CalculatePixelSimilarity(BaseImageProjection, NeighborImageProjection, std::get<0>(rect_patch), std::get<2>(rect_patch));

								local_cost = CalculatePixelSimilarity(buf_base_pt, buf_neighbor_pt, std::get<0>(rect_patch), std::get<2>(rect_patch));

								local_costs_calculator.UpdateCosts( cur_vox_addr, local_cost );
#endif

							}
						}

				}
			}
		}
		//cout << "Aggregate costs..." << endl;
		auto ag_costs = CalculateSGMCosts( local_costs_calculator.LocalCosts );

		//cout << "Calc sgm..." << endl;
		auto depth_map = GetDepthMapSuPix(ag_costs);

		//save depthmap to appropriate rect
		//FILL_EMPTY_CELL_VALUE

		//fill empty cells
		const int FILL_EMPTY_CELL_VALUE = -1;
		for( size_t y(0); y < depth_map.rows; ++y)
		{
			float *const scanLine( depth_map.ptr<float>(y) );
			for( size_t x(0); x < depth_map.cols; ++x)
				if( local_costs_calculator.NumberOfProjections[x][y][scanLine[x]] == 0 )
					scanLine[x] = FILL_EMPTY_CELL_VALUE;
		}

		//SaveDepthMapToBitmap("J:\\delme\\depth_map.bmp");

		cv::Mat trunc_depth_map = depth_map(Rect(block_r.x - extended_rect.x, block_r.y - extended_rect.y, block_r.width, block_r.height)).clone();

		WriteBlock2FloatTiff(depth_map_name, block_r.x, block_r.y, block_r.width, block_r.height, (float*)trunc_depth_map.data);
		//WriteBlock2FloatTiff(depth_map_name, extended_rect.x, extended_rect.y, extended_rect.width, extended_rect.height, (float*)DepthMap.data);


		sgmProgress++;
		//std::system("PAUSE");
	}
}

void MultiViewDepthCalculator::ProcessFullSceneMT( string images_dir, string cache_dir, string depth_map_name )
{
	const string rectification_name = "rectification.dat", left_img_name = "left.tiff", right_img_name = "right.tiff";
	
	VoxelDataShema vds(Doc->_vec_imageNames.size());
	auto ncores = std::thread::hardware_concurrency();

	//calc total memory - 1Gb (4 rectification usage), get threads count, get possible memory for thread

	MEMORYSTATUSEX statex;

	statex.dwLength = sizeof (statex);

	GlobalMemoryStatusEx(&statex);

	int64 free_mem  = (int64)statex.ullAvailPhys*0.9 - ((int64)1024)*((int64)1024)*((int64)1024);
	int64 mem_per_core = ((int64)free_mem)/ncores;

	int dim_x = std::get<0>(GridSize), dim_y = std::get<1>(GridSize), dim_z = std::get<2>(GridSize);

	int64 total_voxel_count = ((int64)dim_x)*((int64)dim_y)*((int64)dim_z);
	int64 total_voxel_memory = ((int64)total_voxel_count)*vds.total_data_size;

	int max_block_size = cvRound(sqrt(mem_per_core/(dim_z*vds.total_data_size)));

	CreateFloatTiffAndFillWithNoData(depth_map_name, dim_x, dim_y, -1);

	vector<Rect> blocks_list;
	int full_im_size_y = dim_y / max_block_size, full_im_size_x = dim_x / max_block_size;
	int last_y_size = dim_y - full_im_size_y*max_block_size, last_x_size = dim_x - full_im_size_x*max_block_size;

	for(int i = 0; i < full_im_size_y; i++)
		for(int j = 0; j < full_im_size_x; j++)
			blocks_list.push_back(Rect(j*max_block_size, i*max_block_size, max_block_size, max_block_size));

	if(last_y_size)
		for(int i = 0; i < full_im_size_x; i++)
			blocks_list.push_back(Rect(i*max_block_size, full_im_size_y*max_block_size, max_block_size, last_y_size));

	if(last_x_size)
		for(int i = 0; i < full_im_size_y; i++)
			blocks_list.push_back(Rect(full_im_size_x*max_block_size, i*max_block_size, last_x_size, max_block_size));

	if(last_y_size&&last_x_size)
		blocks_list.push_back(Rect(full_im_size_x*max_block_size, full_im_size_y*max_block_size, last_x_size, last_y_size));

	cout << "Multi threaded SGM..." << endl;
	C_Progress_display sgmProgress(blocks_list.size());

	auto blocks = p_vx_manager->GetAllBlocks();

#pragma omp parallel for
	for(int i = 0; i < blocks_list.size(); i++)
	{
		auto block_r = blocks_list[i];
		
		Rect extended_rect = Rect(block_r.x - blocks_offset, block_r.y - blocks_offset, block_r.width + 2*blocks_offset, block_r.height + 2*blocks_offset) & Rect(0, 0, dim_x, dim_y);
		//cout << Rect(block_r.x - extended_rect.x, block_r.y - extended_rect.y, block_r.width, block_r.height) << endl;
		int local_dim_x = extended_rect.width, local_dim_y = extended_rect.height, local_dim_z = dim_z;

		//get existing base images
		map<pair<int, int>, int> base_search_container;

		std::shared_ptr<VirtualVoxelBlock> vvb;

#pragma omp critical
		{
			vvb = std::shared_ptr<VirtualVoxelBlock>(new VirtualVoxelBlock(extended_rect.x, extended_rect.y, 0, extended_rect.width, extended_rect.height, dim_z, blocks, vds.total_data_size, p_vx_manager->GetFileHandler()));
		}
		
		int non_empty_voxels_counter = 0;

		//iterate over vb elements
		for(int j = 0; j < vvb->GetSubBlocksCount(); j++)
		{
			char* local_data = NULL;
			VoxelAdress begin_voxel_addr;
			int elements_count;

			//��������� ��������� �� ������ ������ �������� �� ������� 
			if(vvb->GetSubBlockByIndex(j, &local_data, begin_voxel_addr, elements_count))
			{
				for(int k = 0; k < elements_count; k++)
				{
					vds.Init(local_data + k*vds.total_data_size);

					bool non_empty = false;

					int base_img_id;
					vds.get_base_img_id(base_img_id);

					if(base_img_id < 0) continue;

					for(int cam_counter = 0; cam_counter < Doc->_vec_imageNames.size(); cam_counter++)
					if(cam_counter != base_img_id)
					{
						bool vis;
						vds.get_visibility(cam_counter, vis);
						
						if(vis) 
						{						
							base_search_container[make_pair(base_img_id, cam_counter)] = 0;
							non_empty = true;
						}
					}

					if(non_empty) non_empty_voxels_counter++;
				}
			}

		}

		if(non_empty_voxels_counter <= 100) continue;

		vector<Vec3> block_corners_list;
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x, extended_rect.y, 0));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x + extended_rect.width - 1, extended_rect.y, 0));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x, extended_rect.y + extended_rect.height - 1, 0));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x + extended_rect.width - 1, extended_rect.y + extended_rect.height - 1, 0));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x, extended_rect.y, dim_z - 1));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x + extended_rect.width - 1, extended_rect.y, dim_z - 1));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x, extended_rect.y + extended_rect.height - 1, dim_z - 1));
		block_corners_list.push_back(Vcore->Get3DCoordByVoxelPos(extended_rect.x + extended_rect.width - 1, extended_rect.y + extended_rect.height - 1, dim_z - 1));

		//erase repeated pairs
		for(auto it = base_search_container.begin(); it != base_search_container.end();)
			if(base_search_container.find(make_pair(it->first.second, it->first.first)) != base_search_container.end()) base_search_container.erase(it++);
			else ++it;

		//for each elm in base_search_container make rectification & save into bin rect data
		//if exists load from file
		vector<Rect_<float>> image_rects_f_list;
		vector<Rect> image_rects_list;
		for each(auto img_name in Doc->_vec_imageNames)
			{
				auto full_img_name = images_dir + img_name;
				auto full_size = GetImageSize(full_img_name);
				image_rects_f_list.push_back(Rect_<float>(0, 0, full_size.width, full_size.height));
				image_rects_list.push_back(Rect(0, 0, full_size.width, full_size.height));
			}

		map<pair<int, int>, std::tuple<cv::Mat, Rect, cv::Mat, Rect, FusielloRectification>> rectified_pairs;

#pragma omp critical
		{
			for(auto it = base_search_container.begin(); it != base_search_container.end(); it++)
			{
				int id0 = it->first.first, id1 = it->first.second;
				string dir1 = cache_dir + to_string(id0) + "_" + to_string(id1) + "\\", dir2 = cache_dir + to_string(id1) + "_" + to_string(id0) + "\\";

				if(stlplus::folder_exists(dir1) || stlplus::folder_exists(dir2))
				{
					//load data
					//swap if necessary
					bool swap = stlplus::folder_exists(dir2);
					string main_dir = swap ? dir2 : dir1;
					string rectification_data_file_name = main_dir + rectification_name;

					FusielloRectification rectification;
					rectification.Load(rectification_data_file_name);


					auto rect_patch_data = LoadBlocksFormRectifiedImages(block_corners_list, rectification, main_dir + left_img_name, main_dir + right_img_name, Doc, id0, id1, pyr_lvl);

					if(swap) 
					{
						rectification.swap();
						auto buf_img = std::get<0>(rect_patch_data);
						std::get<0>(rect_patch_data) = std::get<2>(rect_patch_data);
						std::get<2>(rect_patch_data)= buf_img;

						auto buf_rect = std::get<1>(rect_patch_data);
						std::get<1>(rect_patch_data) = std::get<3>(rect_patch_data);
						std::get<3>(rect_patch_data)= buf_rect;
					}

					rectified_pairs[make_pair(id0, id1)] = make_tuple(std::get<0>(rect_patch_data), std::get<1>(rect_patch_data), std::get<2>(rect_patch_data), std::get<3>(rect_patch_data), rectification);

					auto reverse_rectification = rectification;
					reverse_rectification.swap();

					rectified_pairs[make_pair(id1, id0)] = make_tuple(std::get<2>(rect_patch_data), std::get<3>(rect_patch_data), std::get<0>(rect_patch_data), std::get<1>(rect_patch_data), reverse_rectification);
				}
				else
				{
					stlplus::folder_create(dir1);
					string inp_img_name_1 = images_dir + Doc->_vec_imageNames[id0], inp_img_name_2 = images_dir + Doc->_vec_imageNames[id1];
					string out_img_name_1 = dir1 + left_img_name, out_img_name_2 = dir1 + right_img_name;
					string rectification_data_name = dir1 + rectification_name;

					//create rectification
					cv::Mat Po1, Po2;
					eigen2cv(Doc->_map_camera[id0]._P, Po1);
					eigen2cv(Doc->_map_camera[id1]._P, Po2);

					FusielloRectification rectification(Po1, Po2);
					rectification.wrap_left(inp_img_name_1, out_img_name_1);
					rectification.wrap_right(inp_img_name_2, out_img_name_2);

					rectification.Write(rectification_data_name);

					//load data into memory
					auto rect_patch_data = LoadBlocksFormRectifiedImages(block_corners_list, rectification, dir1 + left_img_name, dir1 + right_img_name, Doc, id0, id1, pyr_lvl);

					rectified_pairs[make_pair(id0, id1)] = make_tuple(std::get<0>(rect_patch_data), std::get<1>(rect_patch_data), std::get<2>(rect_patch_data), std::get<3>(rect_patch_data), rectification);

					auto reverse_rectification = rectification;
					reverse_rectification.swap();

					rectified_pairs[make_pair(id1, id0)] = make_tuple(std::get<2>(rect_patch_data), std::get<3>(rect_patch_data), std::get<0>(rect_patch_data), std::get<1>(rect_patch_data), reverse_rectification);
				}

			}
		}

		CostsCalculator local_costs_calculator(local_dim_x, local_dim_y, local_dim_z, SGM_NULL_VALUE);

		for(int j = 0; j < vvb->GetSubBlocksCount(); j++)
		{
			char* local_data = NULL;
			VoxelAdress begin_voxel_addr, cur_vox_addr;
			int elements_count;
			VoxelDataShema cur_vds(Doc->_vec_imageNames.size());

			vector<Point2f> base_pts_list(1), neighbor_pts_list(1), rect_base_pts_list(1), rect_neighbor_pts_list(1);
			Point buf_base_pt, buf_neighbor_pt;
			float local_cost;

			if(vvb->GetSubBlockByIndex(j, &local_data, begin_voxel_addr, elements_count))
			{
				for(int k = 0; k < elements_count; k++)
				{
					vvb->ConvertElementsPos2VoxelAddress(j, k, cur_vox_addr);
					std::get<0>(cur_vox_addr)-= extended_rect.x;
					std::get<1>(cur_vox_addr)-= extended_rect.y;

					cur_vds.Init(local_data + k*cur_vds.total_data_size);

					SimpleFloatPoint BaseImageProjection;
					int base_img_id;
					cur_vds.get_base_img_id(base_img_id);

					if(base_img_id < 0) continue;

					cur_vds.get_projection(base_img_id, BaseImageProjection);

					base_pts_list[0] = Point2f(BaseImageProjection.x*pow(2.0, pyr_lvl), BaseImageProjection.y*pow(2.0, pyr_lvl));

					if(!image_rects_f_list[base_img_id].contains(base_pts_list[0])) continue;

					//������ �� ��� ������� �������
					for(int cam_counter = 0; cam_counter < Doc->_vec_imageNames.size(); cam_counter++)
						if(cam_counter != base_img_id)
						{
							bool vis;
							cur_vds.get_visibility(cam_counter, vis);

							if(vis) 
							{
								auto &rect_patch = rectified_pairs[make_pair(base_img_id, cam_counter)];

								SimpleFloatPoint NeighborImageProjection;
								cur_vds.get_projection(cam_counter, NeighborImageProjection);

								neighbor_pts_list[0] = Point2f(NeighborImageProjection.x*pow(2.0, pyr_lvl), NeighborImageProjection.y*pow(2.0, pyr_lvl));

								if(!image_rects_f_list[cam_counter].contains(neighbor_pts_list[0])) continue;

								//convert 2 rectified images
								std::get<4>(rect_patch).convert_2_transform_left(base_pts_list, rect_base_pts_list);
								std::get<4>(rect_patch).convert_2_transform_right(neighbor_pts_list, rect_neighbor_pts_list);

								rect_base_pts_list[0].x/= pow(2.0, pyr_lvl); rect_base_pts_list[0].y/= pow(2.0, pyr_lvl);
								rect_neighbor_pts_list[0].x/= pow(2.0, pyr_lvl); rect_neighbor_pts_list[0].y/= pow(2.0, pyr_lvl);

								//check if pts inside rect

								buf_base_pt = rect_base_pts_list[0]; buf_neighbor_pt = rect_neighbor_pts_list[0];
								if(std::get<1>(rect_patch).contains(buf_base_pt) && std::get<3>(rect_patch).contains(buf_neighbor_pt))
								{
									buf_base_pt.x-= std::get<1>(rect_patch).x; buf_base_pt.y-= std::get<1>(rect_patch).y;

									buf_neighbor_pt.x-= std::get<3>(rect_patch).x; buf_neighbor_pt.y-= std::get<3>(rect_patch).y;

									local_cost = CalculatePixelSimilarity(buf_base_pt, buf_neighbor_pt, std::get<0>(rect_patch), std::get<2>(rect_patch));

									local_costs_calculator.UpdateCosts( cur_vox_addr, local_cost );
								}

							}
						}

				}
			}
		}

		//no need in virtual block data
		vvb.reset();

		auto ag_costs = CalculateSGMCosts( local_costs_calculator.LocalCosts );

		//cout << "Calc sgm..." << endl;
		auto depth_map = GetDepthMapSuPix(ag_costs);

		//save depthmap to appropriate rect
		//FILL_EMPTY_CELL_VALUE

		//fill empty cells
		const int FILL_EMPTY_CELL_VALUE = -1;
		for( size_t y(0); y < depth_map.rows; ++y)
		{
			float *const scanLine( depth_map.ptr<float>(y) );
			for( size_t x(0); x < depth_map.cols; ++x)
				if( local_costs_calculator.NumberOfProjections[x][y][scanLine[x]] == 0 )
					scanLine[x] = FILL_EMPTY_CELL_VALUE;
		}

		//SaveDepthMapToBitmap("J:\\delme\\depth_map.bmp");

		cv::Mat trunc_depth_map = depth_map(Rect(block_r.x - extended_rect.x, block_r.y - extended_rect.y, block_r.width, block_r.height)).clone();

#pragma omp critical
		{
			WriteBlock2FloatTiff(depth_map_name, block_r.x, block_r.y, block_r.width, block_r.height, (float*)trunc_depth_map.data);

			sgmProgress++;
		}
		
	}
}

void MultiViewDepthCalculator::ProcessFullSceneII( int min_pixel_projection_size, string images_dir, string cache_dir, string depth_map_name, string xyzrgb_name, bool use_coarse_dtm, string coarse_dtm_filename)
{
	const string rectification_name = "rectification.dat", left_img_name = "left.tiff", right_img_name = "right.tiff", full_disp_name = cache_dir + "full_disp.tiff", trunc_disp_name = cache_dir + "trunc_disp.tiff";
	const string orthophoto_name = cache_dir + "ortho.tiff";

	if(stlplus::file_exists(xyzrgb_name) && xyzrgb_name!="")
		stlplus::file_delete(xyzrgb_name);

	if(!Doc) return;

	double voxel_step;
	Vec3 min_pt, max_pt;
	int dimx, dimy, dimz;

	if(!CalcVoxelGridMetaData(Doc, min_pixel_projection_size, pyr_lvl, voxel_step, min_pt, max_pt, dimx, dimy, dimz))
	{
		cout << "Error creating voxel grid!!!" << endl;
		return;
	}

	if(use_coarse_dtm && !stlplus::file_exists(coarse_dtm_filename))
	{
		cout << "Coarse dtm doesnt exists" << endl;
		return;
	}

	GISHeader coarse_dtm_header = use_coarse_dtm?GetGISHeader(coarse_dtm_filename):GISHeader();

	cout << "Grid size = " << dimx << " x " << dimy << " x " << dimz << endl << endl;	

	//��������� ���������� ������� �� ��������� �����
#ifdef USE_FLOAT_DISPARITY
	CreateFloatTiffAndFillWithNoData(full_disp_name, dimx, dimy, -1);
	CreateFloatTiffAndFillWithNoData(trunc_disp_name, dimx, dimy, -1);
#else
	CreateIntTiffAndFillWithNoData(full_disp_name, dimx, dimy, -1);
	CreateIntTiffAndFillWithNoData(trunc_disp_name, dimx, dimy, -1);
#endif

	//create gis header
	GISHeader depth_map_header;
	depth_map_header.dx = depth_map_header.dy = voxel_step;
	depth_map_header.north = min_pt.y();
	depth_map_header.west = min_pt.x();

	CreateFloatTiffAndFillWithNoData(depth_map_name, dimx, dimy, DEPTH_MAP_NODATA_VALUE, depth_map_header);

	CreateColorRasterAndFillWithData(orthophoto_name, dimx, dimy);

#ifdef DEBUG_OUTPUT
	string mean_num_file_name = cache_dir + "mean_num.tiff";

	CreateFloatTiffAndFillWithNoData(mean_num_file_name, dimx, dimy, 0);
#endif

	const int cleaning_offset = 60;
	Rect inner_rect(cleaning_offset, cleaning_offset, dimx - 2*cleaning_offset, dimy - 2*cleaning_offset);

	vector<Rect> blocks_list;
	int full_im_size_y = dimy / max_virtual_block_size, full_im_size_x = dimx / max_virtual_block_size;
	int last_y_size = dimy - full_im_size_y*max_virtual_block_size, last_x_size = dimx - full_im_size_x*max_virtual_block_size;

	for(int i = 0; i < full_im_size_y; i++)
		for(int j = 0; j < full_im_size_x; j++)
			blocks_list.push_back(Rect(j*max_virtual_block_size, i*max_virtual_block_size, max_virtual_block_size, max_virtual_block_size));

	if(last_y_size)
		for(int i = 0; i < full_im_size_x; i++)
			blocks_list.push_back(Rect(i*max_virtual_block_size, full_im_size_y*max_virtual_block_size, max_virtual_block_size, last_y_size));

	if(last_x_size)
		for(int i = 0; i < full_im_size_y; i++)
			blocks_list.push_back(Rect(full_im_size_x*max_virtual_block_size, i*max_virtual_block_size, last_x_size, max_virtual_block_size));

	if(last_y_size&&last_x_size)
		blocks_list.push_back(Rect(full_im_size_x*max_virtual_block_size, full_im_size_y*max_virtual_block_size, last_x_size, last_y_size));

	vector<float> max_angles_list;

	for(int i = 0; i < Doc->_vec_imageNames.size(); i++)
	{
		auto fovy = 2*atan(Doc->_map_imageSize[i].second/(Doc->_map_camera[i]._K(1, 1)));
		auto fovy_deg = 180*fovy/boost::math::constants::pi<float>();

		auto fovx = 2*atan(Doc->_map_imageSize[i].first/(Doc->_map_camera[i]._K(0, 0)));
		auto fovx_deg = 180*fovy/boost::math::constants::pi<float>();

		max_angles_list.push_back(max(fovx, fovy)/2);
	}

	vector<Rect> original_images_rects;
	for each(auto img_name in Doc->_vec_imageNames)
	{
		string full_img_name = images_dir + img_name;

		auto cur_size = GetImageSize(full_img_name);

		original_images_rects.push_back(Rect(0, 0, cur_size.width, cur_size.height));
	}

	cout << "Cacheless SGM..." << endl << "blocks count = " << blocks_list.size() << endl;
	C_Progress_display sgmProgress(blocks_list.size());

#pragma omp parallel for
	for(int block_num = 0; block_num < blocks_list.size(); block_num++)
	{
		auto block = blocks_list[block_num];

		Rect extended_rect = Rect(block.x - blocks_offset, block.y - blocks_offset, block.width + 2*blocks_offset, block.height + 2*blocks_offset) & Rect(0, 0, dimx, dimy);

		vector<float> local_angles_list(Doc->_map_camera.size());

		CostsCalculator local_costs_calculator(extended_rect.width, extended_rect.height, dimz, SGM_NULL_VALUE);

		//key (img_0, img_1), val (census_0, rect_0, census_1, rect_1, rectification(img0, img1))
		//����� �� �����
#ifdef ENABLE_RECTIFICATION
		map<pair<int, int>, std::tuple<cv::Mat, Rect, cv::Mat, Rect, FusielloRectification>> rectified_pairs;
#else
		map<int, std::tuple<cv::Mat, Rect>> local_census_data;
#endif


		//����������� ��������� �������� �� �������
		//����� �������� ������ � ������������ ������������
		int64 non_empty_voxels_counter = 0;

		boost::numeric::ublas::matrix<vector<int>> cells(extended_rect.height, extended_rect.width, vector<int>(dimz, -1));

		//cout << endl << "make_rectification" << endl;

#ifndef ENABLE_RECTIFICATION
		map<int, int> used_images_map;
#endif

		for(int i = 0; i < dimz; i++)
			for(int j = extended_rect.x; j < extended_rect.x + extended_rect.width; j++)
				for(int k = extended_rect.y; k < extended_rect.y + extended_rect.height; k++)
				{
					Vec3 cur_voxel_pos = min_pt + Vec3(j*voxel_step + voxel_step/2, k*voxel_step + voxel_step/2, i*voxel_step + voxel_step/2);

					int camera_id = 0;

					pair<int, float> base_img(-1, 1000);

					bool non_empty = false;

					//���������� �������� ������ � ���� ������� � ������ ������
					for(std::map<size_t, PinholeCamera >::const_iterator cam_it = Doc->_map_camera.begin(); cam_it != Doc->_map_camera.end(); cam_it++, camera_id++)
					{
						Vec3 ray_2_voxel_center = cur_voxel_pos - cam_it->second._C;
						ray_2_voxel_center.normalize();

						double dotAngle = cam_it->second.ViewRay().dot(ray_2_voxel_center);
						double mag = cam_it->second.ViewRay().norm()*ray_2_voxel_center.norm();
						auto angle = abs(acos(clamp(dotAngle/mag, -1.0 + 1.e-8, 1.0 - 1.e-8)));

						local_angles_list[camera_id] = angle;

						if(angle < base_img.second) base_img = make_pair(camera_id, angle);
					}

					if(base_img.first < 0) continue;

					camera_id = 0;

					//��� �������, ���������� ������� ��������� ������������ � �������
					//������� �������� ��������� ����� (��� ����� ���������� ��������� �������� ������������ ������ �� �������� ������)
					for(std::map<size_t, PinholeCamera >::const_iterator cam_it = Doc->_map_camera.begin(); cam_it != Doc->_map_camera.end(); cam_it++, camera_id++)
					{
#ifndef ENABLE_RECTIFICATION
						if(local_angles_list[camera_id] <= max_angles_list[camera_id])
							used_images_map[camera_id] = camera_id;
#endif

						if(camera_id != base_img.first)
						{
							if(local_angles_list[camera_id] <= max_angles_list[camera_id])
							{
								non_empty = true;
							
								//check for existance of rectification
								int id0 = base_img.first, id1 = camera_id;

#ifdef ENABLE_RECTIFICATION
								if(rectified_pairs.find(make_pair(id0, id1))==rectified_pairs.end())
								{
#pragma omp critical
									{
										if(rectified_pairs.find(make_pair(id0, id1))==rectified_pairs.end())
										{
											//check if exists on disk -> load and swap
											//else create and swap in memory

											string dir1 = cache_dir + to_string(id0) + "_" + to_string(id1) + "\\", dir2 = cache_dir + to_string(id1) + "_" + to_string(id0) + "\\";

											if(stlplus::folder_exists(dir1) || stlplus::folder_exists(dir2))
											{
												//load data
												//swap if necessary
												bool swap = stlplus::folder_exists(dir2);
												string main_dir = swap ? dir2 : dir1;
												string rectification_data_file_name = main_dir + rectification_name;

												FusielloRectification rectification;
												rectification.Load(rectification_data_file_name);

												if(swap) rectification.swap();

												rectified_pairs[make_pair(id0, id1)] = make_tuple(cv::Mat(), Rect(), cv::Mat(), Rect(), rectification);
											}
											else
											{
												stlplus::folder_create(dir1);
												string inp_img_name_1 = images_dir + Doc->_vec_imageNames[id0], inp_img_name_2 = images_dir + Doc->_vec_imageNames[id1];
												string out_img_name_1 = dir1 + left_img_name, out_img_name_2 = dir1 + right_img_name;
												string rectification_data_name = dir1 + rectification_name;

												//create rectification
												cv::Mat Po1, Po2;
												eigen2cv(Doc->_map_camera[id0]._P, Po1);
												eigen2cv(Doc->_map_camera[id1]._P, Po2);

												FusielloRectification rectification(Po1, Po2);
												rectification.wrap_left(inp_img_name_1, out_img_name_1);
												rectification.wrap_right(inp_img_name_2, out_img_name_2);

												rectification.Write(rectification_data_name);

												rectified_pairs[make_pair(id0, id1)] = make_tuple(cv::Mat(), Rect(), cv::Mat(), Rect(), rectification);
											}
										}
									}
								}
#endif
							}
						}

				}

					if(non_empty) 
					{
						non_empty_voxels_counter++;
						cells(k - extended_rect.y, j - extended_rect.x)[i] = base_img.first;
					}
				}

				if(non_empty_voxels_counter < 1000) continue;

				//����� ����, ��� ������������ ������ ����� ������������ ������� ����� ����� �� ����������������� ������ � ��������� ��������������� ����� ������ �������� � ������

				vector<Vec3> block_corners_list;
				block_corners_list.push_back(min_pt + Vec3(extended_rect.x*voxel_step + voxel_step/2, extended_rect.y*voxel_step + voxel_step/2, 0));
				block_corners_list.push_back(min_pt + Vec3((extended_rect.x + extended_rect.width - 1)*voxel_step + voxel_step/2, extended_rect.y*voxel_step + voxel_step/2, 0));
				block_corners_list.push_back(min_pt + Vec3(extended_rect.x*voxel_step + voxel_step/2, (extended_rect.y + extended_rect.height - 1)*voxel_step + voxel_step/2, 0));
				block_corners_list.push_back(min_pt + Vec3((extended_rect.x + extended_rect.width - 1)*voxel_step + voxel_step/2, (extended_rect.y + extended_rect.height - 1)*voxel_step + voxel_step/2, 0));

				block_corners_list.push_back(min_pt + Vec3(extended_rect.x*voxel_step + voxel_step/2, extended_rect.y*voxel_step + voxel_step/2, dimz*voxel_step));
				block_corners_list.push_back(min_pt + Vec3((extended_rect.x + extended_rect.width - 1)*voxel_step + voxel_step/2, extended_rect.y*voxel_step + voxel_step/2, dimz*voxel_step));
				block_corners_list.push_back(min_pt + Vec3(extended_rect.x*voxel_step + voxel_step/2, (extended_rect.y + extended_rect.height - 1)*voxel_step + voxel_step/2, dimz*voxel_step));
				block_corners_list.push_back(min_pt + Vec3((extended_rect.x + extended_rect.width - 1)*voxel_step + voxel_step/2, (extended_rect.y + extended_rect.height - 1)*voxel_step + voxel_step/2, dimz*voxel_step));
				//block_corners_list.push_back(Vec3(-2.4329, -1.2729, 3.0451));

				/*cout << "Voxel data corners" << endl;
				for each(auto &pt in block_corners_list) cout << pt << endl << endl;
				cout << endl;*/

				Rect initial_img_rect_pyr;

#ifdef ENABLE_RECTIFICATION
				//������ �� rectified_pairs � ��������
				for(auto rect_it = rectified_pairs.begin(); rect_it != rectified_pairs.end(); rect_it++)
				{
					int id0 = rect_it->first.first, id1 = rect_it->first.second;

					auto im_size = GetImageSize(images_dir + Doc->_vec_imageNames[id0]);
					initial_img_rect_pyr = Rect(0, 0, im_size.width/pow(2.0, pyr_lvl), im_size.height/pow(2.0, pyr_lvl));

					string dir1 = cache_dir + to_string(id0) + "_" + to_string(id1) + "\\", dir2 = cache_dir + to_string(id1) + "_" + to_string(id0) + "\\";
					
					if(stlplus::folder_exists(dir1))
					{
						auto rect_patch_data = LoadBlocksFormRectifiedImages(block_corners_list, std::get<4>(rect_it->second), dir1 + left_img_name, dir1 + right_img_name, Doc, id0, id1, pyr_lvl);

						std::get<0>(rect_it->second) = std::get<0>(rect_patch_data);
						std::get<1>(rect_it->second) = std::get<1>(rect_patch_data);
						std::get<2>(rect_it->second) = std::get<2>(rect_patch_data);
						std::get<3>(rect_it->second) = std::get<3>(rect_patch_data);
					}
					else
					{
						FusielloRectification buf_rect = std::get<4>(rect_it->second);
						buf_rect.swap();
						
						auto rect_patch_data = LoadBlocksFormRectifiedImages(block_corners_list, buf_rect, dir2 + left_img_name, dir2 + right_img_name, Doc, id1, id0, pyr_lvl);

						std::get<0>(rect_it->second) = std::get<2>(rect_patch_data);
						std::get<1>(rect_it->second) = std::get<3>(rect_patch_data);
						std::get<2>(rect_it->second) = std::get<0>(rect_patch_data);
						std::get<3>(rect_it->second) = std::get<1>(rect_patch_data);
					}
				}

#else
				vector<int> keys_4_remove;

				for(auto it = used_images_map.begin(); it != used_images_map.end(); it++)
				{
					auto im_size = GetImageSize(images_dir + Doc->_vec_imageNames[it->first]);
					initial_img_rect_pyr = Rect(0, 0, im_size.width/pow(2.0, pyr_lvl), im_size.height/pow(2.0, pyr_lvl));
					
					auto gray_scale_block = LoadBlocksFormImages(block_corners_list, images_dir + Doc->_vec_imageNames[it->first], Doc, it->first, pyr_lvl, false);

					if(std::get<0>(gray_scale_block).empty()) keys_4_remove.push_back(it->first);
					else local_census_data[it->first] = make_tuple(CensusTransform5x5(std::get<0>(gray_scale_block)), std::get<1>(gray_scale_block));
				}

				for each(auto key in keys_4_remove) local_census_data.erase(key);
#endif

				vector<Point2f> base_pts_list(1), neighbor_pts_list(1), rect_base_pts_list(1), rect_neighbor_pts_list(1);
				Point buf_base_pt, buf_neighbor_pt;
				float local_cost;

				//cout << endl << "calc costs" << endl;
				int64 non_projection_cases = 0;

				//������ ����� �� ������� � �������� ������� ����������
				for(int i = 0; i < dimz; i++)
					for(int j = extended_rect.x; j < extended_rect.x + extended_rect.width; j++)
						for(int k = extended_rect.y; k < extended_rect.y + extended_rect.height; k++)
						if(cells(k - extended_rect.y, j - extended_rect.x)[i] >= 0)
						{
							int base_img_id = cells(k - extended_rect.y, j - extended_rect.x)[i];
							
							Vec3 cur_voxel_pos = min_pt + Vec3(j*voxel_step + voxel_step/2, k*voxel_step + voxel_step/2, i*voxel_step + voxel_step/2);

							int camera_id = 0;
							
							auto base_img_prj = Doc->_map_camera[base_img_id].Project(cur_voxel_pos);
							Point2f base_img_projection(base_img_prj.x(), base_img_prj.y());

							if(!original_images_rects[base_img_id].contains(base_img_projection)) continue;

#ifdef ENABLE_RECTIFICATION
							base_pts_list[0] = base_img_projection;
#else
							base_img_projection/= pow(2.0, pyr_lvl);
#endif

							//���������� ��������� � �������� ������
							for(std::map<size_t, PinholeCamera >::const_iterator cam_it = Doc->_map_camera.begin(); cam_it != Doc->_map_camera.end(); cam_it++, camera_id++)
							if(camera_id != base_img_id)
							{
#ifdef ENABLE_RECTIFICATION
								if(rectified_pairs.find(make_pair(base_img_id, camera_id)) == rectified_pairs.end()) continue;

								//angle test
								/*Vec3 ray_2_voxel_center = cur_voxel_pos - cam_it->second._C;
								ray_2_voxel_center.normalize();

								double dotAngle = cam_it->second.ViewRay().dot(ray_2_voxel_center);
								double mag = cam_it->second.ViewRay().norm()*ray_2_voxel_center.norm();
								auto angle = abs(acos(clamp(dotAngle/mag, -1.0 + 1.e-8, 1.0 - 1.e-8)));

								if(angle > max_angles_list[camera_id]) continue;*/

								
								auto &rect_patch = rectified_pairs[make_pair(base_img_id, camera_id)];

								auto search_img_prj = Doc->_map_camera[camera_id].Project(cur_voxel_pos);
								Point2f search_img_projection(search_img_prj.x(), search_img_prj.y());

								if(!original_images_rects[camera_id].contains(search_img_projection)) continue;

								neighbor_pts_list[0] = search_img_projection;

								std::get<4>(rect_patch).convert_2_transform_left(base_pts_list, rect_base_pts_list);
								std::get<4>(rect_patch).convert_2_transform_right(neighbor_pts_list, rect_neighbor_pts_list);

								rect_base_pts_list[0].x/= pow(2.0, pyr_lvl); rect_base_pts_list[0].y/= pow(2.0, pyr_lvl);
								rect_neighbor_pts_list[0].x/= pow(2.0, pyr_lvl); rect_neighbor_pts_list[0].y/= pow(2.0, pyr_lvl);

								buf_base_pt = rect_base_pts_list[0]; buf_neighbor_pt = rect_neighbor_pts_list[0];
								if(std::get<1>(rect_patch).contains(buf_base_pt) && std::get<3>(rect_patch).contains(buf_neighbor_pt))
									{
										buf_base_pt.x-= std::get<1>(rect_patch).x; buf_base_pt.y-= std::get<1>(rect_patch).y;

										buf_neighbor_pt.x-= std::get<3>(rect_patch).x; buf_neighbor_pt.y-= std::get<3>(rect_patch).y;

										local_cost = CalculatePixelSimilarity(buf_base_pt, buf_neighbor_pt, std::get<0>(rect_patch), std::get<2>(rect_patch));

										local_costs_calculator.UpdateCosts( j - extended_rect.x, k - extended_rect.y, i, local_cost );
									}
								else 
								{
									Point2f d_base_pt = base_img_projection/pow(2.0, pyr_lvl), d_neighbor_pt = search_img_projection/pow(2.0, pyr_lvl);
									
									//if(initial_img_rect_pyr.contains(d_base_pt) && initial_img_rect_pyr.contains(d_neighbor_pt)) non_projection_cases++;

									/*cout << "base_img_projection = " << base_img_projection << " image = " << Doc->_vec_imageNames[base_img_id] << endl;
									cout << "search_img_projection = " << search_img_projection << endl << " base_img_id = " << Doc->_vec_imageNames[camera_id] << endl;
									cout << "rect_base = " << rect_base_pts_list[0] << " " << base_img_id << endl;
									cout << "rect_search = " << rect_neighbor_pts_list[0] << " " << camera_id << endl;
									cout << "rect = " << std::get<1>(rect_patch) << endl;
									cout << base_img_id << " " << camera_id << endl;
									cout << k << " " << j << " " << i << endl << cur_voxel_pos << endl;
									std::system("PAUSE");*/

									non_projection_cases++;

									/*Point2f d_base_pt = base_img_projection/pow(2.0, pyr_lvl), d_neighbor_pt = search_img_projection/pow(2.0, pyr_lvl);

									if(initial_img_rect_pyr.contains(d_base_pt) && initial_img_rect_pyr.contains(d_neighbor_pt))
									{
										cout << "base_img_projection = " << base_img_projection << " image = " << Doc->_vec_imageNames[base_img_id] << endl;
										cout << "search_img_projection = " << search_img_projection << endl << " base_img_id = " << Doc->_vec_imageNames[camera_id] << endl;
										cout << "rect_base = " << rect_base_pts_list[0] << " " << base_img_id << endl;
										cout << "rect_search = " << rect_neighbor_pts_list[0] << " " << camera_id << endl;
										cout << "rect = " << std::get<1>(rect_patch) << endl;
										cout << base_img_id << " " << camera_id << endl;
										cout << k << " " << j << " " << i << endl << cur_voxel_pos << endl;
										std::system("PAUSE");
									}*/

									//initial_img_rect_pyr
								}
#else
								if(local_census_data.find(base_img_id)!=local_census_data.end() && local_census_data.find(camera_id)!=local_census_data.end())
								{
									//angle test
									/*Vec3 ray_2_voxel_center = cur_voxel_pos - cam_it->second._C;
									ray_2_voxel_center.normalize();

									double dotAngle = cam_it->second.ViewRay().dot(ray_2_voxel_center);
									double mag = cam_it->second.ViewRay().norm()*ray_2_voxel_center.norm();
									auto angle = abs(acos(clamp(dotAngle/mag, -1.0 + 1.e-8, 1.0 - 1.e-8)));

									if(angle > max_angles_list[camera_id]) continue;*/

									
									auto &local_census_data_1 = local_census_data[base_img_id];
									auto &local_census_data_2 = local_census_data[camera_id];

									auto search_img_prj = Doc->_map_camera[camera_id].Project(cur_voxel_pos);
									Point2f search_img_projection(search_img_prj.x(), search_img_prj.y());

									search_img_projection/= pow(2.0, pyr_lvl);

									buf_base_pt = base_img_projection; buf_neighbor_pt = search_img_projection;
									
									if(std::get<1>(local_census_data_1).contains(buf_base_pt) && std::get<1>(local_census_data_2).contains(buf_neighbor_pt))
									{
										buf_base_pt.x-= std::get<1>(local_census_data_1).x; buf_base_pt.y-= std::get<1>(local_census_data_1).y;

										buf_neighbor_pt.x-= std::get<1>(local_census_data_2).x; buf_neighbor_pt.y-= std::get<1>(local_census_data_2).y;

										local_cost = CalculatePixelSimilarity(buf_base_pt, buf_neighbor_pt, std::get<0>(local_census_data_1), std::get<0>(local_census_data_2));

										local_costs_calculator.UpdateCosts( j - extended_rect.x, k - extended_rect.y, i, local_cost );
									}
									else
									{
										non_projection_cases++;

										/*cout << "buf_base_pt = " << buf_base_pt << " image = " << Doc->_vec_imageNames[base_img_id] << endl;
										cout << "base_img_prj = " << base_img_prj << endl << " base_img_id = " << base_img_id << endl;
										cout << "base_img_projection = " << base_img_projection << endl << " base_img_projection = " << base_img_id << endl;
										cout << "buf_neighbor_pt = " << buf_neighbor_pt << " image = " << Doc->_vec_imageNames[camera_id] << endl;
										cout << "std::get<1>(local_census_data_1) = " << std::get<1>(local_census_data_1) << endl;
										cout << "std::get<1>(local_census_data_2) = " << std::get<1>(local_census_data_2) << endl;
										cout << k << " " << j << " " << i << endl << cur_voxel_pos << endl;
										std::system("PAUSE");*/
										
										/*if(initial_img_rect_pyr.contains(buf_base_pt) && initial_img_rect_pyr.contains(buf_neighbor_pt))
										{
											cout << "buf_base_pt = " << buf_base_pt << " image = " << Doc->_vec_imageNames[base_img_id] << endl;
											cout << "base_img_prj = " << base_img_prj << endl << " base_img_id = " << base_img_id << endl;
											cout << "base_img_projection = " << base_img_projection << endl << " base_img_projection = " << base_img_id << endl;
											cout << "buf_neighbor_pt = " << buf_neighbor_pt << " image = " << Doc->_vec_imageNames[camera_id] << endl;
											cout << "std::get<1>(local_census_data_1) = " << std::get<1>(local_census_data_1) << endl;
											cout << "std::get<1>(local_census_data_2) = " << std::get<1>(local_census_data_2) << endl;
											cout << k << " " << j << " " << i << endl << cur_voxel_pos << endl;
											std::system("PAUSE");
										}*/
										
									}
								}
#endif
							}
						}

		//cout << "non_projection_cases = " << non_projection_cases << endl;
/*
		workbook wb;
		auto sh = wb.sheet("SGM");

		int Width = local_costs_calculator.LocalCosts.size();
		int Height = local_costs_calculator.LocalCosts.front().size();
		int Depth = local_costs_calculator.LocalCosts.front().front().size();
		for(int i = 0; i < Width; i++)
			for(int j = 0; j < Height; j++)
			{
				float max_val = 0;
				for(int k = 0; k < Depth; k++)
					if(local_costs_calculator.LocalCosts[i][j][k] > max_val)
						max_val = local_costs_calculator.LocalCosts[i][j][k];

				sh->number(j, i, max_val);
			}

		wb.Dump("J:\\delme\\partial_sgm.xls");*/

		/*std::system("PAUSE");*/

		//cout << endl << "calc sgm" << endl;

		auto ag_costs = CalculateSGMCosts( local_costs_calculator.LocalCosts );

		//cout << "Calc sgm..." << endl;
#ifdef USE_FLOAT_DISPARITY
		auto full_disp_map = GetDepthMapSuPix(ag_costs);
#else
		auto full_disp_map = GetDepthMapSuPixInt(ag_costs);
#endif

		//save depthmap to appropriate rect
		//FILL_EMPTY_CELL_VALUE

#ifdef DEBUG_OUTPUT
		cv::Mat local_num_mat = cv::Mat::zeros(full_disp_map.size(), CV_32FC1);
#endif // DEBUG_OUTPUT


		//fill empty cells
		const int FILL_EMPTY_CELL_VALUE = -1;
		const float projections_number_t = 1.0f;

		for( size_t y(0); y < full_disp_map.rows; ++y)
		{
#ifdef USE_FLOAT_DISPARITY
#ifdef FILTER_DISP_MAP
			/*float *const scanLine( full_disp_map.ptr<float>(y) );
			for( size_t x(0); x < full_disp_map.cols; ++x)
			{
				if( local_costs_calculator.NumberOfProjections[x][y][scanLine[x]] == 0 )
					scanLine[x] = FILL_EMPTY_CELL_VALUE;
			}*/

			for( size_t x(0); x < full_disp_map.cols; ++x)
			{
				float mean_val = 0;
				
				for(int z = 0; z < dimz; z++)
				{
					mean_val+= local_costs_calculator.NumberOfProjections[x][y][z];
				}

				mean_val/= dimz;

				if(mean_val < projections_number_t)
					full_disp_map.at<float>(y, x) = FILL_EMPTY_CELL_VALUE;
			}
#endif

#ifdef DEBUG_OUTPUT
			for( size_t x(0); x < full_disp_map.cols; ++x)
			{
				for(int z = 0; z < dimz; z++)
				{
					local_num_mat.at<float>(y, x)+= local_costs_calculator.NumberOfProjections[x][y][z];
				}

				local_num_mat.at<float>(y, x)/= dimz;
			}
#endif
			
#else
#ifdef FILTER_DISP_MAP
			int *const scanLine( full_disp_map.ptr<int>(y) );
			for( size_t x(0); x < full_disp_map.cols; ++x)
				if( local_costs_calculator.NumberOfProjections[x][y][scanLine[x]] == 0 )
					scanLine[x] = FILL_EMPTY_CELL_VALUE;
#endif
#endif
		}

#ifdef DEBUG_OUTPUT
		//mean_num_file_name
		cv::Mat trunc_local_num_mat = local_num_mat(Rect(block.x - extended_rect.x, block.y - extended_rect.y, block.width, block.height)).clone();

		WriteBlock2FloatTiff(mean_num_file_name, block.x, block.y, block.width, block.height, (float*)trunc_local_num_mat.data);
#endif

#ifdef USE_FLOAT_DISPARITY
		WriteBlock2FloatTiff(full_disp_name, extended_rect.x, extended_rect.y, extended_rect.width, extended_rect.height, (float*)full_disp_map.data);
#else
		WriteBlock2IntTiff(full_disp_name, extended_rect.x, extended_rect.y, extended_rect.width, extended_rect.height, (int*)full_disp_map.data);
#endif

		cv::Mat trunc_disp_map = full_disp_map(Rect(block.x - extended_rect.x, block.y - extended_rect.y, block.width, block.height)).clone();

		//cout << "inner_rect = " << inner_rect << endl;
#ifdef TRUNCATE_DISP_MAP
		for(int j = 0; j < trunc_disp_map.rows; j++)
			for(int k = 0; k < trunc_disp_map.cols; k++)
			{
				Point pt(k + block.x, j + block.y);

#ifdef USE_FLOAT_DISPARITY
				if(!inner_rect.contains(pt)) trunc_disp_map.at<float>(j, k) = -1;
#else
				if(!inner_rect.contains(pt)) trunc_disp_map.at<int>(j, k) = -1;
#endif
			}
#endif

#pragma omp critical
		{
#ifdef USE_FLOAT_DISPARITY
			WriteBlock2FloatTiff(trunc_disp_name, block.x, block.y, block.width, block.height, (float*)trunc_disp_map.data);
#else
			WriteBlock2IntTiff(trunc_disp_name, block.x, block.y, block.width, block.height, (int*)trunc_disp_map.data);
#endif

			//Save2Geotiff(trunc_depth_map, "J:\\delme\\mvsgm.tiff");
		}

		cv::Mat depth_map(trunc_disp_map.size(), CV_32FC1, DEPTH_MAP_NODATA_VALUE);
		
		for(int i = block.y; i < block.y + block.height; i++)
			for(int j = block.x; j < block.x + block.width; j++)
			{
				int x = j - block.x, y = i - block.y; 
#ifdef USE_FLOAT_DISPARITY
					float z = trunc_disp_map.at<float>(y, x);
#else
					int z = trunc_disp_map.at<int>(y, x);
#endif
				if(z <= FILL_EMPTY_CELL_VALUE) continue;

#ifdef USE_FLOAT_DISPARITY
				float sub_z = z - cvFloor(z);
				
				Vec3 cur_voxel_pos = min_pt + Vec3(j*voxel_step + voxel_step/2, i*voxel_step + voxel_step/2, cvFloor(z)*voxel_step + sub_z*voxel_step);
#else
				Vec3 cur_voxel_pos = min_pt + Vec3(j*voxel_step + voxel_step/2, i*voxel_step + voxel_step/2, z*voxel_step + voxel_step/2);
#endif

				depth_map.at<float>(y, x) = cur_voxel_pos.z();
			}
#pragma omp critical
			{
				WriteBlock2FloatTiff(depth_map_name, block.x, block.y, block.width, block.height, (float*)depth_map.data);
			}

		//cout << endl << "calc xyzrgb" << endl;

#ifdef OUTPUT_XYZRGB


		//��� block ���������� ������� ����� � ��������� ����� ������� ������� �� ����� ������ ��������
		map<int, std::tuple<cv::Mat, Rect>> local_color_data;

#ifdef ENABLE_RECTIFICATION
		for(auto rect_it = rectified_pairs.begin(); rect_it != rectified_pairs.end(); rect_it++)
		{
			string image_name = images_dir + Doc->_vec_imageNames[rect_it->first.first];

			local_color_data[rect_it->first.first] = LoadBlocksFormImages(block_corners_list, image_name, Doc, rect_it->first.first, pyr_lvl);
		}
#else
		for(auto it = local_census_data.end(); it != local_census_data.end(); it++)
		{
			string image_name = images_dir + Doc->_vec_imageNames[it->first];

			local_color_data[it->first] = LoadBlocksFormImages(block_corners_list, image_name, Doc, it->first, pyr_lvl);
		}

#endif

#ifdef ENABLE_RECTIFICATION
		rectified_pairs.clear();
#else
		local_census_data.clear();
#endif

		cv::Mat ortho_photo = cv::Mat::zeros(block.height, block.width, CV_8UC3);

		vector<pcl::PointXYZRGB> xyzrgb_list;
		xyzrgb_list.reserve(block.width*block.height);

		for(int i = block.y; i < block.y + block.height; i++)
			for(int j = block.x; j < block.x + block.width; j++)
			{
				int x = j - block.x, y = i - block.y;
				float z = depth_map.at<float>(y, x);

				if(z <= DEPTH_MAP_NODATA_VALUE) continue;

				Vec3 cur_voxel_pos = min_pt + Vec3(j*voxel_step + voxel_step/2, i*voxel_step + voxel_step/2, 0);
				cur_voxel_pos.z() = z;
				
				//����������� �������� ������ � ������ � ���� �����
				int camera_id = 0;

				int base_img_id = cells(i - extended_rect.y, j - extended_rect.x)[z];

				Vec2 base_pt_projection = Doc->_map_camera[base_img_id].Project(cur_voxel_pos)/pow(2.0, pyr_lvl);

				auto &img_patch = local_color_data[base_img_id];

				Point base_local_pt_prj(base_pt_projection.x(), base_pt_projection.y());

				if(std::get<1>(img_patch).contains(base_local_pt_prj))
				{
					base_local_pt_prj.x-= std::get<1>(img_patch).x; base_local_pt_prj.y-= std::get<1>(img_patch).y;

					ortho_photo.at<Vec3b>(y, x) = std::get<0>(img_patch).at<Vec3b>(base_local_pt_prj.y, base_local_pt_prj.x);
					
					pcl::PointXYZRGB new_pt;
					new_pt.x = cur_voxel_pos.x(); new_pt.y = cur_voxel_pos.y(); new_pt.z = cur_voxel_pos.z();
					new_pt.r = std::get<0>(img_patch).at<Vec3b>(base_local_pt_prj.y, base_local_pt_prj.x)[2];
					new_pt.g = std::get<0>(img_patch).at<Vec3b>(base_local_pt_prj.y, base_local_pt_prj.x)[1];
					new_pt.b = std::get<0>(img_patch).at<Vec3b>(base_local_pt_prj.y, base_local_pt_prj.x)[0];
					
					xyzrgb_list.push_back(new_pt);
				}
			}
		
#pragma omp critical
			{
				WriteColorRasterCV(orthophoto_name, ortho_photo, block);
				
				if(xyzrgb_name!="") write_xyzrgb(xyzrgb_list, xyzrgb_name, true);
			}
#endif

#pragma omp critical
			{
				sgmProgress++;
			}
	
			//std::system("PAUSE");
	}
}

bool CalcVoxelGridMetaData(Document *Doc, int min_pixel_projection_size, int pyr_lvl, double &voxel_step, Vec3 &min_pt, Vec3 &max_pt, int &dimx, int &dimy, int &dimz, int k_disp_offset)
{
	if(!Doc) return false;

	//initialization

	vector<openMVG::Vec3> tie_pts_list;

	auto trackIndex = 0;

	//find pt visible on first image
	int track_id_4_first_img = -1;

	for (std::map< size_t, tracks::submapTrack >::const_iterator iterTracks = Doc->_tracks.begin(); iterTracks != Doc->_tracks.end(); ++iterTracks,++trackIndex)
	{
		const tracks::submapTrack & map_track = iterTracks->second;

		if(track_id_4_first_img < 0)
			for(tracks::submapTrack::const_iterator iter_pts = map_track.begin(); iter_pts != map_track.end(); ++iter_pts)
				if(iter_pts->first == 0)
					track_id_4_first_img = trackIndex;

		const float * ptr3D = & Doc->_vec_points[trackIndex*3];

		tie_pts_list.push_back(openMVG::Vec3(ptr3D[0], ptr3D[1], ptr3D[2]));
	}

	Vec3 min_pos = tie_pts_list[0], max_pos = tie_pts_list[0];

	for each (auto &pt in tie_pts_list)
	{
		if(pt.x() < min_pos.x()) min_pos.x() = pt.x();
		if(pt.y() < min_pos.y()) min_pos.y() = pt.y();
		if(pt.z() < min_pos.z()) min_pos.z() = pt.z();

		if(pt.x() > max_pos.x()) max_pos.x() = pt.x();
		if(pt.y() > max_pos.y()) max_pos.y() = pt.y();
		if(pt.z() > max_pos.z()) max_pos.z() = pt.z();
	}

	auto working_aabb = make_pair(min_pos, max_pos);

	auto box_differ = max_pos - min_pos;
	float max_size = max(abs(box_differ.x()), max(abs(box_differ.y()), abs(box_differ.z())));

	//calc working octree lvl

	int cur_lvl = 0;
	auto center = tie_pts_list[track_id_4_first_img];//(min_pos + max_pos)/2;
	pair<Vec3, Vec3> cur_box(center - Vec3(max_size/2, max_size/2, max_size/2), center + Vec3(max_size/2, max_size/2, max_size/2));
	vector<Vec3> prj_list(8);

	while(true)
	{
		prj_list[0] = cur_box.first;
		prj_list[1] = Vec3(cur_box.second.x(), cur_box.first.y(), cur_box.first.z());
		prj_list[2] = Vec3(cur_box.second.x(), cur_box.second.y(), cur_box.first.z());
		prj_list[3] = Vec3(cur_box.first.x(), cur_box.second.y(), cur_box.first.z());
		prj_list[4] = cur_box.second;
		prj_list[5] = Vec3(cur_box.first.x(), cur_box.second.y(), cur_box.second.z());
		prj_list[6] = Vec3(cur_box.first.x(), cur_box.first.y(), cur_box.second.z());
		prj_list[7] = Vec3(cur_box.second.x(), cur_box.first.y(), cur_box.second.z());

		Vec2 min_proj = Doc->_map_camera.begin()->second.Project(prj_list[0])/pow(2.0, pyr_lvl), max_proj = min_proj;

		for(int i = 0; i < 8; i++)
		{
			auto proj = Doc->_map_camera.begin()->second.Project(prj_list[i]);
			auto down_proj = proj/pow(2.0, pyr_lvl);

			if(down_proj.x() < min_proj.x()) min_proj.x() = down_proj.x();
			if(down_proj.y() < min_proj.y()) min_proj.y() = down_proj.y();

			if(down_proj.x() > max_proj.x()) max_proj.x() = down_proj.x();
			if(down_proj.y() > max_proj.y()) max_proj.y() = down_proj.y();
		}

		auto diff_2d = (max_proj - min_proj).norm();
		if(diff_2d <= min_pixel_projection_size) 
			break;

		Vec3 local_size = cur_box.second - cur_box.first;
		cur_box = make_pair(center - local_size/4, center + local_size/4);

		cur_lvl++;
	}

	//������ �������
	voxel_step = abs(cur_box.second.x() - cur_box.first.x());

	working_aabb.first.z()-= k_disp_offset*voxel_step;
	working_aabb.second.z()+= k_disp_offset*voxel_step;

	//������ ���������� �������
	dimx = boost::math::round((working_aabb.second.x() - working_aabb.first.x())/voxel_step);
	dimy = boost::math::round((working_aabb.second.y() - working_aabb.first.y())/voxel_step);
	dimz = boost::math::round((working_aabb.second.z() - working_aabb.first.z())/voxel_step);

	min_pt = working_aabb.first;
	max_pt = working_aabb.second;

	return true;
}

std::tuple<cv::Mat, Rect> LoadBlocksFormImages(const vector<Vec3> &inp_pts, string image_name, Document *Doc, int id, int pyr_lvl, bool use_color)
{
	//calc projection of corners into original image
	vector<Vec2> original_img_projection_list;
	for each(auto &pt in inp_pts)
		original_img_projection_list.push_back(Doc->_map_camera[id].Project(pt));

	//convert projection 2 rectified image
	Vec2 min_pt(original_img_projection_list[0]), max_pt(original_img_projection_list[0]);

	for each(auto &pt in original_img_projection_list)
	{
		if(pt.x() < min_pt.x()) min_pt.x() = pt.x();
		if(pt.y() < min_pt.y()) min_pt.y() = pt.y();

		if(pt.x() > max_pt.x()) max_pt.x() = pt.x();
		if(pt.y() > max_pt.y()) max_pt.y() = pt.y();
	}

	/*cout << "image = " << Doc->_vec_imageNames[id] << endl;
	cout << "min_pt = " << min_pt << endl;
	cout << "max_pt = " << max_pt << endl;
	cout << endl;*/


	auto img_size = GetImageSize(image_name);

	Rect img_rect(0, 0, img_size.width, img_size.height);

	//and expand windows by 5
	const int local_offset = 5;

	Rect rect = Rect(cvFloor(min_pt.x()) - local_offset, cvFloor(min_pt.y()) - local_offset, cvCeil(max_pt.x()) - cvFloor(min_pt.x()) + 2*local_offset + 1, cvCeil(max_pt.y()) - cvFloor(min_pt.y()) + 2*local_offset + 1) & img_rect;

	if(!rect.width||!rect.height) return make_tuple(cv::Mat(), Rect());

	//load from rect file & calc census
	cv::Mat img_patch = use_color?LoadRaster(image_name, pyr_lvl, false, rect):LoadGrayScaleImageCV(image_name, pyr_lvl, rect);

	rect.x/= pow(2.0, pyr_lvl); rect.y/= pow(2.0, pyr_lvl); rect.width/= pow(2.0, pyr_lvl); rect.height/= pow(2.0, pyr_lvl);

	/*imwrite("J:\\delme\\left.bmp", img_patch_1); imwrite("J:\\delme\\right.bmp", img_patch_2);
	cout << "dump patches" << endl;
	std::system("PAUSE");*/

	return make_tuple(img_patch, rect);
}

std::tuple<cv::Mat, Rect, cv::Mat, Rect> LoadBlocksFormRectifiedImages(const vector<Vec3> &inp_pts, FusielloRectification &rectification, string image_name_1, string image_name_2, Document *Doc, int id0, int id1, int pyr_lvl, bool use_census)
{
	//calc projection of corners into original image
	vector<Vec2> original_img_projection_list_1, original_img_projection_list_2;
	for each(auto &pt in inp_pts)
	{
		original_img_projection_list_1.push_back(Doc->_map_camera[id0].Project(pt));
		original_img_projection_list_2.push_back(Doc->_map_camera[id1].Project(pt));
	}

	//convert projection 2 rectified image
	vector<Vec2> rect_img_projection_list_1 = rectification.convert_2_transform_left(original_img_projection_list_1), rect_img_projection_list_2 = rectification.convert_2_transform_right(original_img_projection_list_2); 
	
	Vec2 min_pt_1(rect_img_projection_list_1[0]), max_pt_1(rect_img_projection_list_1[0]), min_pt_2(rect_img_projection_list_2[0]), max_pt_2(rect_img_projection_list_2[0]);

	for each(auto &pt in rect_img_projection_list_1)
	{
		if(pt.x() < min_pt_1.x()) min_pt_1.x() = pt.x();
		if(pt.y() < min_pt_1.y()) min_pt_1.y() = pt.y();

		if(pt.x() > max_pt_1.x()) max_pt_1.x() = pt.x();
		if(pt.y() > max_pt_1.y()) max_pt_1.y() = pt.y();
	}

	for each(auto &pt in rect_img_projection_list_2)
	{
		if(pt.x() < min_pt_2.x()) min_pt_2.x() = pt.x();
		if(pt.y() < min_pt_2.y()) min_pt_2.y() = pt.y();

		if(pt.x() > max_pt_2.x()) max_pt_2.x() = pt.x();
		if(pt.y() > max_pt_2.y()) max_pt_2.y() = pt.y();
	}

/*
	cout << "image_name_1 = " << image_name_1 << endl;
	cout << "image_name_2 = " << image_name_2 << endl;*/

	auto left_rect_img_size = GetImageSize(image_name_1), right_rect_img_size = GetImageSize(image_name_2);

	Rect left_rect_img_rect(0, 0, left_rect_img_size.width, left_rect_img_size.height), right_rect_img_rect(0, 0, right_rect_img_size.width, right_rect_img_size.height);

	//and expand windows by 5
	const int local_offset = 5;

#ifdef USE_FULLIMAGES
	Rect rect_1 = left_rect_img_rect, rect_2 = right_rect_img_rect;
#else
	Rect rect_1 = Rect(cvFloor(min_pt_1.x()) - local_offset, cvFloor(min_pt_1.y()) - local_offset, cvCeil(max_pt_1.x()) - cvFloor(min_pt_1.x()) + 2*local_offset + 1, cvCeil(max_pt_1.y()) - cvFloor(min_pt_1.y()) + 2*local_offset + 1) & left_rect_img_rect;
	Rect rect_2 = Rect(cvFloor(min_pt_2.x()) - local_offset, cvFloor(min_pt_2.y()) - local_offset, cvCeil(max_pt_2.x()) - cvFloor(min_pt_2.x()) + 2*local_offset + 1, cvCeil(max_pt_2.y()) - cvFloor(min_pt_2.y())+ 2*local_offset + 1) & right_rect_img_rect;
#endif

	//load from rect file & calc census
	cv::Mat img_patch_1 = LoadGrayScaleImageCV(image_name_1, pyr_lvl, rect_1), img_patch_2 = LoadGrayScaleImageCV(image_name_2, pyr_lvl, rect_2);

	rect_1.x/= pow(2.0, pyr_lvl); rect_1.y/= pow(2.0, pyr_lvl); rect_1.width/= pow(2.0, pyr_lvl); rect_1.height/= pow(2.0, pyr_lvl);
	rect_2.x/= pow(2.0, pyr_lvl); rect_2.y/= pow(2.0, pyr_lvl); rect_2.width/= pow(2.0, pyr_lvl); rect_2.height/= pow(2.0, pyr_lvl);

	/*imwrite("J:\\delme\\left.bmp", img_patch_1); imwrite("J:\\delme\\right.bmp", img_patch_2);
	cout << "dump patches" << endl;
	std::system("PAUSE");*/

	return make_tuple(use_census?CensusTransform5x5(img_patch_1):img_patch_1, rect_1, use_census?CensusTransform5x5(img_patch_2):img_patch_2, rect_2);
}


cv::Mat CensusTransform3x3( const cv::Mat & Image )
{
	cv::Mat CensusMap = cv::Mat::zeros( Image.size(), CV_8U );
	vector< const unsigned char * > ReadLines( 3 );
	
	for( int j = 1; j < Image.rows - 1; ++j )
	{
		for( int l = 0; l < 3; ++l )
			ReadLines[l] = Image.ptr<unsigned char> ( j + l - 1 );
		unsigned char * CensusWriteLine = CensusMap.ptr<unsigned char>(j);
		for( int i = 1; i < Image.cols - 1; ++i )
		{
			int CurValue = ReadLines[1][i];

			/*CensusWriteLine[i] = 0;
			for( int l = 0; l < 3; ++l )
				for( int k = 0; k < 3; ++k )
				{
					if( k == 0 && l == 0 ) continue;
					bool Less = ReadLines[l][i + k - 1] < CurValue;
					CensusWriteLine[i] <<= 1;
					if( Less ) CensusWriteLine[i] += 1;
				}*/
			
			unsigned char CensusValue = 0;
			for( int l = 0; l < 3; ++l )
				for( int k = 0; k < 3; ++k )
				{
					if( k == 0 && l == 0 ) continue;
					bool Less = ReadLines[l][i + k - 1] < CurValue;
					CensusValue <<= 1;
					if( Less ) CensusValue += 1;
				}
			CensusWriteLine[i] = CensusValue;
		}
	}
	return CensusMap;
}

cv::Mat CensusTransform5x5( const cv::Mat & Image )
{
	cv::Mat CensusMap = cv::Mat::zeros( Image.size(), CV_32SC1 );
	vector< const unsigned char * > ReadLines( 5 );
	
	for( int j = 2; j < Image.rows - 2; ++j )
	{
		for( int l = 0; l < 5; ++l )
			ReadLines[l] = Image.ptr<unsigned char>( j + l - 2 );
		int * CensusWriteLine = CensusMap.ptr<int>(j);
		for( int i = 2; i < Image.cols - 2; ++i )
		{
			int CurValue = ReadLines[2][i];

			/*CensusWriteLine[i] = 0;
			for( int l = 0; l < 5; ++l )
				for( int k = 0; k < 5; ++k )
				{
					bool Less = ReadLines[l][i + k - 2] < CurValue;
					CensusWriteLine[i] <<= 1;
					if( Less ) CensusWriteLine[i] += 1;
				}*/
			int CensusValue = 0;
			for( int l = 0; l < 5; ++l )
				for( int k = 0; k < 5; ++k )
				{
					if( k == 0 && l == 0 ) continue;
					bool Less = ReadLines[l][i + k - 2] < CurValue;
					CensusValue <<= 1;
					if( Less ) CensusValue += 1;
				}
			CensusWriteLine[i] = CensusValue;
		}
	}
	return CensusMap;
}

cv::Mat CensusTransform7x7( const cv::Mat & Image )
{
	cv::Mat_< long int > CensusMap( Image.size() );
	vector< const unsigned char * > ReadLines( 7 );
	
	for( int j = 3; j < Image.rows - 3; ++j )
	{
		for( int l = 0; l < 7; ++l )
			ReadLines[l] = Image.ptr<unsigned char>( j + l - 3 );
		long int * CensusWriteLine = CensusMap.ptr<long int>(j);
		for( int i = 3; i < Image.cols - 3; ++i )
		{
			long int CurValue = ReadLines[3][i];

			long int CensusValue = 0;
			for( int l = 0; l < 7; ++l )
				for( int k = 0; k < 7; ++k )
				{
					if( k == 0 && l == 0 ) continue;
					bool Less = ReadLines[l][i + k - 3] < CurValue;
					CensusValue <<= 1;
					if( Less ) CensusValue += 1;
				}
			CensusWriteLine[i] = CensusValue;
		}
	}
	return CensusMap;
}


int HammingDistance( int A, int B )
{
	int S = A ^ B;
	int BitCounter = 0;

	if( A < 0 || B < 0 )
		return 8*sizeof(int);

	while( S )
	{
		if( S & 1 )
			++BitCounter;
		S >>= 1;
	}
	return BitCounter;
}

int HammingDistance( long int A, long int B )
{
	long int S = A ^ B;
	int BitCounter = 0;

	if( A < 0 || B < 0 )
		return 8*sizeof(long int);

	while( S )
	{
		if( S & 1 )
			++BitCounter;
		S >>= 1;
	}
	return BitCounter;
}

CostsCalculator::CostsCalculator( int dimx, int dimy, int dimz, float default_val ): CostThreshold(30.0f)
{
	LocalCosts = vector< vector< vector < float > > > ( dimx, vector< vector< float > > ( dimy, vector< float > ( dimz, default_val )  ) );//10001.0
	NumberOfProjections = vector< vector< vector < char > > > ( dimx, vector< vector< char > > ( dimy, vector< char > ( dimz, 0 )  ) );
}

void CostsCalculator::UpdateCosts( std::tuple<int, int, int> xyz, float IndividualCost )
{
	UpdateCosts( std::get<0>(xyz), std::get<1>(xyz), std::get<2>(xyz), IndividualCost );
}

void CostsCalculator::UpdateCosts( int x, int y, int z, float IndividualCost )
{
	float TrimmedCost = IndividualCost < CostThreshold ? IndividualCost : CostThreshold;
	int n = NumberOfProjections[x][y][z];
	if( n == 0 )
		LocalCosts[x][y][z] = TrimmedCost;
	else
	{
		float old_value = LocalCosts[x][y][z];
		float new_value = ( n*old_value + TrimmedCost ) / float(n+1);
		LocalCosts[x][y][z] = new_value;
	}
	NumberOfProjections[x][y][z] += 1;
}
