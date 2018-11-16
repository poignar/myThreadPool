/**
  ******************************************************************************
  * @file ldw_cluster.cpp 
  * @author wuyuan
  * @mod by zhangwei
  * @version v1.0.0
  * @date 2018-10-26
  * @brief lane cluster
  ******************************************************************************
  * @attention
  *
  * File For Autocruis Software Team Only
  *
  * Copyright (C), 2017-2027, Autocruis Software Team
  ******************************************************************************
**/ 
#include "ldw_cluster.h"

#ifndef CLUSTER_DEBUG
#define CLUSTER_DEBUG 1
#endif

#define MAX_DIS_DEF 12 //全尺寸使用

//===内部函数、结构体声明===+

//创建使用库拟合多项式的用户函数
struct vars_struct {
	double *x;
	double *y;
	double *ey;//数据误差
};//定义用户私有数据结构

struct post5_result{
	vector<vector<cv::Point2f> > result_lanes;
	vector<bool> is_update;
	};

inline double caculate_distance_point_to_line(cv::Point2f point, double A, double B, double C)//直线类型为Ax+by+c=0
{
	return (abs(A*point.x + B*point.y + C) / sqrt(A*A + B*B));
}

inline double L1_distance_points(cv::Point2f x1, cv::Point2f x2)
{
	return(abs(x1.x - x2.x) + abs(x1.y - x2.y));
}

bool isfind(vector<cv::Point2f> points, cv::Point2f pt);

int isfind_id(vector<cv::Point2f> points, cv::Point2f pt);

//找出点集的x范围以及y范围
void find_pt_roi(vector<Point2f> src_pt, double* pmin_x, double* pmax_x, double* pmin_y, double* pmax_y);

//传入的points是按照y值从小到大排列的
post5_result post_5(vector<cv::Point2f> points);

//直线使用单向图聚类
void use_gap_dis_L_MOD(InputArray coffe, vector<vector<Point2f>>& correspond_Point, vector<vector<Point2f>>& result_Point, int length, vector<bool>& is_update);

//寻找拟合点(使用中点)
void find_fit_pt(vector<vector<Point2f>>& input, vector<vector<Point2f>>& output);

void show_res(std::vector<std::vector<cv::Point2f>> &visble_Point, std::vector<std::vector<float>> &coffe, Mat show_im);

//普通的最小二乘多项式拟合-1
bool polynomial_curve_fit(vector<Point2f>& key_point, int n, cv::Mat& A);

#ifdef MPSOC_ZCU104
void polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order);
#endif

//普通的最小二乘多项式拟合-2
Mat polyfit_y(std::vector<cv::Point2f> &chain, int n);

//使用伪逆的多项式拟合,结果系数是 a0,a1,...	(a0+a1*x+a2*x^2+...)
vector <float> fitPoly(const vector <Point> &src, int order);

// 使用伪逆的多项式拟合, 结果系数是 a0, a1, ...	(a0 + a1*y + a2*y ^ 2 + ...)
vector <float> fitPoly_reverse(const vector <Point2f> &src, int order);

bool ascendSort(vector<Point> a, vector<Point> b);

//查找点集中是否存在孤立点，并删除孤立点(孤立点周围的总数小于5)
bool is_exisit_lone_pt(vector<Point2f>& current_pt,Size img_size);

//生成视频的下列函数
void generate_frame(cv::String name,Mat& src_rgb);

void iter_file();

//预处理
void preWork(Mat &src, vector<DBSCAN::Point> &points, Mat &I, Mat &homg);

//密度聚类
void doCluster(vector<DBSCAN::Point> &points, vector<vector<cv::Point2f>> &dbscan_points, Mat I);

//后处理
void postWork(Mat &src, vector<vector<cv::Point2f> > &dbscan_points, vector<vector<float> > &coffes_m, Mat inv_homg, Mat I);

//===内部函数、结构体声明===-



bool isfind(vector<cv::Point2f> points, cv::Point2f pt)
{
	for (int i = 0; i < points.size(); i++)
	{
		if ((int)points[i].x == (int)pt.x && (int)points[i].y == (int)pt.y)
			return true;
	}
	return false;
}

int isfind_id(vector<cv::Point2f> points, cv::Point2f pt)
{
	for (int i = 0; i < points.size(); i++)
	{
		if ((int)points[i].x == (int)pt.x && (int)points[i].y == (int)pt.y)
			return i;
	}
	return -1;
}



//找出点集的x范围以及y范围
void find_pt_roi(vector<Point2f> src_pt, double* pmin_x, double* pmax_x, double* pmin_y, double* pmax_y)
{
	double min_y = 1e12;
	double max_y = 1e-12;
	double min_x = 1e12;
	double max_x = 1e-12;
	for (int j = 0; j < src_pt.size(); j++)
	{
		double current_y = src_pt[j].y;
		double current_x = src_pt[j].x;
		if (current_y < min_y)
		{
			min_y = current_y;

		}
		if (current_y > max_y)
		{
			max_y = current_y;
		}

		if (current_x < min_x)
		{
			min_x = current_x;
		}
		if (current_x > max_x)
		{
			max_x = current_x;
		}
	}
	*pmin_x = min_x;
	*pmax_x = max_x;
	*pmin_y = min_y;
	*pmax_y = max_y;
}



//传入的points是按照y值从小到大排列的
post5_result post_5(vector<cv::Point2f> points)//is_update判断后续使用图分类是否分类，即使用该分类分开后的类，不能再合并
{
	//判断相连与否
	//step-1,查找尖点
	//cout<<"===post_5 thread commit!==="<<endl;
	double current_y = 0, current_x = 0;
	double left_dis = 0, right_dis = 0;

	post5_result res;

	//step-1,获取每个Cluster中y值最小的点,同时获取x的最小值与最大值
	//for (int i = 0; i < points.size(); i++)
	{
		vector<cv::Point2f> temp_lane;
		int index = -1;
		double min_y = 1e12;
		double max_y = 1e-12;
		double min_x = 1e12;
		double max_x = 1e-12;
		for (int j = 0; j < points.size(); j++)
		{
			current_y = points[j].y;
			current_x = points[j].x;
			if (current_y < min_y)
			{
				min_y = current_y;
				index = j;
			}
			if (current_y > max_y)
			{
				max_y = current_y;
			}

			if (current_x < min_x)
			{
				min_x = current_x;
			}
			if (current_x > max_x)
			{
				max_x = current_x;
			}
		}
		int dis_max_index = -1;
		double dis_max = 1e-12;
		cv::Point2f peak = points[index];
#if 1//直接使用水平线划分类
		double dis_xin = max_y - min_y;
			vector<cv::Point2f>fit_points, hou_fit_points;
			int count = 0;
			bool is_break = false;
			bool fit_flag = false;
			bool left_flag = false;
			bool right_flag = false;
			bool continue_flag = false;
			int continue_len = 0;
			double max_gap_dis = -1;
			//for (int j = int(min_y); j <= int(max_y); j++)
#if 0
			for (int j = int(max_y); j >= int(min_y); j--)//从下往上搜索
			{
				vector<cv::Point2f>line_points;
				for (int f = (int)min_x; f <= (int)max_x; f++)
				{
					cv::Point2f pt(f, j);
					if (isfind(points, pt))
						line_points.emplace_back(pt);
				}

				if (line_points.size() > 0)
				{

					continue_len++;

					for (int f = 0; f<line_points.size() - 1; f++)
					{
						double tmp = L1_distance_points(line_points[f], line_points[f + 1]);
						if (tmp >= 2)//存在间隙,保存最远点拟合
						{
							if (tmp>max_gap_dis)
								max_gap_dis = tmp;
							count++;
							continue_flag = true;

							fit_points.push_back(line_points[0]);
							hou_fit_points.push_back(line_points[line_points.size() - 1]);

							break;
						}

					}


				}

			}
#endif

#if 1
			bool isfirst = true;
			bool is_gap = false;//是否有间隙
			bool isend = false;
			vector<cv::Point2f>line_points;//从右向左保存起始点与终止点
			for (int j = points.size()-1; j >= 0; j--)//从下往上搜索,points从后向前x按降序排列（也就是x是从右向左的）
			{
				if (isfirst)
				{
					continue_len++;
					line_points.clear();
					line_points.emplace_back(points[j]);//从右向左保存第一个起始点
					isfirst = false;
					isend = false;
				}
				if (j != 0)//不是最后一个元素，points[j-1]有值
				{
					if (points[j].y != points[j - 1].y)//与下一个元素的y值不相同,说明当前点是结束点，保存
					{
						line_points.emplace_back(points[j]);
						isend = true;
						isfirst = true;
					}
					else//如果与下一个元素的y值相同，查找是否存在间隙
					{
						double tmp=points[j].x - points[j - 1].x;
						if (tmp >= 2)//如果存在间隙，置标志位
						{
							if (tmp>max_gap_dis)
								max_gap_dis = tmp;
							count++;
							is_gap = true;
						}
					}
				}
				else//如果当前是最后一个元素,直接保存，并置结束标志位
				{
					line_points.emplace_back(points[j]);
					isend = true;
				}
				if (is_gap && isend)//如果存在间隙，且isend为真，保存到对应容器中
				{
					fit_points.push_back(line_points[line_points.size() - 1]);
					hou_fit_points.push_back(line_points[0]);
					is_gap = false;

				}


				
			}

#endif

#endif


			if (count > 0) //只要有间隙就分开类别，但后续，保存统计间隔大小和两类的点数比值来确定是否真的需要分开
			{
				fit_flag = true;
			}
			if (fit_flag)//为真，分开类别
			{
				//计算边缘拟合点的均值L
				double mean_fit_x = 0;
				double mean_fit_y = 0;

#if 1//使用拟合直线
				cv::Vec4f lines;
				cv::fitLine(fit_points, lines, CV_DIST_HUBER, 0, 0.01, 0.01);
				double mean_k = lines[0] / lines[1];//x=k*y+b
				double mean_b = lines[2] - lines[3] * mean_k;
#endif

				double hou_k = 0;
				double hou_b = 0;

#if 1//hou的拟合使用直线拟合
				cv::Vec4f line_hou;
				cv::fitLine(hou_fit_points, line_hou, CV_DIST_HUBER, 0, 0.01, 0.01);
				hou_k = line_hou[0] / line_hou[1];//x=k*y+b
				hou_b = line_hou[2] - line_hou[3] * hou_k;
#endif

				//根据类中的点到两条直线的距离分割类
				vector<cv::Point2f>cluster_1;
				vector<cv::Point2f>cluster_2;
				//设置两类的计数
				int mean_cn = 0;
				int hou_cn = 0;

				//for (int g = 0; g < points[i].size(); g++)
				for (int g = points.size() - 1; g >= 0; g--)//由于传入的points是按y升序排列的，而搜索是从下往上进行的，故要从后向前索引
				{

					{
						double dis_1 = caculate_distance_point_to_line(points[g], -1, hou_k, hou_b);
						double dis_2 = caculate_distance_point_to_line(points[g], -1, mean_k, mean_b);//均值点与尖点构成的直线
						if (dis_1 < dis_2)
						{
							hou_cn++;
							cluster_1.push_back(points[g]);
						}
						else
						{
							mean_cn++;
							cluster_2.push_back(points[g]);
						}
					}
				}
				
				if (double(hou_cn) / double(mean_cn) < 0.1 || double(mean_cn) / double(hou_cn) < 0.1)//如果两类的点数比<0.1，就删除点数少的类
				{
					if (double(hou_cn) / double(mean_cn) < 0.1)//mean_cn对应的cluster2，hou_cn对应的cluster1
					{
						res.result_lanes.push_back(cluster_2);
						res.is_update.push_back(true);
					}
					if (double(mean_cn) / double(hou_cn) < 0.1)
					{
						res.result_lanes.push_back(cluster_1);
						res.is_update.push_back(true);
					}
				}
				else if (count < continue_len*0.1 && max_gap_dis <= 4)//如果间隙长度较小，不必分开
				{
					res.result_lanes.push_back(points);
					res.is_update.push_back(false);
				}
				else if (count<continue_len*0.1 && max_gap_dis >= 8)
				{
					res.result_lanes.push_back(points);
					res.is_update.push_back(false);
				}
				else//分开类别
				{
					if (cluster_1.size() > 0)
					{
						res.result_lanes.push_back(cluster_1);
						res.is_update.push_back(true);
					}
					if (cluster_2.size() > 0)
					{
						res.result_lanes.push_back(cluster_2);
						res.is_update.push_back(true);
					}
				}

			}
			else
			{
				res.result_lanes.push_back(points);
				res.is_update.push_back(false);
			}
		}
		//cout<<"===post_5 thread finished==="<<endl;
	return res;

	}





//单向图聚类
void use_gap_dis_L_MOD(InputArray coffe, vector<vector<Point2f>>& correspond_Point, vector<vector<Point2f>>& result_Point, int length, vector<bool>& is_update)//直线使用单向图
{

	Mat m_coffe = coffe.getMat();
	

	m_coffe = m_coffe.reshape(1, length);
	m_coffe.convertTo(m_coffe, CV_32F);
	int coffe_num = m_coffe.cols;
	assert(coffe_num > 1 && m_coffe.rows > 1 && correspond_Point.size() == length);
	
	//cout << m_coffe << endl;
	Graph gp;
	Graph_Cluster gp_cluster(&gp, length);


	for (int i = 0; i < length; i++)
	{
		int current_cluster = i;
		vector<double> current_dis;
		//		if (i == 2)
		//			printf("\n");
		for (int j = 0; j < correspond_Point.size(); j++)
		{

			if (current_cluster != j)
			{
				double max_dis = -1;
				double cu_dis = 0;
				for (int m = 0; m < correspond_Point[j].size(); m++)
				{
					double y_coffe = 0;
					for (int k = 0; k < coffe_num; k++)
						//y_coffe += m_coffe.at<float>(current_cluster, k)*pow(correspond_Point[j][m].x, k);//计算拟合后的值
						y_coffe += m_coffe.at<float>(current_cluster, k)*pow(correspond_Point[j][m].y, k);
					//cu_dis = abs(y_coffe - correspond_Point[j][m].y);
					cu_dis = abs(y_coffe - correspond_Point[j][m].x);
					if (cu_dis>max_dis)
					{
						max_dis = cu_dis;
					}
				}
				//if (max_dis < 15)//最大距离阈值
				current_dis.push_back(max_dis);
			}
		}
#if 0
		//寻找current_dis的最小值
		vector<double>::iterator min_d = min_element(current_dis.begin(), current_dis.end());
		//		if (i == 2)
		//			printf("\n");
		if (*min_d < MAX_DIS_DEF)//最大距离阈值15,合并类
		{
			int dis = distance(current_dis.begin(), min_d);
			int combin_id = dis < current_cluster ? dis : (dis + 1);
			Graph_fill_edge_v(&gp, current_cluster, combin_id);//加入到有向图中

		}
#endif
		for (int k = 0; k < current_dis.size(); k++)
		{
			if (current_dis[k] < MAX_DIS_DEF)//最大距离阈值15,合并类
			{
				int combin_id = k < current_cluster ? k : (k + 1);
				if (abs(current_cluster - combin_id) != 1 || !(is_update[combin_id] && is_update[current_cluster]) )
					gp_cluster.Graph_fill_edge(current_cluster, combin_id);//加入到有向图中

			}
		}
	}

	//搜索有向图的节点，合并类
	vector<int> vis;
	for (int i = 0; i < length; i++)
	{
		vis.clear();
		gp_cluster.wrap_DFS(i, vis);
		//for (int j = 0; j < vis.size(); j++)
			//cout << vis[j] << "  ";
		//cout << endl;
	}
	//查找邻接矩阵的对角值，如果都为1，保存
	vector<vector<int>> id_ju;
	for (int i = 0; i < length; i++)
	{

		for (int j = 0; j < length; j++)
		{
			if (j != i)
			{
				if (gp.arc[i][j] && gp.arc[j][i])
				{
					vector<int> temp_id;
					int fin = max(i, j);
					int pr = min(i, j);
					//bool flag = false;
					int id = -1;

					for (int m = 0; m < id_ju.size(); m++)
					{
						vector<int>::iterator result_pr;
						result_pr = find(id_ju[m].begin(), id_ju[m].end(), pr);
						if (result_pr != id_ju[m].end())
						{
							id = m;
							break;
						}

					}
					if (id >= 0)
					{
						vector<int>::iterator result_fin;
						result_fin = find(id_ju[id].begin(), id_ju[id].end(), fin);
						if (result_fin == id_ju[id].end())//fin不在当前的类中，就加入到其中
						{
							id_ju[id].push_back(fin);
							//flag = true;
						}
					}
					else//pr没有搜索到
					{

						for (int m = 0; m < id_ju.size(); m++)
						{
							vector<int>::iterator result_fin;
							result_fin = find(id_ju[m].begin(), id_ju[m].end(), fin);
							if (result_fin != id_ju[m].end())
							{
								id = m;
								break;
							}

						}
						if (id >= 0)
						{
							id_ju[id].push_back(pr);
						}
						else//fin和pr都没搜索到，直接加入作为新的一类
						{
							temp_id.push_back(pr);
							temp_id.push_back(fin);
							id_ju.push_back(temp_id);
						}
					}

				}
			}
		}
	}
	gp_cluster.Graph_release();
	for (int i = 0; i < id_ju.size(); i++)
	{
		//for (int j = 0; j < id_ju[i].size(); j++)
			//printf("%4d", id_ju[i][j]);
		//printf("\n");
	}
	vector<vector<Point2f>> tmp_Point(correspond_Point.size());
	for (int i = 0; i < correspond_Point.size(); i++)
	{
		tmp_Point[i].assign(correspond_Point[i].begin(), correspond_Point[i].end());
	}
	tmp_Point.swap(result_Point);
	vector<int> remind_id;
	for (int i = 0; i < id_ju.size(); i++)
	{
		for (int j = 0; j < id_ju[i].size(); j++)
		{
			if (j>0)
			{
				result_Point[id_ju[i][0]].insert(result_Point[id_ju[i][0]].end(), result_Point[id_ju[i][j]].begin(), result_Point[id_ju[i][j]].end());
				//result_Point.erase(result_Point.begin() + j);
				remind_id.push_back(id_ju[i][j]);
			}
		}
	}
	std::sort(remind_id.begin(), remind_id.end());
	int count = 0;
	for (int i = 0; i < remind_id.size(); i++)
	{
		int ces = remind_id[i];
		result_Point.erase(result_Point.begin() + (ces - count));
		count++;
	}
}


//寻找拟合点(使用中点)
void find_fit_pt(vector<vector<Point2f>>& input, vector<vector<Point2f>>& output)
{
	double current_y = 0, current_x = 0;
	double left_dis = 0, right_dis = 0;



	for (int i = 0; i < input.size(); i++)
	{
		vector<cv::Point2f> temp_mid;
		//step-1,获取每个Cluster中y值最大最小值,同时获取x的最小值与最大值
		int index = -1;
		double min_y = 1e12;
		double max_y = 1e-12;
		double min_x = 1e12;
		double max_x = 1e-12;
#if 0
		for (int j = 0; j < input[i].size(); j++)
		{
			current_y = input[i][j].y;
			current_x = input[i][j].x;
			if (current_y < min_y)
			{
				min_y = current_y;
				index = j;
			}
			if (current_y > max_y)
			{
				max_y = current_y;
			}

			if (current_x < min_x)
			{
				min_x = current_x;
			}
			if (current_x > max_x)
			{
				max_x = current_x;
			}
		}
		//从左至右，从上到下扫描类
		for (int k = (int)min_y; k <= round(max_y); k++)
		{
			int min_tm = 10000;
			int max_tm = -1;
			for (int m = int(min_x); m <= round(max_x); m++)
			{
				
				if (isfind(input[i], Point2f(m, k)))
				{
					if (m > max_tm)
					{
						max_tm = m;
					}
					if (m < min_tm)
					{
						min_tm = m;
					}
				}
			}
			if (max_tm>=min_tm)
			{
				temp_mid.push_back(Point2f((max_tm + min_tm) / 2.0, k));
			}
			
		}
#endif

#if 1
		temp_mid;
		double current_y = -1;
		bool first = true;
		int start_id = 0;
		int cn = 0;
		for (int j = 0; j < input[i].size(); j++)
		{
			if (first)
			{
				start_id = j;
				first = false;
			}
			if (input[i][j].y != input[i][start_id].y)
			{
				temp_mid.emplace_back((input[i][j - 1].x + input[i][start_id].x) / 2, input[i][start_id].y);
				first = true;
				j--;
			}
			if (j == input[i].size() - 1 && input[i][j].y == input[i][start_id].y)//处理最后是连续和单个点的情况
			{
				temp_mid.emplace_back((input[i][j].x + input[i][start_id].x) / 2, input[i][start_id].y);
			}

		}


#endif
		//扫描完一个类，将中点值加入到输出,由于多次拟合最少需要4个点，只将超过4个点的类加入
		if (temp_mid.size() >= 4)
		{
			output.push_back(temp_mid);
		}
	}
}

void show_res(std::vector<std::vector<cv::Point2f>> &visble_Point, std::vector<std::vector<float>> &coffe, Mat show_im)
{
	//Mat show_im = Mat::zeros(360, 640, CV_8UC3);
	Scalar lin_color(0, 0, 255);
	Scalar point_color(255, 0, 0);
	for (int i = 0; i < visble_Point.size(); i++)
	{
		for (int j = 0; j < visble_Point[i].size(); j++)
			circle(show_im, visble_Point[i][j], 0, point_color, 1);

		Mat scoff(coffe[i]);
		draw_ploy(show_im, scoff, lin_color);
	}
	imshow("res", show_im);
	cv::waitKey();
}


//普通的最小二乘多项式拟合-1
bool polynomial_curve_fit(vector<Point2f>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();

	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}

	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = cv::Mat::zeros(n + 1, 1, CV_32FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	A.convertTo(A, CV_32FC1);
	return true;
}

#ifdef MPSOC_ZCU104
void polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order)
{
    CV_Assert((src_x.rows>0)&&(src_y.rows>0)&&(src_x.cols==1)&&(src_y.cols==1)
            &&(dst.cols==1)&&(dst.rows==(order+1))&&(order>=1));
    Mat X;
    X = Mat::zeros(src_x.rows, order+1,CV_32FC1);
    Mat copy;
    for(int i = 0; i <=order;i++)
    {
        copy = src_x.clone();
        pow(copy,i,copy);
        Mat M1 = X.col(i);
        copy.col(0).copyTo(M1);
    }
    Mat X_t, X_inv;
    transpose(X,X_t);
    Mat temp = X_t*X;
    Mat temp2;
    invert (temp,temp2);
    Mat temp3 = temp2*X_t;
    Mat W = temp3*src_y;
    W.copyTo(dst);

}
#endif


//普通的最小二乘多项式拟合-2
Mat polyfit_y(std::vector<cv::Point2f> &chain, int n)
{
	Mat y(chain.size(), 1, CV_32F, Scalar::all(0));

	/* ********【预声明phy超定矩阵】************************/
	/* 多项式拟合的函数为多项幂函数
	* f(x)=a0+a1*x+a2*x^2+a3*x^3+......+an*x^n
	* 矩阵式: phy*x=y
	*a0、a1、a2......an是幂系数，也是拟合所求的未知量。设有m个抽样点，则：
	* 超定矩阵phy=1 x1 x1^2 ... ...  x1^n
	*           1 x2 x2^2 ... ...  x2^n
	*           1 x3 x3^2 ... ...  x3^n
	*              ... ... ... ...
	*              ... ... ... ...
	*           1 xm xm^2 ... ...  xm^n
	*
	* *************************************************/

	cv::Mat phy(chain.size(), n, CV_32F, Scalar::all(0));
	for (int i = 0; i<phy.rows; i++)
	{
		float* pr = phy.ptr<float>(i);
		for (int j = 0; j<phy.cols; j++)
		{
			pr[j] = pow(chain[i].x, j);
		}
		y.at<float>(i) = chain[i].y;
	}
	Mat fit_mat;
	solve(phy, y, fit_mat, cv::DECOMP_SVD);
	fit_mat.convertTo(fit_mat, CV_32FC1);
	return fit_mat;

}

//使用伪逆的多项式拟合,结果系数是 a0,a1,...  (a0+a1*x+a2*x^2+...)
vector <float> fitPoly(const vector <Point> &src, int order){
	Mat src_x = Mat(src.size(), 1, CV_32F);
	Mat src_y = Mat(src.size(), 1, CV_32F);
	for (int i = 0; i < src.size(); i++){
		src_x.at<float>(i, 0) = (float)src[i].x;
		src_y.at<float>(i, 0) = (float)src[i].y;
	}
	Mat dst = Mat(order+1, 1, CV_32F);;
	polyfit(src_x, src_y, dst, order);//使用伪逆求解
	vector<float> res;
	res.assign( (float*)dst.datastart,(float*)dst.dataend);
	return res;
}

// 使用伪逆的多项式拟合, 结果系数是 a0, a1, ...  (a0 + a1*y + a2*y ^ 2 + ...)
vector <float> fitPoly_reverse(const vector <Point2f> &src, int order){
	Mat src_x = Mat(src.size(), 1, CV_32F);
	Mat src_y = Mat(src.size(), 1, CV_32F);
	for (int i = 0; i < src.size(); i++){
		src_x.at<float>(i, 0) = (float)src[i].y;
		src_y.at<float>(i, 0) = (float)src[i].x;
	}
	Mat dst = Mat(order + 1, 1, CV_32F);;
	polyfit(src_x, src_y, dst, order);//使用伪逆求解
	vector<float> res;
	res.assign((float*)dst.datastart, (float*)dst.dataend);
	return res;
}


bool ascendSort(vector<Point> a, vector<Point> b) {
	return a.size() < b.size();

}

//查找点集中是否存在孤立点，并删除孤立点(孤立点周围的总数小于5)
bool is_exisit_lone_pt(vector<Point2f>& current_pt,Size img_size)
{
	

		//转换点集到图像
		Mat trans_img = Mat::zeros(img_size, CV_8UC1);
		for (int i = 0; i < current_pt.size(); i++)
		{
			trans_img.at<uchar>((int)(current_pt[i].y), (int)(current_pt[i].x)) = 255;
		}
#if 0
		Mat labImg;
		labImg.create(trans_img.size(), CV_32SC1); // bwImg.convertTo( labImg, CV_32SC1 );
		labImg = Scalar(0);
		labImg.setTo(Scalar(1), trans_img);

		const int Rows = trans_img.rows - 1, Cols = trans_img.cols - 1;
		int label = 1;
		vector<int> labelSet;
		labelSet.push_back(0);
		labelSet.push_back(1);
			// the first pass
		int *data_prev = (int*)labImg.data; // 0-th row : int* data_prev = labImg.ptr<int>(i-1);
		int *data_cur = (int*)(labImg.data + labImg.step); // 1-st row : int* data_cur = labImg.ptr<int>(i);
		for (int i = 1; i<Rows; i++){
			data_cur++;
			data_prev++;
			for (int j = 1; j<Cols; j++, data_cur++, data_prev++){
				if (*data_cur != 1)
					continue;
				int left = *(data_cur - 1);
				int up = *data_prev;
				int neighborLabels[2];
				int cnt = 0;
				if (left>1)
					neighborLabels[cnt++] = left;
				if (up>1)
					neighborLabels[cnt++] = up;
				if (!cnt){
					labelSet.push_back(++label);
					labelSet[label] = label;
					*data_cur = label;
					continue;
				}
				int smallestLabel = neighborLabels[0];
				if (cnt == 2 && neighborLabels[1]<smallestLabel)
					smallestLabel = neighborLabels[1];
				*data_cur = smallestLabel;
				// 保存最小等价表
				for (int k = 0; k<cnt; k++){
					int tempLabel = neighborLabels[k];
					int& oldSmallestLabel = labelSet[tempLabel];
					if (oldSmallestLabel > smallestLabel){
						labelSet[oldSmallestLabel] = smallestLabel;
						oldSmallestLabel = smallestLabel;
					}
					else if (oldSmallestLabel<smallestLabel)
						labelSet[smallestLabel] = oldSmallestLabel;
					}
				}
				data_cur++;
				data_prev++;
			}
			// 更新等价对列表,将最小标号给重复区域
			for (size_t i = 2; i < labelSet.size(); i++){
				int curLabel = labelSet[i];
				int preLabel = labelSet[curLabel];
				while (preLabel != curLabel){
					curLabel = preLabel;
					preLabel = labelSet[preLabel];
				}
				labelSet[i] = curLabel;
			}
			// second pass
			data_cur = (int*)labImg.data;
			for (int i = 0; i<Rows; i++){
				for (int j = 0; j<trans_img.cols - 1; j++, data_cur++)
					*data_cur = labelSet[*data_cur];
				data_cur++;
			}
#endif		
		vector< vector< Point> > contours;
		bool flag = false;
		findContours(trans_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		sort(contours.begin(), contours.end(), ascendSort);
		for (int j = 0; j < contours.size(); j++)
		{
			double area = contourArea(contours[j]);
			if (area < 8)
			{
				//Rect rect = boundingRect(contours[j]);
				vector< vector< Point> > contours2;
				vector<Point> contours_lp;
				//contours_lp.push_back(Point(rect.x, rect.y));
				//contours_lp.push_back(Point(rect.x, rect.y + rect.height));
				//contours_lp.push_back(Point(rect.x + rect.width, rect.y + rect.height));
				//contours_lp.push_back(Point(rect.x + rect.width, rect.y));
				//contours2.push_back(contours_lp);
				flag = true;
				contours2.push_back(contours[j]);
				drawContours(trans_img, contours2, -1, Scalar(0, 0, 0), CV_FILLED);
			}
		}

		if (flag)
		{
			//查找点集的范围
			double min_y = 1e12;
			double max_y = 1e-12;
			double min_x = 1e12;
			double max_x = 1e-12;
			find_pt_roi(current_pt, &min_x, &max_x, &min_y, &max_y);
			if (current_pt.size()>0)
				current_pt.clear();
		
			for (int i = (int)min_x; i <= (int)max_x; i++)
			{
				for (int j = (int)min_y; j <= (int)max_y; j++)
				{
					if (trans_img.at<uchar>(j, i) >0)
						current_pt.push_back(Point2f(i, j));
				}
			}
#if 1
			cv::RNG rng(12345);
			cv::Mat xianshi_hou = cv::Mat::zeros(img_size, CV_8UC3);

			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			for (int j = 0; j < current_pt.size(); j++)
			{
				cv::circle(xianshi_hou, cv::Point(current_pt[j].x, current_pt[j].y), 0, color, 1);
			}

			//cv::imshow("houchuli", xianshi_hou);
			//cv::waitKey();
#endif
		}
		return flag;
}

//生成视频的下列函数
/*
void generate_frame(cv::String name,Mat& src_rgb)
{
			
			cout << "process image:" << name << endl;
			Mat src = imread(name, 0);
			//_____________________________________________________________________t透视图变换，为减小计算量,下采样
			Point left_up(471, 440);
			Point right_up(837, 440);
			Point left_down(132, 675);
			Point right_down(1184, 675);

			vector<Point2f>src_point;//透视变换的源点
			src_point.push_back(left_up);
			src_point.push_back(right_up);
			src_point.push_back(left_down);
			src_point.push_back(right_down);


			vector<Point2f>dst_point;//透视变换的目的点
			dst_point.push_back(Point(120, 150));
			dst_point.push_back(Point(220, 150));
			dst_point.push_back(Point(120, 250));
			dst_point.push_back(Point(220, 250));
			//+20，+40
			//		dst_point.push_back(Point(120+20, 150+40));
			//		dst_point.push_back(Point(220+20, 150+40));
			//		dst_point.push_back(Point(120+20, 250+40));
			//		dst_point.push_back(Point(220+20, 250+40));

			//		dst_point.push_back(Point(120+20, 150+40));
			//		dst_point.push_back(Point(170+20, 150+40));
			//		dst_point.push_back(Point(120+20, 180+40));
			//		dst_point.push_back(Point(170+20, 180+40));

			medianBlur(src, src, 3);
			Mat homg = findHomography(src_point, dst_point);//计算透视变换矩阵
			Mat inv_homg = homg.inv(DECOMP_SVD);
			//Mat ends = (Mat_<double>(3, 1) << 1280 * 2, 720, 1);
			Mat ends = (Mat_<double>(3, 1) << 1280 * 2, 720, 1);//扩大范围

			Mat jie = homg*ends;

			//cout << jie << endl;
			Mat hom_src;
			warpPerspective(src, hom_src, homg, Size(jie.at<double>(0, 0) / jie.at<double>(2, 0), jie.at<double>(1, 0) / jie.at<double>(2, 0)), 1);
			//char dest[20] = { "hom_quarter_" };

			//strcat(dest, name);
			Mat half_hom;
			pyrDown(hom_src, half_hom, Size(hom_src.size().width / 2, hom_src.size().height / 2));
			//_____________________________________________________________________t透视变换结束
#if 1
			double tm1, tm2;
			tm1 = getTickCount();
			Mat out_hom;
			cv::threshold(half_hom, half_hom, 80, 255, CV_THRESH_BINARY);
			//precv::filterLines(half_hom, out_hom,11,11,7,7);
			precv::filterLines(half_hom, out_hom, 11, 11, 7, 7);
			//precv::filterLines(half_hom, out_hom, 13, 13, 5, 5);
			Mat lps;
			//threshold(out_hom, lps, 0, 255, CV_THRESH_OTSU);
			int ostuThreshold = precv::OTSU(out_hom);
			ostuThreshold = max(ostuThreshold, 1);
			//float qtileThreshold = precv::getQuantile(out_hom, 0.9);
			//int lowerThreshold = int(qtileThreshold * 255);
			int compThreshold = (ostuThreshold)*0.5+10;

			cv::threshold(out_hom, lps, compThreshold, 255, CV_THRESH_BINARY);
			Mat xins;
			bitwise_and(lps, half_hom, xins);
			tm2 = getTickCount();
			cout << "preprocess time is " << (tm2 - tm1) / (getTickFrequency()) * 1000 << "ms" << endl;
			Mat I = xins.clone();
			//imshow("half_hom", half_hom);
			//imshow("out_hom", out_hom);
			//imshow("lps", lps);
			//imshow("fsf", xins);
			//waitKey();
#endif
//_____________________________________________________________________t初始化Dbscan，并进行密度聚类
			//_____________________________________________________________t step_1:读取有效点，并转化为聚类的输入数据格式
			vector<DBSCAN::Point> points;
			//cv::threshold(half_hom, half_hom, 80, 255, CV_THRESH_BINARY);
			int count = 0;
			//Mat I = half_hom.clone();
			for (int i = 0; i < I.rows; i++)
			{
				uchar* prows = I.ptr<uchar>(i);
				for (int j = 0; j < I.cols; j++)
				{
					if (prows[j])
					{
						points.push_back({ j, i, 0, DBSCAN::NOT_CLASSIFIED });
						//count++;
					}
				}
			}

			//_______________________________________________________________t step_2:初始化参数，聚类
			string n("0");
			//string eps("30");
			string eps("28");
			string minPts("30");

			double t1 = cv::getTickCount();

			DBSCAN::DBCAN dbScan(stoi(n), stod(eps), stoi(minPts), points,0);
			dbScan.run();

			double t2 = cv::getTickCount();
			cout << "dbScan time:" << (t2 - t1) * 1000 / cv::getTickFrequency() << "ms" << endl;

			vector<vector<int> > cluster = dbScan.getCluster();

			//_______________________________________________________________t step_3:显示聚类结果
#if 1
			cv::RNG rng(12345);
			cv::Mat xianshi = cv::Mat::zeros(I.size(), CV_8UC3);

			vector<vector<cv::Point2f>> dbscan_points, res_points;

			for (int i = 0; i < cluster.size(); i++)
			{
				cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				vector<cv::Point2f> temp_points;
				//保存聚类结果到文件
#if 0
				char dir_name[20] = { 0 };
				sprintf(dir_name, "%d", im);
				if (_mkdir(dir_name) || _access(dir_name, 0) != -1)
				{
					char name[20] = { 0 };
					sprintf(name, "%s/%d.xls", dir_name, i);
					ofstream fsm;
					fsm.open(name, ios::out);
					fsm << 0 << "\t" << 0 << endl;

#endif
					for (int j = 0; j < cluster[i].size(); j++)
					{
						int id = cluster[i][j];
						//cv::circle(xianshi, cv::Point(points[id].x, points[id].y), 0, color, 1);
#if 0
						//fsm << points[id].x<< " "<<points[id].y << endl;
						fsm << points[id].x << "\t" << 127 - points[id].y << endl;
#endif
						temp_points.emplace_back(points[id].x, points[id].y);
					}
#if 0
					fsm << 165 << "\t" << 127 << endl;
					fsm.close();
				}
#endif
				dbscan_points.push_back(temp_points);
			}

			//cv::imshow("dbScan show", xianshi);
			//cv::waitKey();
#endif

			//_______________________________________________________________t step_4:分开不正确的聚类(聚类的后处理)
			cv::Mat show = xianshi.clone();
			t1 = cv::getTickCount();

#if 1
			vector<bool> is_update;
			//post_2(dbscan_points, res_points, is_update, show);
			post_5(dbscan_points, res_points, is_update);
#endif

			t2 = cv::getTickCount();
			cout << "post time:" << (t2 - t1) * 1000 / cv::getTickFrequency() << "ms" << endl;

			//_______________________________________________________________t step_5:显示最终的聚类结果
#if 1
			cv::Mat xianshi_2 = cv::Mat::zeros(I.size(), CV_8UC3);
			for (int i = 0; i < res_points.size(); i++)
			{
				cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				for (int j = 0; j < res_points[i].size(); j++)
				{
					cv::circle(xianshi_2, res_points[i][j], 0, color, 1);
				}

			}
			cout << "Dbscan cluster:" << res_points.size() << endl;
			//cv::imshow("post", xianshi_2);
			//cv::waitKey();
#endif
			//_____________________________________________________________________t密度聚类结束

			//_____________________________________________________________________t使用无向图进行再聚类
			//_____________________________________________________________t step_1:对现有类进行直线拟合
			vector<Vec2f> coffe;
			for (int i = 0; i < res_points.size(); i++)
			{
				Vec4f fit;
				fitLine(res_points[i], fit, CV_DIST_HUBER, 0, 0.01, 0.01);
				Vec2f result;
				result[1] = fit[0] / fit[1]; //直线方程为x=k*y+b;
				result[0] = fit[2] - result[1] * fit[3];
				coffe.push_back(result);
			}
			vector<vector<Point2f>> visble_Point;
			if (coffe.size()>1)
				//use_gap_dis_L(coffe, res_points, visble_Point, coffe.size());//直线使用单向图
				use_gap_dis_L_MOD(coffe, res_points, visble_Point, coffe.size(),is_update);//直线使用单向图


			//_____________________________________________________________t step_2:显示聚类结果
#if 1
			cv::Mat xianshi_3 = cv::Mat::zeros(I.size(), CV_8UC3);
			for (int i = 0; i < visble_Point.size(); i++)
			{
				cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				for (int j = 0; j < visble_Point[i].size(); j++)
				{
					cv::circle(xianshi_3, visble_Point[i][j], 0, color, 1);
				}

			}

#if 0
			//cv::Mat xianshi_3 = cv::Mat::zeros(I.size(), CV_8UC3);

			for (int i = 0; i < res_points.size(); i++)
			{
				line(xianshi_3, Point(coffe[i][0], 0), Point(coffe[i][0] + coffe[i][1] * xianshi_3.rows, xianshi_3.rows), Scalar(0, 0, 255), 1);
			}
#endif

			cout << "graph cluster:" << visble_Point.size() << endl;
			//cv::imshow("graph", xianshi_3);
			//cv::waitKey();
#endif
			//使用每类的中点去拟合，这样就减少了拟合的点数
#if 1 
			//由于透视图上存在干扰，所以首先去除干扰
			vector<vector<Point2f>> output;
			find_fit_pt(visble_Point, output);//提取中点
			//变换到原图
#if 1//原图三通道融合
			vector<Mat>merge_img;
			//Mat src_rgb;
			merge_img.push_back(src);
			merge_img.push_back(src);
			merge_img.push_back(src);
			cv::merge(merge_img, src_rgb);

#endif

			vector<vector<Point2f>> res_point;
			for (int j = 0; j < output.size(); j++)
			{
				Mat dst;
				vector<Point2f> temp_2;
				for (int k = 0; k < output[j].size(); k++)
					temp_2.push_back(Point2f(int(output[j][k].x * 2), int(output[j][k].y * 2)));//去除下采样导致的坐标变化
				vector<Point2f> temp_3;
				perspectiveTransform(temp_2, temp_3, inv_homg);
				int cn = 0;
				for (int h = 0; h < temp_3.size(); h++)//剔除错误点
				{
					if (temp_3[h].x<0 || temp_3[h].x>src_rgb.cols - 1 || temp_3[h].y<0 || temp_3[h].y>src_rgb.rows - 1)
					{
						temp_3.erase(temp_3.begin() + h - cn);
						cn++;
					}
				}
				res_point.push_back(temp_3);

#if 1//显示逆变换后的点
				cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				for (int h = 0; h < temp_3.size(); h++)
				{
					cv::circle(src_rgb, temp_3[h], 2, color, 1);
				}
			}
			//imshow("src_r", src_rgb);
			//cv::waitKey();
			//拟合车道


			vector<vector<float>> coffes_m;
			for (int k = 0; k < res_point.size(); k++)
			{
				//反转点，为(y,x)
				int best_coffe_ids = -1;
				double min_f_error = 1e12;
				vector<cv::Point> reverse_pt;
				vector<vector<float>> coffes_ms;
				for (int m = 0; m < res_point[k].size(); m++)
					reverse_pt.emplace_back(res_point[k][m].y, res_point[k][m].x);
				for (int n = 1; n <= 3; n++)
				{
					vector<float> coffe_ls;
					coffe_ls = fitPoly(reverse_pt, n);
					//计算误差
					double f_error = 0;

					for (int h = 0; h < res_point[k].size(); h++)
					{
						double sf_error = 0;
						for (int s = 0; s < coffe_ls.size(); s++)
							sf_error += coffe_ls[s] * pow(res_point[k][h].y, s);
						f_error += abs(sf_error - res_point[k][h].x);
					}
					cout << f_error << endl;
					coffes_ms.push_back(coffe_ls);
					coffe_ls.clear();
					if (f_error < min_f_error)
					{
						min_f_error = f_error;
						best_coffe_ids = n - 1;
					}
				}
				if (best_coffe_ids >= 0)
				{
					coffes_m.push_back(coffes_ms[best_coffe_ids]);
					coffes_ms.clear();
				}


			}
			//显示最后拟合线
			for (int s = 0; s < coffes_m.size(); s++)
			{
				Mat scoff(coffes_m[s]);
				draw_ploy(src_rgb, scoff, Scalar(0, 0, 255));
			}
			//imshow("src_r", src_rgb);
			//waitKey();
			
#endif

#endif

}
*/

/*
void iter_file()
{
	std::vector<cv::String> filenames;
	cv::String folder = "F:\\matlab_result\\*.jpg";
	glob(folder, filenames, true);
	VideoWriter fvw("res_fin1.avi", -1, 20, Size(1280, 720), true);
	for (int i = 0; i < filenames.size(); i++)
	{
		Mat frame;
		generate_frame(filenames[i], frame);
		fvw << frame;
		//imshow("sf", frame);
		waitKey(0);
	}
	fvw.release();
}
*/

void preWork(Mat &src, vector<DBSCAN::Point> &points, Mat &I, Mat &homg)
{
		Point left_up(471, 440);
		Point right_up(837, 440);
		Point left_down(132, 675);
		Point right_down(1184, 675);

		vector<Point2f>src_point;//透视变换的源点
		src_point.push_back(left_up);
		src_point.push_back(right_up);
		src_point.push_back(left_down);
		src_point.push_back(right_down);


		vector<Point2f>dst_point;//透视变换的目的点
		dst_point.push_back(Point(120 , 150) );
		dst_point.push_back(Point(220, 150 ) );
		dst_point.push_back(Point(120, 250) );
		dst_point.push_back(Point(220, 250) );
		//+20，+40
//		dst_point.push_back(Point(120+20, 150+40));
//		dst_point.push_back(Point(220+20, 150+40));
//		dst_point.push_back(Point(120+20, 250+40));
//		dst_point.push_back(Point(220+20, 250+40));

//		dst_point.push_back(Point(120+20, 150+40));
//		dst_point.push_back(Point(170+20, 150+40));
//		dst_point.push_back(Point(120+20, 180+40));
//		dst_point.push_back(Point(170+20, 180+40));

		//medianBlur(src, src, 3);
		//imshow("median", src);
		homg = findHomography(src_point, dst_point);//计算透视变换矩阵
		Mat inv_homg = homg.inv(DECOMP_SVD);
		//Mat ends = (Mat_<double>(3, 1) << 1280 * 2, 720, 1);
		Mat ends = (Mat_<double>(3, 1) << 1280*2 , 720, 1);//扩大范围

		Mat jie = homg*ends;

		//cout << jie << endl;
		Mat hom_src;
		warpPerspective(src, hom_src, homg, Size(jie.at<double>(0, 0) / jie.at<double>(2, 0), jie.at<double>(1, 0) / jie.at<double>(2, 0)), 1);
		//char dest[20] = { "hom_quarter_" };
		medianBlur(hom_src, hom_src, 3);
		//strcat(dest, name);
		Mat half_hom;
		pyrDown(hom_src, half_hom, Size(hom_src.size().width / 2, hom_src.size().height / 2));
		//imshow("pyrDown", half_hom);
//_____________________________________________________________________t透视变换结束
#if 1
		//cv::threshold(half_hom, half_hom, 80, 255, CV_THRESH_BINARY);
		cv::threshold(half_hom, half_hom, 0, 255, CV_THRESH_OTSU);
		Mat out_hom;
		//precv::filterLines(half_hom, out_hom,15,15,13,13);
		//cv::threshold(half_hom, half_hom, 80, 255, CV_THRESH_BINARY);
		//precv::filterLines(half_hom, out_hom, 11, 11, 7, 7);

		precv::filterLines(half_hom, out_hom, 11, 11, 11, 11);
		//precv::filterLines(half_hom, out_hom, 11, 3, 7, 1);

		//precv::filterLines(half_hom, out_hom, 3, 3, 1, 1);
		//precv::filterLines(half_hom, out_hom, 13, 13, 5, 5);
		Mat lps;
		//threshold(out_hom, lps, 0, 255, CV_THRESH_OTSU);
		int ostuThreshold = precv::OTSU(out_hom);
		ostuThreshold = max(ostuThreshold, 1);
		//float qtileThreshold = precv::getQuantile(out_hom, 0.9);
		float qtileThreshold = 0;
		int lowerThreshold = int(qtileThreshold * 255);
		//int compThreshold = (lowerThreshold + ostuThreshold)*0.5+10;
		int compThreshold = ostuThreshold*0.5;
		cv::threshold(out_hom, lps, compThreshold, 255, CV_THRESH_BINARY);
		Mat xins;
		bitwise_and(lps, half_hom, xins);
		I = xins.clone();
		/*imshow("half_hom", half_hom);
		imshow("out_hom", out_hom);
		imshow("lps", lps);
		imshow("fsf", xins);
		waitKey();*/
#endif
//_____________________________________________________________________t初始化Dbscan，并进行密度聚类
		//_____________________________________________________________t step_1:读取有效点，并转化为聚类的输入数据格式

		//cv::threshold(half_hom, half_hom, 80, 255, CV_THRESH_BINARY);
		int count = 0;
		//Mat I = half_hom.clone();
		for (int i = 0; i < I.rows; i++)
		{
			uchar* prows = I.ptr<uchar>(i);
			for (int j = 0; j < I.cols; j++)
			{
				if (prows[j])
				{
					points.emplace_back(j, i, 0, DBSCAN::NOT_CLASSIFIED);
					//count++;
				}
			}
		}
}

void doCluster(vector<DBSCAN::Point> &points, vector<vector<cv::Point2f>> &dbscan_points, Mat I)
{
	//_______________________________________________________________t step_2:初始化参数，聚类
	string n("0");
	//string eps("40");
	string eps("28");
	//string minPts("30");
	string minPts("25");
	vector<vector<int> > cluster;

	#if CLUSTER_DEBUG
	cout<<">>>point size: "<<points.size()<<"<<<"<<endl;
	#endif
	
	DBSCAN::DBCAN dbScan(stod(eps), stoi(minPts), points,0);
	dbScan.run();

	cluster = dbScan.getCluster();
#if 1
		cv::RNG rng(12345);
		cv::Mat xianshi = cv::Mat::zeros(I.size(), CV_8UC3);

		for (int i = 0; i < cluster.size(); i++)
		{
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			vector<cv::Point2f> temp_points;
			//保存聚类结果到文件
#if 0
			char dir_name[20] = { 0 };
			sprintf(dir_name, "%d", im);
			if (_mkdir(dir_name) || _access(dir_name,0)!=-1)
			{
				char name[20] = { 0 };
				sprintf(name, "%s/%d.xls", dir_name, i);
				ofstream fsm;
				fsm.open(name, ios::out);
				fsm << 0<< "\t" << 0 << endl;

#endif
//===mod op===
				/*for (int j = 0; j < cluster[i].size(); j++)
				{
					int id = cluster[i][j];
					cv::circle(xianshi, cv::Point(points[id].x, points[id].y), 0, color, 1);
#if 0
					//fsm << points[id].x<< " "<<points[id].y << endl;
					fsm << points[id].x << "\t" << 127-points[id].y << endl;
#endif
					temp_points.emplace_back(points[id].x, points[id].y);
				}*/
//===mod new===
				for(auto & id :cluster[i])
					{
						cv::circle(xianshi, cv::Point(points[id].x, points[id].y), 0, color, 1);
						temp_points.emplace_back(points[id].x, points[id].y);
					}

//===mod ed===

#if 0
				fsm << 165<< "\t" << 127 << endl;
				fsm.close();
			}
#endif
			dbscan_points.emplace_back(temp_points);
		}
		
		//cv::imshow("dbScan show", xianshi);
		//cv::waitKey();
		
#endif
}

void postWork(Mat &src, vector<vector<cv::Point2f> > &dbscan_points, vector<vector<float> > &coffes_m, Mat inv_homg, Mat I)
{
		//_______________________________________________________________t step_4:分开不正确的聚类(聚类的后处理)
		

		vector<vector<cv::Point2f>> res_points;

		vector<bool> is_update;
		
		//post_5(dbscan_points, res_points, is_update);
		//===线程池方式实现===start
		vector< std::future<post5_result> > results_post5;
		//vector<vector<vector<cv::Point2f> > > poolPt;
		//vector<vector<bool> > poolFlag;

		std::threadpool postPool(dbscan_points.size());
		for (int i = 0; i < dbscan_points.size(); ++i) {
            results_post5.emplace_back(
                postPool.commit(post_5, dbscan_points[i])
                );
        }
        //std::cout << " =======  commit all task ========= " << std::this_thread::get_id() << std::endl;

        for (auto && result : results_post5){
			//std::cout << result.get() << ' ';
			post5_result res = result.get();
			for(int j = 0; j < res.result_lanes.size(); ++j)
				{
					res_points.emplace_back(res.result_lanes[j]);
				}
				

			for(int k = 0; k < res.is_update.size(); ++k)
				{
					is_update.push_back(res.is_update[k]);
				}
		}
		vector<std::future<post5_result> >().swap(results_post5);

        std::cout << std::endl;

		//===线程池方式实现===end

		cv::RNG rng(12345);

		//_______________________________________________________________t step_5:显示最终的聚类结果
#if 1
				cv::Mat xianshi_2 = cv::Mat::zeros(I.size(), CV_8UC3);
				for (int i = 0; i < res_points.size(); i++)
				{
					cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
					for (int j = 0; j < res_points[i].size(); j++)
					{
						cv::circle(xianshi_2, res_points[i][j], 0, color, 1);
					}
		
				}

				//cv::imshow("post", xianshi_2);
				//cv::waitKey();
#endif
		
		//_____________________________________________________________________t密度聚类结束
		
		//_____________________________________________________________________t使用无向图进行再聚类
				//_____________________________________________________________t step_1:对现有类进行直线拟合
				vector<Vec2f> coffe;
#if 1//使用拟合直线作用于无向图
				for (int i = 0; i < res_points.size(); i++)
				{
					Vec4f fit;
					fitLine(res_points[i], fit, CV_DIST_HUBER, 0, 0.01, 0.01);
					Vec2f result;
					result[1] = fit[0] / fit[1]; //直线方程为x=k*y+b;
					result[0] = fit[2] - result[1] * fit[3];
					coffe.push_back(result);
				}

#endif
		
#if 0//使用首尾中点的直线方程作用于无向图
				for (int i = 0; i < res_points.size(); i++)//res_points中点集已按y升序排列
				{
					int cmt = 0;
					//寻找首部的中点
					double starty = res_points[i][0].y;
					double sum_start_x = 0;
					for (int k = 0; k < res_points[i].size(); k++)
					{
						if (abs(res_points[i][k].y - starty)<0.1)
						{
							cmt++;
							sum_start_x += res_points[i][k].x;
						}
						else
							break;
					}
					if (cmt>0)
						sum_start_x /= cmt;
					//寻找尾部的中点
					cmt = 0;
					double endy = res_points[i][res_points[i].size()-1].y;
					double sum_end_x = 0;
					for (int k = res_points[i].size() - 1; k >=0; k--)
					{
						if (abs(res_points[i][k].y - endy)<0.1)
						{
							cmt++;
							sum_end_x += res_points[i][k].x;
						}
						else
							break;
					}
					if (cmt>0)
						sum_end_x /= cmt;
		
					//计算方程
					if (sum_end_x && sum_start_x && (endy - starty))
					{
						Vec2f result;
						result[1] = (sum_end_x-sum_start_x) / (endy-starty); //直线方程为x=k*y+b;
						result[0] = sum_end_x - endy*result[1];
						coffe.push_back(result);
					}
				}
#endif
		
#if 0//使用中点的直线方程作用于无向图
				vector<vector<Point2f>> zhongdian;
				find_fit_pt(res_points, zhongdian);
		
				for (int i = 0; i < zhongdian.size(); i++)
				{
					Vec4f fit;
					fitLine(zhongdian[i], fit, CV_DIST_HUBER, 0, 0.01, 0.01);
					Vec2f result;
					result[1] = fit[0] / fit[1]; //直线方程为x=k*y+b;
					result[0] = fit[2] - result[1] * fit[3];
					coffe.push_back(result);
				}
#endif
		
#if 0//使用最优次数拟合作用于无向图，最高2次,该方法并不适用
				vector<Vec3f> coffe;
				vector<vector<Point2f>> zhongdian;
				find_fit_pt(res_points, zhongdian);
				for (int i = 0; i < zhongdian.size(); i++)
				{
			
					vector<Vec3f> result;
					vector<double> result_vals;
					for (int ct = 1; ct < 3; ct++)
					{
						double sum_error = 0;
						vector<float> result_t;
						result_t = fitPoly_reverse(zhongdian[i], ct);
						for (int k = 0; k < zhongdian[i].size(); k++)
						{
							double sf_error = 0;
							for (int s = 0; s < result_t.size(); s++)
								sf_error += result_t[s] * pow(zhongdian[i][k].y, s);
							sum_error = abs(sf_error - zhongdian[i][k].x);
						}
						if (ct == 1)
						{
							result_t.push_back(0);
						}
						Vec3f res_jian;
						res_jian[0] = result_t[0];
						res_jian[1] = result_t[1];
						res_jian[2] = result_t[2];
						result.push_back(res_jian);
						result_vals.push_back(sum_error);
					}
					if (result_vals[0]<=result_vals[1])
						coffe.push_back(result[0]);
					else
						coffe.push_back(result[1]);
				}
#endif
				vector<vector<Point2f>> visble_Point;
				if (coffe.size()>1)
					{
						//use_gap_dis_L(coffe, res_points, visble_Point, coffe.size());//直线使用单向图
						
						use_gap_dis_L_MOD((InputArray)coffe, res_points, visble_Point, coffe.size(), is_update);
						//use_gap_dis_L_MOD((InputArray)coffe, zhongdian, visble_Point, coffe.size(), is_update);
					}
					
				else
					res_points.swap(visble_Point);
					//zhongdian.swap(visble_Point);
		
		
				//_____________________________________________________________t step_2:显示聚类结果
#if 1
				cv::Mat xianshi_3 = cv::Mat::zeros(I.size(), CV_8UC3);

				for (int i = 0; i < visble_Point.size(); i++)
				{
					cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
					for (int j = 0; j < visble_Point[i].size(); j++)
					{
						cv::circle(xianshi_3, visble_Point[i][j], 0, color, 1);
					}
		
				}

				//cv::imshow("graph", xianshi_3);
				//cv::waitKey();
#endif
				//使用每类的中点去拟合，这样就减少了拟合的点数
#if 1 
				//由于透视图上存在干扰，所以首先去除干扰

				vector<vector<Point2f>> output;
				find_fit_pt(visble_Point, output);//提取中点

				//变换到原图
#if 1//原图三通道融合
				vector<Mat> merge_img;
			 	Mat src_rgb;
				merge_img.push_back(src);
				merge_img.push_back(src);
				merge_img.push_back(src);
				cv::merge(merge_img, src_rgb);
				Mat srg = src_rgb.clone();
#endif
				
				vector<vector<Point2f>> res_point;
				for (int j = 0; j < output.size(); j++)
				{
					Mat dst;
					vector<Point2f> temp_2;
					for (int k = 0; k < output[j].size(); k++)
						temp_2.push_back(Point2f(int(output[j][k].x * 2), int(output[j][k].y * 2)));//去除下采样导致的坐标变化
					vector<Point2f> temp_3;
					perspectiveTransform(temp_2, temp_3, inv_homg);
					int cn = 0;
					for (int h = 0; h < temp_3.size(); h++)//剔除错误点
					{
						if (temp_3[h].x<0 || temp_3[h].x>src_rgb.cols - 1 || temp_3[h].y<0 || temp_3[h].y>src_rgb.rows - 1)
						{
							temp_3.erase(temp_3.begin() + h - cn);
							cn++;
						}
					}
					res_point.push_back(temp_3);
					
		
#if 1//显示逆变换后的点
					cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
					for (int h = 0; h < temp_3.size(); h++)
					{
						cv::circle(src_rgb, temp_3[h], 2, color, 1);
					}
				}

				//imshow("src_r", src_rgb);
				//cv::waitKey();
				//拟合车道
		
				for (int k = 0; k < res_point.size(); k++)
				{
					//反转点，为(y,x)
					int best_coffe_ids = -1;
					double min_f_error = 1e12;
					vector<cv::Point> reverse_pt;
					vector<vector<float>> coffes_ms;
					for (int m = 0; m < res_point[k].size(); m++)
						reverse_pt.emplace_back(res_point[k][m].y, res_point[k][m].x);
					for (int n = 1; n <= 3; n++)
					{
						vector<float> coffe_ls;
						coffe_ls=fitPoly(reverse_pt, n);
						//计算误差
						double f_error = 0;
#if 0	
						for (int h = 0; h < res_point[k].size(); h++)
						{
							double sf_error = 0;
							for (int s = 0; s < coffe_ls.size(); s++)
								sf_error += coffe_ls[s] * pow(res_point[k][h].y, s);
							f_error += abs(sf_error - res_point[k][h].x);
						}
#endif
		
#if 1	
						for (int h = 0; h < reverse_pt.size(); h++)
						{
							double sf_error = 0;
							for (int s = 0; s < coffe_ls.size(); s++)
								sf_error += coffe_ls[s] * pow(reverse_pt[h].x, s);
							f_error += abs(sf_error - reverse_pt[h].y);
						}
#endif
						//cout << f_error / res_point[k].size() << endl;
						coffes_ms.push_back(coffe_ls);
						coffe_ls.clear();
						if (f_error < min_f_error)
						{
							min_f_error = f_error;
							best_coffe_ids = n - 1;
						}
					}
#if 1
					bool again = false;
					if (min_f_error / res_point[k].size() >10)//如果拟合的平均误差超过阈值，说明聚类结果可能是错误的，需要对错误的类进行再次聚类
					{
						again = true;
						vector<vector<cv::Point2f>> dbscan_points, dbscanh_points;
						if (!is_exisit_lone_pt(visble_Point[k], I.size()) && visble_Point[k].size() > 0)
						{
							vector<DBSCAN::Point> points_af;
							string n("0");
							string eps("28");
							string minPts("5");
		
							for (int h = 0; h < visble_Point[k].size(); h++)
								points_af.push_back({ visble_Point[k][h].x, visble_Point[k][h].y, 0, DBSCAN::NOT_CLASSIFIED });
		
							DBSCAN::DBCAN dbScan_af(stod(eps), stoi(minPts), points_af, 1);
		
							dbScan_af.run();
		
							vector<vector<int> > cluster_af = dbScan_af.getCluster();
		
							//_______________________________________________________________t step_3:显示聚类结果
#if 1
							cv::RNG rng(12345);
							cv::Mat xianshi_hou = cv::Mat::zeros(I.size(), CV_8UC3);
		
		
							for (int i = 0; i < cluster_af.size(); i++)
							{
								cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
								vector<cv::Point2f> temp_points;
								for (int j = 0; j < cluster_af[i].size(); j++)
								{
									int id = cluster_af[i][j];
									cv::circle(xianshi_hou, cv::Point(points_af[id].x, points_af[id].y), 0, color, 1);
									temp_points.emplace_back(points_af[id].x, points_af[id].y);
								}
		
								dbscan_points.push_back(temp_points);
							}
		
							//cv::imshow("dbScan hou", xianshi_hou);
							//cv::waitKey();
						}
		
#endif
						else
						{
							if (visble_Point[k].size() > 0)
								dbscan_points.push_back(visble_Point[k]);
						}
						if (dbscan_points.size() > 0)
						{
						
							find_fit_pt(dbscan_points, dbscanh_points);//提取中点
							coffes_ms.clear();
						}
		
							for (int k = 0; k < dbscanh_points.size(); k++)
							{
								vector<cv::Point2f> tmps, tmpk;
								for (int h = 0; h < dbscanh_points[k].size(); h++)
									tmps.push_back(Point2f(int(dbscanh_points[k][h].x * 2), int(dbscanh_points[k][h].y * 2)));//去除下采样导致的坐标变化
								vector<Point> temp_reverse;
								perspectiveTransform(tmps, tmpk, inv_homg);
								int cn = 0;
								for (int h = 0; h < tmpk.size(); h++)//剔除错误点
								{
									if (tmpk[h].x<0 || tmpk[h].x>src_rgb.cols - 1 || tmpk[h].y<0 || tmpk[h].y>src_rgb.rows - 1)
									{
										tmpk.erase(tmpk.begin() + h - cn);
										cn++;
									}
								}
								cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		
								for (int h = 0; h < tmpk.size(); h++)
								{
									cv::circle(srg, tmpk[h], 2, color, 1);
								}
		
								//反转点拟合
								for (int m = 0; m < tmpk.size(); m++)
									temp_reverse.emplace_back(tmpk[m].y, tmpk[m].x);
								for (int n = 1; n <= 3; n++)
								{
									vector<float> coffe_ls;
									coffe_ls = fitPoly(temp_reverse, n);
									//计算误差
									double f_error = 0;
		
									for (int h = 0; h < temp_reverse.size(); h++)
									{
										double sf_error = 0;
										for (int s = 0; s < coffe_ls.size(); s++)
											sf_error += coffe_ls[s] * pow(temp_reverse[h].x, s);
										f_error += abs(sf_error - temp_reverse[h].y);
									}
									//cout << f_error / res_point[k].size() << endl;
									coffes_ms.push_back(coffe_ls);
									coffe_ls.clear();
									if (f_error < min_f_error)
									{
										min_f_error = f_error;
										best_coffe_ids = n - 1;
									}
								}
								if (best_coffe_ids >= 0)
								{
									//coffes_m.pop_back();
									//coffes_m.erase(coffes_m.end());
		
									coffes_m.push_back(coffes_ms[best_coffe_ids]);
									coffes_ms.clear();
								}
		
								vector<Point> zhi_pt;
								for (int wg = srg.rows / 2; wg < srg.rows; wg++)
								{
									double x_jie = 0;
									for (int ci = 0; ci < coffes_m[coffes_m.size() - 1].size(); ci++)
									{
										x_jie += coffes_m[coffes_m.size() - 1][ci] * pow(wg, ci);
									}
									if (int(x_jie) >= 0 && int(x_jie) <= srg.cols - 1)
										zhi_pt.push_back(Point(x_jie, wg));
								}
								for (int hg = 0; hg < zhi_pt.size() - 1; hg++)
								{
									line(srg, zhi_pt[hg], zhi_pt[hg + 1], Scalar(0, 0, 255), 2);
								}
								//imshow("zhijie", srg);
								//waitKey();
							}
						
						
		
					}
#endif
					if (best_coffe_ids >= 0 && !again)
					{
						coffes_m.push_back(coffes_ms[best_coffe_ids]);
						coffes_ms.clear();
					}
					
		
				}


#endif
				
#endif

}

//计算当前车道线和压线告警
int getCurrentLane(cv::Point2f headCenterPt, vector<vector<float> > coffes_m, float carWidth, int *currentLane, int &warnFlag)
{	
	//拟合参数coffes_m: x = a0 + a1*y + a2*y^2 + a3*y^3
	int found = 0;
	int numLane = coffes_m.size();
	//vector<float> crossPtLeft;
	//vector<float> crossPtRight;
	//int[2] currentLane; //index 0:left, 1:right;
	float leftMax = -65535;
	float rightMin = 65535;
	if(numLane >= 2)
		{
			for(int i = 0; i < numLane; ++i)
				{
					float tempX = 0.0;
					for(int j = 0; j < coffes_m[i].size(); ++j)
						{
							tempX += coffes_m[i][j] * pow(headCenterPt.y, j);
						}
					
					if(tempX <= headCenterPt.x && tempX >= leftMax)
						{
							leftMax = tempX;
							currentLane[0] = i;
						}
					if(tempX >= headCenterPt.x && tempX <= rightMin)
						{
							rightMin = tempX;
							currentLane[1] = i;
						}
				}
			if(leftMax != -65535 && rightMin != 65535)
				{
					//currentLanePT[0] = leftMax;
					//currentLanePT[1] = rightMin;
					found = 1;
					float halfWidth = carWidth / 2;
					//左压线1
					if(leftMax >= headCenterPt.x - halfWidth && rightMin >= headCenterPt.x + halfWidth)
						{
							warnFlag = 1;
						}
					//右压线2
					else if(leftMax <= headCenterPt.x - halfWidth && rightMin <= headCenterPt.x + halfWidth)
						{
							warnFlag = 2;
						}
					else
						{
							warnFlag = 0;
						}
				}
		}
	else
		{
#if CLUSTER_DEBUG
			cout<<"===Less than 2 line detected! ==="<<endl;
#endif
		}

	return found;
	
}

/***************************************************
 * Function : draw_ploy
 * Author : wuyuan
 * mod by zhangw
 * Creat Date : 2018/10/26  
 * Description : 根据拟合参数画出车道线
 * In-Parameter :原始语义分割图像src，拟合参数coffe，线条颜色color
 * Return : 带车道线的图像src
 * Modify : none
 **************************************************/
void draw_ploy(Mat &src, InputArray coffe, Scalar color)
{
	assert(coffe.type() == CV_32F);
	Mat sd = coffe.getMat();
	//cout <<">>> sd size "<<sd.size()<<": "<< sd <<" <<<"<< endl;
	int wid = src.cols;
	int hei = src.rows;
	vector<Point> valid_point;
	//cout << "coffe:" << sd << endl;
	//for (int i = 0; i < wid; i++)
	//for (int i = 0; i < hei; i++)
	for (int i = hei / 2.0; i < hei; i++)//限制画线区域
	{
		double y_temp = 0;
		for (int j = 0; j < sd.rows; j++)
			//for (int j = sd.rows-1; j >= 0; j--)
			//y_temp += sd.at<float>(j, 0)*pow(i, sd.rows - 1 - j);//次数从大到小 ,a0*x^3+a1*x^2+...
			y_temp += sd.at<float>(j, 0)*pow(i,j);//次数从小到大 ,a0+a1*x+...
		if (y_temp >= 0)
		{
			int y = round(y_temp);
			//valid_point.push_back(Point(i, y));
			valid_point.push_back(Point(y, i));
		}
		//cout << i << endl;
	}
	
	for (int i = 0; i < int(valid_point.size() - 1); i++)
	{
		line(src, valid_point[i], valid_point[i + 1], color, 2);
	}
}

/***************************************************
 * Function : clusterLane
 * Author : wuyuan
 * mod by zhangw
 * Creat Date : 2018/10/26  
 * Description : 计算拟合车道线参数
 * In-Parameter : 原始语义分割图像src
 * Return : 拟合出的多项式参数数组coffes_m
 * Modify : none
 **************************************************/
int clusterLane(Mat &src, vector<vector<float> > &coffes_m)
{

//参数定义
	int found_lane = 0;							//是否找到车道线，0：没找到，1：找到
	vector<DBSCAN::Point> points;				//待聚类目标点集
	Mat I;										//透视变换I矩阵
	Mat homg;									//透视变换矩阵
	vector<vector<cv::Point2f>> dbscan_points;	//密度聚类结果点集
	Mat inv_homg;								//透视反变换矩阵
	
//预处理
	preWork(src, points, I, homg);

//密度聚类
	doCluster(points, dbscan_points, I);

//后处理
	inv_homg = homg.inv(DECOMP_SVD);
	postWork(src, dbscan_points, coffes_m, inv_homg, I);

	if(!coffes_m.empty())
		{
			found_lane = 1;
		}
	return found_lane;
	

}

