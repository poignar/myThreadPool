/**
  ******************************************************************************
  * @file ldw_cluster.h 
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
#ifndef LDW_CLUSTER_H
#define LDW_CLUSTER_H

#include"Dbscan_cluster.h"
#include"graph.hpp"
#include"process.h"
#include"thread_pool.h"
#include <fstream>

#ifndef MPSOC_ZCU104
#include <direct.h>
#include <io.h>
#endif

using namespace std;
using namespace cv;

//计算当前车道线
int getCurrentLane(cv::Point2f headCenterPt, vector<vector<float> > coffes_m,
						float carWidth, int *currentLane, int &warnFlag);

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
void draw_ploy(Mat &src, InputArray coffe, Scalar color);

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
int clusterLane(Mat &src, vector<vector<float> > &coffes_m);

#endif
