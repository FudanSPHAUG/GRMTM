#include "tmc.h"
#include <iostream>

template<class Type>
Type max( Type* arr, int len ) {
    Type m = arr[0];
	for (int i = 1; i < len; i++)
		if (arr[i] > m)
			m = arr[i];
	return m;
}

inline Mat gX( Mat tar ) {
    Mat ret;
    ret = Mat::zeros( tar.rows, tar.cols, CV_8UC1 );
    
    Mat roi = ret(Rect(1, 1, tar.cols-2, tar.rows-2));
    Rect r1(2, 1, tar.cols-2, tar.rows-2);
    Rect r2(0, 1, tar.cols-2, tar.rows-2);
    roi = tar(r1) - tar(r2);
    roi /= 2;
    return ret;
}

inline Mat gY( Mat tar ) {
    Mat ret;
    ret = Mat::zeros( tar.rows, tar.cols, CV_8UC1 );
    
    Mat roi = ret(Rect(1, 1, tar.cols-2, tar.rows-2));
    Rect r1(1, 2, tar.cols-2, tar.rows-2);
    Rect r2(1, 0, tar.cols-2, tar.rows-2);
    roi = tar(r1) - tar(r2);
    roi /= 2;
    return ret;
}


GRMTM::GRMTM() {
    calcTable();
}

void GRMTM::addTemplate( Mat* tmp ) {
    Mat temp = *tmp;
    int cur_idx = (int)vec_template.size();
    cvtColor( temp, temp, CV_BGR2GRAY );
    vec_template.push_back( temp );
    calcOrientation( cur_idx );
}

void GRMTM::feed( Mat* tar ) {
    cvtColor( *tar, cur_tar, CV_BGR2GRAY );
    calcOrientation();
}

void GRMTM::match() {

    
    Mat similarity, roi;
    int h, w;
    
    w = smap[0].cols;
    h = smap[0].rows;
    
    for ( int i = 0; i < key_point.size(); i++ ) {
        similarity = Mat::zeros( h-vec_template[i].rows, w-vec_template[i].cols, CV_32FC1 );
        for ( int j = 0; j < key_point[i].size(); j++ ) {
            roi = smap[key_point[i][j].z]( Rect( key_point[i][j].x,
                                                 key_point[i][j].y,
                                                 w-vec_template[i].cols,
                                                 h-vec_template[i].rows ) );
            similarity += roi;
        }
        similarity /= key_point[i].size();
        smat.push_back( similarity );
    }
    

}

void GRMTM::calcOrientation( int index ) {
    Mat& src = vec_template[index];
    Mat grad_x, grad_y, mat_ori;
    float grad, ori;
    vector<Point3i> kp;
    
    grad_x = gX( src );
    grad_y = gY( src );
    
    for ( int row = 0; row < grad_x.rows;row++ ) {
        for ( int col = 0; col < grad_x.cols; col++ ) {
            grad = sqrt( pow( grad_x.at<uint8_t>( row, col ), 2 ) +
                         pow( grad_y.at<uint8_t>( row, col ), 2 ) );
            if ( grad > THRESHOLD && grad_y.at<uint8_t>( row, col ) > 0 ) {
                ori = acos( grad_x.at<uint8_t>( row, col ) / grad );
                ori = 1 << (int)floor( ori / 0.62831852 );
                if ( ori != 0 ) {
                    kp.push_back( Point3i( col, row, (int)ori-1 ) );
                }
            }
        }
    }
    
    key_point.push_back( kp );
}

void GRMTM::calcOrientation() {
    Mat grad_x, grad_y, mat_ori;
    float grad, ori;
    int rows, cols;
    
    rows = cur_tar.rows;
    cols = cur_tar.cols;
    
    mat_ori = Mat( rows, cols, CV_8UC1 );

    grad_x = gX( cur_tar );
    grad_y = gY( cur_tar );
    
    smap.clear();
    
    smap.push_back( Mat::zeros( rows, cols, CV_32FC1 ) );
    smap.push_back( Mat::zeros( rows, cols, CV_32FC1 ) );
    smap.push_back( Mat::zeros( rows, cols, CV_32FC1 ) );
    smap.push_back( Mat::zeros( rows, cols, CV_32FC1 ) );
    smap.push_back( Mat::zeros( rows, cols, CV_32FC1 ) );
    
    
    uchar* data = mat_ori.data;
    int step = (int)mat_ori.step;
    for ( int row = 1; row < rows; row++ ) {
        for ( int col = 1; col < cols; col++ ) {
            grad = sqrt( pow( grad_x.at<uint8_t>( row, col ), 2 ) +
                         pow( grad_y.at<uint8_t>( row, col ), 2 ) );
            if ( grad > THRESHOLD && grad_y.at<uint8_t>( row, col ) > 0 ) {
                ori = acos( grad_x.at<uint8_t>( row, col ) / grad );
                if ( ori != 0 ) {
                    uint8_t val = 1 << (int)floor( ori / 0.62831852 );
                    
                    data[col  ] |= val;
                    data[col-1] |= val;
                    data[col+1] |= val;
                    
                    data -= step;
                    data[col  ] |= val;
                    data[col-1] |= val;
                    data[col+1] |= val;
                    
                    data += 2 * step;
                    data[col  ] |= val;
                    data[col-1] |= val;
                    data[col+1] |= val;
                    
                    data -= step;
                }
            }
        }
        data += step;
    }
    
    data = mat_ori.data;
    step = (int)mat_ori.step;
    for ( int row = 0; row < rows; row++ ) {
        for ( int col = 0; col < cols; col++ ) {
            for ( int i = 0; i < 5; i++) {
                int o = data[col];
                if ( o != 0 )
                    smap[i].at<float>( row, col ) = TABLE[i][o-1];
                else
                    smap[i].at<float>( row, col ) = 0;
            }
        }
        data += step;
    }
}

void GRMTM::calcTable() {
	float angle;
    float arr[5];
	
	for ( int i = 0; i < 5; i++ ) {
		angle = 0.62831852 * i + 0.31415926;
		for ( int j = 1; j < 32; j++ ) {
			for ( int k = 0; k < 5; k++ ) {
				if ( ( j & ( 1 << k ) ) >> k )
					arr[k] = abs( cos( angle - 0.62831852 * k - 0.31415926 ) );
				else
					arr[k] = 0;
			}
			TABLE[i][j-1] = max( arr, 5 );
		}
	}
}