#include <iostream>
#include "tmc.h"

vector<Point> find( Mat smat, float th ) {
    vector<Point> matched;
    for ( int row = 0; row < smat.rows; row++ ) {
        for ( int col = 0; col < smat.cols; col++ ) {
            if ( smat.at<float>( row, col ) >= th ) {
                matched.push_back( Point( col, row ) );
            }
        }
    }
    return matched;
}

int main(int argc, const char * argv[]) {
    Mat mat_tmp, mat_tar;

    Point min_pt, max_pt;
	double min_val, max_val;
    
    vector<Point> vec;
    
    mat_tmp = imread("/Users/admin/Desktop/template.jpg");
    mat_tar = imread("/Users/admin/Desktop/target.jpg");

    GRMTM matcher;
    
    matcher.addTemplate(&mat_tmp);
    matcher.feed(&mat_tar);
    matcher.match();

    //vec = find( matcher.smat[0], 0.91 );
    //for ( int i = 0 ; i< vec.size();i++ ) {
    //    rectangle( mat_tar, Rect( vec[i].x, vec[i].y, mat_tmp.cols, mat_tmp.rows ), Scalar( 0, 255, 0 ), 2 );
    //}

    namedWindow("Result");
    minMaxLoc( matcher.smat[0], &min_val, &max_val, &min_pt, &max_pt );
    rectangle( mat_tar, Rect( max_pt.x, max_pt.y, mat_tmp.cols, mat_tmp.rows ), Scalar( 0, 255, 0 ), 2 );
    imshow("Result", mat_tar);
    waitKey(0);
    
    return 0;
}

