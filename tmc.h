#ifndef __TMC__
#define __TMC__

// GRM Based Template Match Core Funtionality

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

using namespace std;
using namespace cv;

class GRMTM {
public:
    GRMTM();

    void match();
    void feed( Mat* tar );
    void addTemplate( Mat* tar );

    vector<Mat> smat;
private:
    void calcOrientation( int index );
    void calcOrientation();
    void calcTable();
    
    Mat cur_tar;
    
    vector<Mat> vec_template;
    vector<Mat> smap;
    vector<vector<Point3i> > key_point;
    float TABLE[5][31];
    float THRESHOLD = 20.0;
};

#endif