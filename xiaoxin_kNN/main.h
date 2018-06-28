/*
 * IMG.h
 *
 *  Created on: 2/05/2013
 *      Author: rtang
 */

#ifndef IMG_H_
#define IMG_H_

#include <fstream>
#include <stdio.h>

using namespace std;

#include "sdc_index.h"
#include <unistd.h>

#define KNN 2
#define EXACT 16
//#define PRECISION

class IMG {
public:
    char name[50];
    float *data;
    int rows, cols;

    static int ALG;
    static int REP;

    SDCIndex *sdc;

    ~IMG() {
        delete[] data;
        delete sdc;
        rows = 0;
        cols = 0;
    }

    void buildIndex() {
        sdc = new SDCIndex(this->data, this->rows, this->cols);
        sdc->buildIndex();
    }

    /**@param qimg the query image
     * @param iindex holds the index for the query image
     * @param idist
     * @param knn currently, knn = 2
     * @param nthreads use how many threads to do the match*/
    void march(IMG &qimg, int *iindex, float *idist, int knn, int nthreads) {
        float *query = qimg.data;

        // If I need to use numa to allocate memory, this could be the place
        Matrix<float> que(query, qimg.rows, qimg.cols);
        Matrix<int> ind(iindex, qimg.rows, knn);
        Matrix<float> dis(idist, qimg.rows, knn);
        SearchParams par;
        par.checks = 512;
        par.cores = nthreads;

        sdc->knnSearch(que, ind, dis, knn, par);
        //		writeResultsInStream(qimg.rows, knn, iindex, idist);

#ifdef PRECISION
        int size = qimg.rows * EXACT;
        int *index = new int[size];
        float *dist = new float[size];
        readResultsInStream(qimg.rows, EXACT, index, dist);
        Matrix<int> ein(index, qimg.rows, EXACT);
        Matrix<float> edist(dist, qimg.rows, EXACT);

        int num = 0;
        for (int i = 0; i < qimg.rows; ++i) {
          int *indx1 = ein[i];
          int *indx2 = ind[i];
          int cur1 = 0, cur2 = 0;
          while (cur1 < knn) {
            if (indx1[cur1] != indx2[cur2]) {
              ++cur1;
            } else {
              ++cur1;
              ++cur2;
              ++num;
            }
          }
        }
        float pre = num * 1.0f / (qimg.rows * knn);
        printf("%.4f ", pre);

        float variance = 0;
        for (int i = 0; i < qimg.rows; ++i) {
          float *dist1 = edist[i];
          float *dist2 = dis[i];
          for (int j = 0; j < knn; ++j) {
            float tmp = std::pow(dist1[j], 0.5f) - std::pow(dist2[j], 0.5f);
            variance += tmp * tmp;
          }
        }
        variance /= qimg.rows * knn;
        variance = std::pow(variance, 0.5f);
        printf("%.4f\n", variance);

        delete[] index;
        delete[] dist;
#endif
    }

    void readImg() {
        readImgInStream();
    }

    void writeImg() { writeImgInStream(); }

    void generate(char *name_, int rows_, int cols_) {
        strcpy(name, name_);
        rows = rows_;
        cols = cols_;
        int size = rows * cols;
        data = new float[size];
        srand(time(0));
        for (int i = 0; i < size; ++i) {
            data[i] = rand() % 128;
        }
        writeImgInStream();
        delete[] data;
        sleep(2);
    }

private:
    void readImgInStream() {
        ifstream in;
        char tname[50];
        strcpy(tname, this->name);
        strcat(tname, ".bin");
        in.open(tname);

        in.read((char *) &this->rows, sizeof(int));
        in.read((char *) &this->cols, sizeof(int));
        printf("%s %d %d\n", this->name, this->rows, this->cols);

        /**This is where IMG allocate data*/
        int size = this->rows * this->cols;
        /**allocate memory in parallel using OpenMP*/
        this->data = new float[size];
        in.read((char *) this->data, size * sizeof(float));
        in.close();
    }

    void writeImgInStream() {
        ofstream out;
        char tname[50];
        strcpy(tname, name);
        strcat(tname, ".bin");
        out.open(tname);
        out.write((char *) &rows, sizeof(int));
        out.write((char *) &cols, sizeof(int));
        int size = rows * cols;
        out.write((char *) data, size * sizeof(float));
        out.close();
    }

    void readImgInTxt() {
        int read = 0;
        char tname[50];
        strcpy(tname, name);
        strcat(tname, ".txt");
        printf("%s\n", tname);
        if (freopen(tname, "r", stdin)) {
        }
        read = scanf("%d  %d\n", &rows, &cols);
        printf("%d, %d\n", rows, cols);
        data = new float[rows * cols];
        float *cur = data;
        //		read = scanf("[");
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                //				read = scanf("%f%[,;]] ", cur++, &c);
                // digits
                //				read = scanf("%f, ", cur++);
                read = scanf("%f  ", cur++);
                //				if(j == 0){
                //					printf("%.2f\n", *(cur-1));
                //				}
            }
            // digits
            //			scanf("%d\n", &read);
        }
        if (read) {
        }
    }

    void readResultsInStream(int qrows, int knn, int *iindex, float *idist) {
        ifstream in;
        char tname[50];
        strcpy(tname, name);
        strcat(tname, ".exact");
        in.open(tname);

        int rrows, ccols;
        in.read((char *) &rrows, sizeof(int));
        in.read((char *) &ccols, sizeof(int));
        //		if (rrows != qrows || ccols != knn) {
        //			printf("Not the right image pair!!\n");
        //			exit(0);
        //		}
        if (iindex == NULL || idist == NULL) {
            printf("Empty buffer!!\n");
            exit(0);
        }
        int size = qrows * knn;
        in.read((char *) iindex, size * sizeof(int));
        in.read((char *) idist, size * sizeof(float));
        in.close();
    }

    void writeResultsInStream(int qrows, int knn, int *iindex, float *idist) {
        ofstream out;
        char tname[50];
        strcpy(tname, name);
        strcat(tname, ".exact");
        out.open(tname);
        out.write((char *) &qrows, sizeof(int));
        out.write((char *) &knn, sizeof(int));
        int size = qrows * knn;
        out.write((char *) iindex, size * sizeof(int));
        out.write((char *) idist, size * sizeof(float));
        out.close();
    }
};

int IMG::ALG = 0;
int IMG::REP = 1;

#endif /* IMG_H_ */
