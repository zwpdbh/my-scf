/*
 * ckdtree.h
 *
 *  Created on: Jul 22, 2013
 *      Author: xiaoxintang
 */

#ifndef CKDTREE_H_
#define CKDTREE_H_

#include "common.h"
#include "sdc_kmeans.h"
#include <vector>

using namespace std;

#define N_ITERS 10
#define TRAINT 10

#define SCALE_U 0.2
#define SCALE_D 0.6

class SDCIndex {
public:
    SDCIndex(float *, int, int);
    virtual ~SDCIndex();
    void buildIndex();
    void buildIndex(int, int, int);

    void knnSearch(Matrix<float> &, Matrix<int> &, Matrix<float> &, int, SearchParams &);
    void update(float pre, float precision);
    void getStatics(int, int, int, float *, float *, int, int);
    float *init(float *, float *md = NULL);
    int getClusterSize();
    float estimate(int, float *);
    float estimate(unsigned char *, float *);
    void explore(float *, ResultSet<float> *, float *);

    /**drows is the number of features in the first image, = 10164*/
    int drows;

    /**dcols is 128*/
    int dcols;

    /**n_clusters is the number of clusters specified during clustering in each
     * subspace*/
    int n_clusters;
    int num;
    int total;

    /**a vector of Cluster, each Cluster is a hierarchical cluster*/
    vector<Cluster *> clusters;
    static float scale_u, scale_d;
    static int N_SUBSPACES_, N_CLUSTERS_;

private:
    float *ddata;
    unsigned char *beLongTo;

    static L2<float> distance_;
};

#endif /* CKDTREE_H_ */
