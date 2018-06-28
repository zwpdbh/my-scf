/*
 * ckdtree.cpp
 *
 *  Created on: Jul 22, 2013
 *      Author: xiaoxintang
 */

#include "cmath"
#include "sdc_index.h"

float     SDCIndex::scale_u = SCALE_U;
float     SDCIndex::scale_d = SCALE_D;
int       SDCIndex::N_SUBSPACES_ = 16;
int       SDCIndex::N_CLUSTERS_ = 8;
L2<float> SDCIndex::distance_;

struct Compare {
    Compare(float *d) {
        distances = d;
    }

    float *distances;

    bool operator()(int i, int j) {
        return (distances[i] < distances[j]);
    }
};

bool compare(Cluster *c1, Cluster *c2) {
    return (c1->variance > c2->variance);
}

SDCIndex::SDCIndex(float *data_, int rows_, int cols_) {
    this->ddata = data_;
    this->drows = rows_;
    this->dcols = cols_;
    this->beLongTo = NULL;
    this->n_clusters = this->num = this->total = 0;
}

SDCIndex::~SDCIndex() {
    ddata = NULL;
    drows = dcols = 0;
    for (size_t i = 0; i < clusters.size(); ++i) {
        clusters[i]->~Cluster();
    }
    clusters.clear();
    if (beLongTo != NULL) {
        delete[] beLongTo;
    }
}

void SDCIndex::buildIndex() {
    // build clusters:
    cout << "N_SUBSPACES_ = " << N_SUBSPACES_ << endl;
    cout << "N_CLUSTERS_ = " << N_CLUSTERS_ << endl;
    cout << "N_ITERS = " << N_ITERS << endl;
    buildIndex(N_SUBSPACES_, N_CLUSTERS_, N_ITERS);
}

/**build index for current IMG
 * IMG::sdc->buildIndex();
 * where sdc is SDCIndex *sdc;
 */
void SDCIndex::buildIndex(int n_subspaces, int n_clusters_, int iter_) {
    /**free previous clusters*/
    for (size_t i = 0; i < clusters.size(); ++i) {
        clusters[i]->~Cluster();
    }
    clusters.clear();
    if (beLongTo != NULL) {
        delete[] beLongTo;
    }

    /**Begin to build
     * DEPTH_ = 16
     * BRANCH_ = 32
     * ITER = 10
     */
    n_clusters = n_clusters_;

    /**par, the parameter used for doing hierarchical clustering needed by flann's kmean*/
    KMeansParams par;
    par["branching"] = n_clusters_;
    par["iterations"] = iter_;

    /**number of subspace = depth
     */
    int dimen = dcols / n_subspaces;
    cout << "sudspace dimension = " << dimen << endl;

    /**This is the place where it builds index in each subspace
     * Each mat is: 10164 * 8
     * I could use pthread to do this in parallel
     * */
    for (int i = 0; i < n_subspaces; ++i) {
        // If I want to use numa, this could be the place

        /**mat holds the data for each subspace, its shape is number of features in the
         * first image times (128/n_subspaces)*/
        Matrix<float> mat(ddata + dimen * i, drows, dimen, dcols * sizeof(float));

        KMeans<L2<float>> kmeans(mat, par);

        /**see buildIndexImpl in sdc_kmeans.h*/
        kmeans.buildIndex();

        Cluster *clu = kmeans.getCluster();
        clu->start = dimen * i;
        clusters.push_back(clu);
    }

    /**Why sort?*/
    std::sort(clusters.begin(), clusters.end(), compare);

    /**csize is the number of total number of hierarchical clusters,
     * it equals the number of subspaces, which is 16*/
    int csize = clusters.size();

    beLongTo = new unsigned char[drows * csize];
    for (int i = 0; i < csize; ++i) {
        unsigned char *blt = beLongTo + i;
        int *bel = clusters[i]->beLongTo;

        for (int j = 0; j < drows; ++j) {
            *blt = bel[j];
            blt += csize;
        }
    }
}

void SDCIndex::update(float pre, float precision) {
    float r1 = std::abs(scale_u - SCALE_U);
    float r2 = std::abs(scale_d - SCALE_D);
    float acc = std::abs((precision - pre));
    acc = std::min(acc, 0.15f);
    acc = std::max(acc, 0.05f);
    if (pre > precision) {
        if (r1 > r2) {
            scale_d -= acc;
        } else {
            scale_u += acc;
        }
    } else {
        if (r1 >= r2) {
            scale_d += acc;
        } else {
            scale_u -= acc;
        }
    }
    //	printf("U: %.4f, D: %.4f ", scale_u, scale_d);
}


float *SDCIndex::init(float *query, float *min_distances) {
    int csize = clusters.size();
    if (min_distances == NULL) {
        min_distances = new float[csize * n_clusters];
    }
    for (int j = 0; j < csize; ++j) {
        Cluster *clu = clusters[j];
        float *que = query + clu->start;
        float *min_dists = min_distances + j * n_clusters;
        for (int m = 0; m < n_clusters; ++m) {
            float *cen = clu->centers + m * clu->cols;
            float d = distance_(que, cen, clu->cols);
            float v = d / clu->radiuses[m] / clu->radiuses[m];
            float tmp = 0;
            if (v <= 1) {
                tmp = 1 + scale_d * (v - 1);
            } else {
                tmp = 1 + scale_u * (v - 1);
            }
            d *= tmp * tmp;

            min_dists[m] = d;
        }
    }
    return min_distances;
}


/**@param que has shape 5091 * 128, where 5019 is the number of features in the second image, and 128 is
 * the number of dimensions of each feature
 * @param iindex has shape 5091 * 2
 * @param idist has 5091 * 2
 * @param knn knn = 2
 * */
void SDCIndex::knnSearch(Matrix<float> &que,
                         Matrix<int> &iindex,
                         Matrix<float> &idist,
                         int knn,
                         SearchParams &par) {

    int ssize = que.rows;

#pragma omp parallel num_threads(par.cores)
#pragma omp for schedule(dynamic)
    for (int i = 0; i < ssize; ++i) {

        // Calculate distances to each clusters:
        // Calculate the minimum squared distance:
        float *min_distances = this->init(que[i]);

        // Search the CKD-tree:
        KNNSimpleResultSet<float> results(knn);

        this->explore(que[i], &results, min_distances);

        size_t *ii = new size_t[knn];
        results.copy(ii, idist[i], knn, false);
        int *iin = iindex[i];
        for (int j = 0; j < knn; ++j) {
            iin[j] = ii[j];
        }

        delete[] ii;
        delete[] min_distances;
    }
}

/**@param query one feature of the image
 * @param results
 * @param mdists*/
void SDCIndex::explore(float *query, ResultSet<float> *results, float *mdists) {
    int csize = clusters.size();
    unsigned char *blt = beLongTo;
    float worst = results->worstDist();

    for (int i = 0; i < drows; ++i, blt += csize) {
        float *min_dists = mdists;
        float dist = 0;
        int j = 0;
        while (j < csize) {
            dist += min_dists[blt[j++]];
            min_dists += n_clusters;
            dist += min_dists[blt[j++]];
            min_dists += n_clusters;
        }
        if (dist < worst) {
            float *feature = ddata + i * dcols;
            dist = distance_(feature, query, dcols);
            results->addPoint(dist, i);
            worst = results->worstDist();
        }
    }
}

int SDCIndex::getClusterSize() {
    return clusters.size() * n_clusters;
}




float SDCIndex::estimate(int index, float *mdists) {
    int csize = clusters.size();
    unsigned char *blt = beLongTo + index * csize;
    float dist = 0;
    int j = 0;
    while (j < csize) {
        dist += mdists[blt[j++]];
        mdists += n_clusters;
        dist += mdists[blt[j++]];
        mdists += n_clusters;
    }
    return dist;
}

float SDCIndex::estimate(unsigned char *blt, float *mdists) {
    int csize = clusters.size();
    float dist = 0;
    int j = 0;
    while (j < csize) {
        dist += mdists[blt[j++]];
        mdists += n_clusters;
        dist += mdists[blt[j++]];
        mdists += n_clusters;
    }
    return dist;
}

void SDCIndex::getStatics(int s,
                          int c,
                          int iter,
                          float *results,
                          float *queries,
                          int qrows,
                          int knn) {
    TIMER_T start, end;
    TIMER_READ(start);
    buildIndex(s, c, iter);
    TIMER_READ(end);
    printf("%.1f & ", TIMER_DIFF_SECONDS(start, end));

    int csize = clusters.size();
    float *distances = new float[csize * n_clusters];
    float *estimated = new float[drows];
    float small_average = 0;
    float average_average = 0;
    float standard_average = 0;

    float est_time = 0;
    KNNSimpleResultSet<float> heap(knn);
    size_t *index1 = new size_t[knn];
    float *dist1 = new float[knn];
    size_t *index2 = new size_t[knn];
    float *dist2 = new float[knn];
    int filtered = 0;
    int correct = 0;

    for (int q = 0; q < qrows; ++q) {
        float *query = queries + q * dcols;

        TIMER_READ(start);
        for (int i = 0; i < csize; ++i) {
            Cluster *clu = clusters[i];
            float *que = query + clu->start;
            float *dists = distances + i * n_clusters;
            for (int j = 0; j < n_clusters; ++j) {
                float *cen = clu->centers + j * clu->cols;
                dists[j] = distance_(que, cen, clu->cols);
            }
        }

        unsigned char *blt = beLongTo;
        for (int i = 0; i < drows; ++i, blt += csize) {
            float *min_dists = distances;
            float dist = 0;
            int j = 0;
            while (j < csize) {
                dist += min_dists[blt[j++]];
                min_dists += n_clusters;
            }
            estimated[i] = dist;
        }
        TIMER_READ(end);
        est_time += TIMER_DIFF_SECONDS(start, end);
        for (int i = 0; i < drows; ++i) {
            estimated[i] = sqrt(estimated[i]);
        }

        float *result = results + q * drows;

        int small = 0;
        for (int i = 0; i < drows; ++i) {
            if (estimated[i] <= result[i]) {
                ++small;
            }
        }
        small_average += small * 1.0f / drows;

        float average = 0;
        for (int i = 0; i < drows; ++i) {
            average += abs(result[i] - estimated[i]);
        }
        average /= drows;
        average_average += average;

        float standard = 0;
        for (int i = 0; i < drows; ++i) {
            float tmp = abs(result[i] - estimated[i]);
            tmp -= average;
            standard += tmp * tmp;
        }
        standard /= drows;
        standard = sqrt(standard);
        standard_average += standard;

        heap.clear();
        for (int i = 0; i < drows; ++i) {
            heap.addPoint(result[i], i);
        }
        heap.copy(index1, dist1, knn, false);

        heap.clear();
        for (int i = 0; i < drows; ++i) {
            if (estimated[i] <= heap.worstDist()) {
                heap.addPoint(result[i], i);
            } else {
                ++filtered;
            }
        }
        heap.copy(index2, dist2, knn, false);

        int cur1 = 0, cur2 = 0;
        while (cur1 < knn) {
            if (index1[cur1] != index2[cur2]) {
                ++cur1;
            } else {
                ++cur1;
                ++cur2;
                ++correct;
            }
        }
    }
    small_average /= qrows;
    average_average /= qrows;
    standard_average /= qrows;
    //	printf("%.4f %.2f %.2f\n", small_average, average_average,
    //			standard_average);

    printf("%.1f & ", est_time);
    printf("%.1f & ", average_average);
    printf("%.1f\\%% & ", filtered * 100.0f / drows / qrows);
    printf("%.1f\\%%", correct * 100.0f / knn / qrows);

    delete[] distances;
    delete[] estimated;
    delete[] index1;
    delete[] dist1;
    delete[] index2;
    delete[] dist2;
}
