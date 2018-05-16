

#include "common.h"
#include "csv_parser.h"
#include "main.h"
#include <stdio.h>
#include <string>

#define PIPE_FILENO 3

inline void RECORD_PMC() {
    if (write(PIPE_FILENO, "PMC_CMD: read_pmc\n", 18)) {
    }
}

/**./bin/scf_main 5 16 32 0 0 features/_DSC0%s.jpg features/gps-sift-small.csv*/
int main(int argc, char **argv) {
    IMG::ALG = 0;   // use scf algorithm
    IMG::REP = atoi(argv[1]);
    SDCIndex::N_SUBSPACES_ = atoi(argv[2]);
    SDCIndex::N_CLUSTERS_ = atoi(argv[3]);
    SDCIndex::scale_d = 0;
    SDCIndex::scale_u = 0;
    char *format = argv[6];
    char *csv_file = argv[7];

//    printf("format is: %s\n", format);
//    printf("csv_file is: %s\n", csv_file);

    vector<IMG *> imgs;
    TIMER_T runtime[4];

    TIMER_READ(runtime[0]);
    csv_parser csv;

    csv.init(csv_file);
    csv.set_enclosed_char('"', ENCLOSURE_OPTIONAL);
    csv.set_field_term_char(',');
    csv.set_line_term_char('\n');
    csv.get_row();

    printf("\n");

    /**load IMG data*/
    while (csv.has_more_rows()) {
        IMG *img = new IMG();
        /**compose a formatted string, instead of print it out to stdout, it put the string in to the buffer.*/
        snprintf(img->name, 50, format, csv.get_row()[1].c_str());
        img->readImg();
        imgs.push_back(img);
    }
    cout << "Loaded " << imgs.size() << " images" << endl;
    printf("\n");

    TIMER_READ(runtime[1]);
    printf("Read images in: %f seconds\n", TIMER_DIFF_SECONDS(runtime[0], runtime[1]));

    /**n_imgs = imgs.size() - 1*/
    unsigned long n_imgs = imgs.size() - 1;
    for (unsigned long i = 0; i < n_imgs; ++i) {
        imgs[i]->buildIndex();
    }
    TIMER_READ(runtime[2]);
    printf("Build indexes in: %f seconds\n", TIMER_DIFF_SECONDS(runtime[1], runtime[2]));

    /**index has the size twice of the number of features in the second image*/
    int *index = new int[imgs[n_imgs]->rows * KNN];
    float *dist = new float[imgs[n_imgs]->rows * KNN];
    uint64_t start, end, time = 0;
    int threads = 64;

    for (int k = 0; k < n_imgs; ++k) {
        printf("\nImage %d", k);
        RECORD_PMC();
        start = tick();

        imgs[k]->march(*imgs[n_imgs], index, dist, KNN, threads);

        end = tick();
        printf(", %d threads, tick: ", threads);
        printf("%llu ", end - start);

        fflush(0);
        RECORD_PMC();

        sleep(2);
        printf("\n");
    }
    delete[] index;
    delete[] dist;

    TIMER_READ(runtime[3]);
    printf("\nKnn searching cost: %f seconds\n", TIMER_DIFF_SECONDS(runtime[2], runtime[3]));

    return 0;
}
