#include "argparse.h"

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--num_cluster or -k <an integer specifying the number of clusters>" << std::endl;
        std::cout << "\t--dim or -d <an integer specifying the dimension of the data>" << std::endl;
        std::cout << "\t--input_filename or -i <a string specifying the input filename>" << std::endl;
        std::cout << "\t--max_num_iter or -m <an integer specifying the maximum number of iterations>" << std::endl;
        std::cout << "\t--threshold or -t <a double specifying the threshold for convergence>" << std::endl;
        std::cout << "\t--seed or -s <an integer specifying the seed for rand()>"  << std::endl;
        std::cout << "\t[Optional]--centroid_output or -c <a flag to control whether output the centroids or labels>" << std::endl;
        exit(0);
    }

    // Initialize opts
    opts->centroid_output = false;



    struct option l_opts[] = {
        {"num_cluster", required_argument, NULL, 'k'},
        {"dim", required_argument, NULL, 'd'},
        {"input_filename", required_argument, NULL, 'i'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"centroid_output", no_argument, NULL, 'c'},
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:c", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->num_cluster = atoi((char *)optarg);
            break;
        case 'd':
            opts->dim = atoi((char *)optarg);
            break;
        case 'i':
            opts->input_filename = (char *)optarg;
            break;
        case 'm':
            opts->max_num_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            break;
        case 'c':
            opts->centroid_output = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}

