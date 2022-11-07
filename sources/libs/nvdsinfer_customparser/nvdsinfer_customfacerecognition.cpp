#include <assert.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include <codecvt>
#include <cstring>
#include <fstream>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferClassiferParseCustomFaceRecognition(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString);

extern "C" bool NvDsInferClassiferParseCustomFaceRecognition(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString)
{
    static faiss::Index *faiss_index = nullptr;
    static std::vector<std::string> labels;
    static char *lastModified = new char[100];
    static char *curModified = new char[100];
    static int intervalNumber = -1;
    intervalNumber++;

    if (intervalNumber % 10 == 0) {
        struct stat attr;
        stat("./faiss.index", &attr);
        strcpy(curModified, ctime(&attr.st_mtime));
    }

    if (faiss_index == NULL || strcmp(curModified, lastModified) != 0) {
        faiss_index = faiss::read_index("./faiss.index");

        printf("index loaded!\n");

        printf("labels is:\n");
        auto labels_file = std::fstream("./labels.txt");
        std::string line;
        if (labels_file.is_open()) {
            while (std::getline(labels_file, line)) {
                labels.push_back(line);
                printf("\t%s\n", line.c_str());
            }
            printf("\n");

        } else {
            fprintf(stderr, "failed to load labels file\n");
        }
        strcpy(lastModified, curModified);
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    float *buffer = (float *)outputLayersInfo[0].buffer;

    faiss::fvec_renorm_L2(512, 1, buffer);

    long I = 0;
    float D = 0;

    faiss_index->search(1, buffer, 1, &D, &I);

    // std::cout << "I: " << I << " D: " << D << std::endl;

    if (D > classifierThreshold) {
        NvDsInferAttribute attr;

        attr.attributeIndex = static_cast<unsigned int>(I);
        attr.attributeValue = 1;
        attr.attributeConfidence = static_cast<float>(D);
        attr.attributeLabel = strdup(labels[static_cast<std::size_t>(I)].c_str());

        attrList.push_back(attr);

        descString.append(attr.attributeLabel).append(" ");
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassiferParseCustomFaceRecognition);
