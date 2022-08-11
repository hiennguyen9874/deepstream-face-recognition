#include <assert.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <stdio.h>
#include <string.h>

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

    if (faiss_index == NULL) {
        faiss_index = faiss::read_index(
            "/home/jovyan/workspace/deepstreams/deepstream_face_detection/faiss.index");

        printf("index loaded!\n");

        printf("labels is:\n");
        auto labels_file =
            std::fstream("/home/jovyan/workspace/deepstreams/deepstream_face_detection/labels.txt");
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
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    float *buffer = (float *)outputLayersInfo[0].buffer;

    long I = 0;
    float D = 0;

    faiss_index->search(1, buffer, 1, &D, &I);

    if (D > classifierThreshold) {
        NvDsInferAttribute attr;

        attr.attributeIndex = I;
        attr.attributeValue = 1;
        attr.attributeConfidence = D;
        attr.attributeLabel = strdup(labels[static_cast<std::size_t>(I)].c_str());

        attrList.push_back(attr);

        descString.append(attr.attributeLabel).append(" ");
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassiferParseCustomFaceRecognition);
