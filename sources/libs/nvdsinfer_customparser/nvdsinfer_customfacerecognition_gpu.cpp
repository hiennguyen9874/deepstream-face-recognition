#include <assert.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
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
extern "C" bool NvDsInferClassiferParseCustomFaceRecognitionGpu(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString);

extern "C" bool NvDsInferClassiferParseCustomFaceRecognitionGpu(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString)
{
    static faiss::gpu::StandardGpuResources res;
    static faiss::Index *faiss_cpu_index = nullptr;
    static faiss::gpu::GpuIndex *faiss_gpu_index = nullptr;
    static std::vector<std::string> labels;

    if (faiss_cpu_index == NULL) {
        faiss_cpu_index = faiss::read_index("./faiss.index");

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
    }

    if (faiss_gpu_index == NULL) {
        faiss_gpu_index = reinterpret_cast<faiss::gpu::GpuIndex *>(
            faiss::gpu::index_cpu_to_gpu(&res, 0, faiss_cpu_index));
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    float *buffer = (float *)outputLayersInfo[0].buffer;

    faiss::fvec_renorm_L2(512, 1, buffer);

    long I = 0;
    float D = 0;

    faiss_gpu_index->search(1, buffer, 1, &D, &I);

    // std::cout << "I: " << I << " D: " << D << std::endl;

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
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassiferParseCustomFaceRecognitionGpu);
