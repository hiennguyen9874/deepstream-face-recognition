#include <assert.h>
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

static bool label_ready = false;
std::vector<std::string> labels;

bool fetch_labels(std::string const path)
{
    std::ifstream flabel;

    if (!label_ready) {
        flabel.open(path);

        if (!flabel.is_open()) {
            std::cout << "open dictionary file failed." << std::endl;
            return false;
        }

        while (!flabel.eof()) {
            std::string strLineAnsi;

            if (getline(flabel, strLineAnsi)) {
                labels.push_back(strLineAnsi);
            }
        }
        label_ready = true;

        flabel.close();
    }
    return true;
}

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferClassiferParseCustomAttributeRecognition(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString);

extern "C" bool NvDsInferClassiferParseCustomAttributeRecognition(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString)
{
    if (!fetch_labels("./samples/models/Secondary_Attribute/attribute_labels.txt"))
        return false;

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    /* Get the number of attributes supported by the classifier. */
    // unsigned int numAttributes = outputLayersInfo[0].dims.numElements;

    const uint numAttributes = layer.inferDims.d[0];

    const float *probs = (const float *)(layer.buffer);

    for (unsigned int i = 0; i < numAttributes; i++) {
        NvDsInferAttribute attr;

        attr.attributeIndex = i;
        attr.attributeValue = 1;
        attr.attributeConfidence = probs[i];
        attr.attributeLabel = strdup(labels[i].c_str());

        attrList.push_back(attr);
    }

    descString.append(" ");
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassiferParseCustomAttributeRecognition);
