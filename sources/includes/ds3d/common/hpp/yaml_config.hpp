/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef DS3D_COMMON_HPP_YAML_CONFIG_HPP
#define DS3D_COMMON_HPP_YAML_CONFIG_HPP

#include <ds3d/common/common.h>
#include <ds3d/common/config.h>
#include <ds3d/common/idatatype.h>
#include <ds3d/common/type_trait.h>
#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>

namespace ds3d {
namespace config {

ErrCode inline CatchYamlCall(std::function<ErrCode()> f)
{
    ErrCode code = ErrCode::kGood;
    DS3D_TRY { code = f(); }
    DS3D_CATCH_ERROR(Exception, ErrCode::kConfig, "parse config error")
    DS3D_CATCH_ERROR(std::runtime_error, ErrCode::kConfig, "parse config failed")
    DS3D_CATCH_ERROR(std::exception, ErrCode::kConfig, "parse config failed")
    DS3D_CATCH_ANY(ErrCode::kConfig, "parse config failed")
    return code;
}

template <class F, typename... Args>
ErrCode CatchConfigCall(F f, Args &&... args)
{
    ErrCode code = ErrCode::kGood;
    DS3D_TRY { code = f(std::forward<Args>(args)...); }
    DS3D_CATCH_ERROR(Exception, ErrCode::kConfig, "parse config error")
    DS3D_CATCH_ERROR(std::runtime_error, ErrCode::kConfig, "parse config failed")
    DS3D_CATCH_ERROR(std::exception, ErrCode::kConfig, "parse config failed")
    DS3D_CATCH_ANY(ErrCode::kConfig, "parse config failed")
    return code;
}

inline ErrCode parseComponentConfig(const std::string &yamlComp,
                                    const std::string &path,
                                    ComponentConfig &config)
{
    config.rawContent = yamlComp;
    config.filePath = path;

    YAML::Node node = YAML::Load(yamlComp);
    config.name = node["name"].as<std::string>();
    DS3D_FAILED_RETURN(!config.name.empty(), ErrCode::kConfig, "name not found in config");
    std::string type = node["type"].as<std::string>();
    DS3D_FAILED_RETURN(!type.empty(), ErrCode::kConfig, "component type must be specified");
    config.type = componentType(type);
    DS3D_FAILED_RETURN(config.type != ComponentType::kNone, ErrCode::kConfig,
                       "type not found in config");
    auto inCapsNode = node["in_caps"];
    if (inCapsNode) {
        config.gstInCaps = inCapsNode.as<std::string>();
    }
    auto outCapsNode = node["out_caps"];
    if (outCapsNode) {
        config.gstOutCaps = outCapsNode.as<std::string>();
    }

    if (node["custom_lib_path"]) {
        config.customLibPath = node["custom_lib_path"].as<std::string>();
        DS3D_FAILED_RETURN(!config.customLibPath.empty(), ErrCode::kConfig,
                           "custom_lib_path not found in config");
    }
    if (node["custom_create_function"]) {
        config.customCreateFunction = node["custom_create_function"].as<std::string>();
        DS3D_FAILED_RETURN(!config.customCreateFunction.empty(), ErrCode::kConfig,
                           "custom_create_function not found in config");
    }
    //
    if (node["config_body"]) {
        YAML::Emitter body;
        body << node["config_body"];
        DS3D_THROW_ERROR(body.good(), ErrCode::kConfig,
                         "config_body error in config, yaml error: " + body.GetLastError());
        config.configBody = body.c_str();
    }
    return ErrCode::kGood;
}

inline ErrCode parseFullConfig(const std::string &yamlDoc,
                               const std::string &path,
                               std::vector<ComponentConfig> &all)
{
    auto nodes = YAML::LoadAll(yamlDoc);
    for (const auto &doc : nodes) {
        if (!doc) {
            continue;
        }
        YAML::Emitter content;
        content << doc;
        DS3D_THROW_ERROR(content.good(), ErrCode::kConfig,
                         "component error in config, yaml error: " + content.GetLastError());
        ComponentConfig config;
        DS3D_ERROR_RETURN(parseComponentConfig(content.c_str(), path, config),
                          "parsing a component failed in config");
        all.push_back(config);
    }
    return ErrCode::kGood;
}

} // namespace config
} // namespace ds3d

#endif // DS3D_COMMON_HPP_YAML_CONFIG_HPP