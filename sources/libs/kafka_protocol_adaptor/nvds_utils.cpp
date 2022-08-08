/*
 * Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

// This source file presents some common functions/utils used by adapter libraries

#include "nvds_utils.h"

#include <glib.h>
#include <openssl/sha.h>
#include <string.h>

#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "nvds_logger.h"
#include "nvds_msgapi.h"

using namespace std;

#define CONFIG_GROUP_MSG_BROKER "message-broker"

/*
 * Release memory allocated for gobjects
 */
void free_gobjs(GKeyFile *gcfg_file, GError *error, gchar **keys, gchar *key_name)
{
    if (keys != NULL)
        g_strfreev(keys);
    if (error != NULL)
        g_error_free(error);
    if (gcfg_file != NULL)
        g_key_file_free(gcfg_file);
    if (key_name != NULL)
        g_free(key_name);
}

/*
 * Internal function to verify if the config string value is a quoted string
 * config key = "value"
 * If so, strip beginning and ending quote
 */
NvDsMsgApiErrorType strip_quote(char *cfg_value, const char *cfg_key, const char *LOG_CAT)
{
    // If value is not empty
    if (strcmp(cfg_value, "")) {
        size_t conflen = strlen(cfg_value);

        // remove "". (string length needs to be at least 2)
        // Could use g_shell_unquote but it might have other side effects
        if ((conflen < 3) || (cfg_value[0] != '"') || (cfg_value[conflen - 1] != '"')) {
            nvds_log(LOG_CAT, LOG_ERR, "invalid key=value format. Must Start and end with \"\"\n",
                     cfg_key);
            return NVDS_MSGAPI_ERR;
        } else {
            string res(cfg_value);
            res.erase(0, 1);             // erase first "
            res.erase(res.length() - 1); // erase last "
            strcpy(cfg_value, res.c_str());
            nvds_log(LOG_CAT, LOG_INFO, "cfg setting %s = %s\n", cfg_key, cfg_value);
        }
    }
    return NVDS_MSGAPI_OK;
}

/*
 *Function to open a file and search for a config value for a config key
 *If found, place the result in cfg_val
 */
NvDsMsgApiErrorType fetch_config_value(char *config_path,
                                       const char *cfg_key,
                                       char *cfg_val,
                                       int len,
                                       const char *LOG_CAT)
{
    // iterate over the config params to set one by one
    GKeyFile *key_file = g_key_file_new();
    gchar **keys = NULL;
    gchar **key = NULL;
    GError *error = NULL;

    if (!config_path) {
        nvds_log(LOG_CAT, LOG_ERR,
                 "Error parsing config file. config file path pointer cant be NULL");
        return NVDS_MSGAPI_ERR;
    }

    if (!g_key_file_load_from_file(key_file, config_path, G_KEY_FILE_NONE, &error)) {
        nvds_log(LOG_CAT, LOG_ERR, "unable to load config file at path[%s]; error message = %s\n",
                 config_path, error->message);
        return NVDS_MSGAPI_ERR;
    }

    keys = g_key_file_get_keys(key_file, CONFIG_GROUP_MSG_BROKER, NULL, &error);
    if (error) {
        nvds_log(LOG_CAT, LOG_ERR, "config group[%s] in cfg not found. Error:%s\n",
                 CONFIG_GROUP_MSG_BROKER, error->message);
        return NVDS_MSGAPI_ERR;
    }
    gchar *str = NULL;
    for (key = keys; *key; key++) {
        if (!g_strcmp0(*key, cfg_key)) {
            str = g_key_file_get_string(key_file, CONFIG_GROUP_MSG_BROKER, cfg_key, &error);

            if (error) {
                nvds_log(LOG_CAT, LOG_ERR, "Error parsing config file. %s\n", error->message);
                free_gobjs(key_file, error, keys, str);
                return NVDS_MSGAPI_ERR;
            }
            if (len < (int)strlen(str) + 1) {
                nvds_log(LOG_CAT, LOG_ERR,
                         "Output string capacity is not sufficient to hold config value. Should be "
                         "atleast %d",
                         strlen(str) + 1);
                free_gobjs(key_file, error, keys, str);
                return NVDS_MSGAPI_ERR;
            }
            strcpy(cfg_val, str);
            break;
        }
    }
    free_gobjs(key_file, error, keys, str);
    return NVDS_MSGAPI_OK;
}

/*
 *Generate hash value of a string using SHA256
 */
string generate_sha256_hash(string str)
{
    unsigned char hashval[SHA256_DIGEST_LENGTH];
    int len = SHA256_DIGEST_LENGTH * 2 + 1;
    char res[len];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.c_str(), str.length());
    SHA256_Final(hashval, &sha256);
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        sprintf(res + (i * 2), "%02x", hashval[i]);
    }
    return string(res);
}

/*
 * remove leading & trailing whitespaces from a string
 */
string trim(string str)
{
    auto low = str.find_first_not_of(" \t");
    if (low == string::npos)
        return "";
    auto high = str.find_last_not_of(" \t");
    return str.substr(low, high - low + 1);
}

/*
 * Sort string of format key=value;key1=value1
 * ex: str = "pq=89;xyz=33;abc=123;"
 * output  = "abc=123;pq=89;xyz=33"
 */
string sort_key_value_pairs(string str)
{
    map<string, string> mymap;
    istringstream iss(str);
    string kv_pair;
    while (getline(iss, kv_pair, ';')) {
        auto pos = kv_pair.find('=');
        string key = trim(kv_pair.substr(0, pos));
        string val = trim(kv_pair.substr(pos + 1));
        if (key != "")
            mymap[key] = val;
    }
    string res = "";
    for (auto i : mymap) {
        res += i.first + "=" + i.second;
    }
    return res;
}
