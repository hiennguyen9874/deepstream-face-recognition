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

#ifndef _DS3D_COMMON_ABI_DATA_OBJ__H
#define _DS3D_COMMON_ABI_DATA_OBJ__H

#include <ds3d/common/common.h>
#include <ds3d/common/idatatype.h>
#include <ds3d/common/type_trait.h>

namespace ds3d {

struct abiRefObj {
    virtual void destroy() = 0;
    virtual abiRefObj *refCopy_i() const = 0;
    virtual ~abiRefObj() = default;
};

template <class T>
struct abiRefT : public abiRefObj {
    virtual T *data() const = 0;
    abiRefT *refCopy() const { return dynamic_cast<abiRefT *>(refCopy_i()); }
};

typedef abiRefT<void> abiRefAny;

class abiDataMap {
public:
    abiDataMap() = default;
    virtual ~abiDataMap() = default;
    virtual ErrCode setBuf_i(const char *key, TIdType tid, abiRefAny *data) = 0;
    virtual ErrCode getBuf_i(const char *key, TIdType tid, const abiRefAny *&data) const = 0;
    virtual ErrCode removeBuf_i(const char *key) = 0;
    virtual bool has_i(const char *key) const = 0;
    virtual ErrCode clear_i() = 0;
    DS3D_DISABLE_CLASS_COPY(abiDataMap);
};

typedef abiRefT<abiDataMap> abiRefDataMap;

// define Callback abi interfaces
template <typename... Args>
struct abiCallBackT : public abiRefObj {
    using CppFunc = std::function<void(Args...)>;
    virtual void notify(Args... args) = 0;
    abiCallBackT *refCopy() const { return dynamic_cast<abiCallBackT *>(refCopy_i()); }
};

typedef abiCallBackT<ErrCode, const char *> abiErrorCB;
typedef abiCallBackT<ErrCode, const abiRefDataMap *> abiOnDataCB;

} // namespace ds3d

#endif // _DS3D_COMMON_ABI_DATA_OBJ__H