/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CORE_H
#define CORE_H

namespace cvcore {

// Enable dll imports/exports in case of windows support
#if defined _WIN32 && defined CVCORE_SHARED_LIB
#ifdef CVCORE_EXPORT_SYMBOLS             // Needs to be enabled in case of compiling dll
#define CVCORE_API __declspec(dllexport) // Exports symbols when compiling the library.
#else
#define CVCORE_API __declspec(dllimport) // Imports the symbols when linked with library.
#endif
#else
#define CVCORE_API
#endif

} // namespace cvcore
#endif // CORE_H
