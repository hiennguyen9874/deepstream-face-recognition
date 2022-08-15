# Select install dir using provided version.
SET(NVDS_INSTALL_DIR /opt/nvidia/deepstream/deepstream)

# List all libraries in deepstream.
SET(NVDS_LIBS
  nvdsgst_meta
  nvdsgst_helper
  nvdsgst_smartrecord
  nvds_meta
  nvds_msgbroker
  nvds_utils
  nvds_batch_jpegenc
)

# Find all libraries in list.
foreach(LIB ${NVDS_LIBS})
  find_library(${LIB}_PATH NAMES ${LIB} PATHS ${NVDS_INSTALL_DIR}/lib)
  if(${LIB}_PATH)
    set(NVDS_LIBRARIES ${NVDS_LIBRARIES} ${${LIB}_PATH})
  else()
    message(FATAL ERROR " Unable to find lib: ${LIB}")
    set(NVDS_LIBRARIES FALSE)
    break()
  endif()
endforeach()

# Find include directories.
find_path(NVDS_INCLUDE_DIRS
  NAMES
    nvds_version.h
  HINTS
    ${NVDS_INSTALL_DIR}/sources/includes
)

# Check libraries and includes.
if (NVDS_LIBRARIES AND NVDS_INCLUDE_DIRS)
  set(NVDS_FOUND TRUE)
else()
  message(FATAL ERROR " Unable to find NVDS")
endif()

