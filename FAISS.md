# Install faiss

## Openblas

-   `sudo apt update && sudo apt install -y libopenblas-dev`

## Swig

-   Download swig from [www.swig.org/download.html](https://www.swig.org/download.html)
-   `wget https://prdownloads.sourceforge.net/swig/swig-4.1.0.tar.gz`
-   `tar -xzvf swig-4.0.2.tar.gz`
-   `cd swig-4.0.2/`
-   `./configure --prefix=/usr/swig`
-   `sudo make && sudo make install`
-   Add into `.bashrc` or `.zshrc`
    ```
    export SWIG_PATH=/usr/swig/bin
    export PATH=$SWIG_PATH:$PATH
    ```

## Faiss

-   `git clone https://github.com/facebookresearch/faiss.git`
-   `cd faiss`
-   `mkdir build`
-   `cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DPython_EXECUTABLE=$(which python3) -DFAISS_OPT_LEVEL=generic -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON .`
-   `make -C build -j faiss`
-   `make -C build -j swigfaiss`
-   `cd build/faiss/python && python3 setup.py install && cp ./*.so /usr/local/lib/`
-   `sudo make -C build install`
-   Add into `.bashrc` or `.zshrc`:
    `export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH`
