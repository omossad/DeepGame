mkdir build
cd build
#cmake -G "Visual Studio 10 2010" -A x64 ..
cmake -DCMAKE_PREFIX_PATH=C:\Users\omossad\Desktop\codes\DeepGame\torchscript\libtorch\ ..
cmake --build . --config Release