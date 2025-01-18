echo "#################"
echo "    COMPILING    "
echo "#################"

g++ -Wall -fopenmp -std=c++23 -O3 src/*.cpp -o network

echo "#################"
echo "     RUNNING     "
echo "#################"
./network
